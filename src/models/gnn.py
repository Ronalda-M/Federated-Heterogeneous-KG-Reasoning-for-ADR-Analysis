"""
Relation-aware GNN for typed link prediction over heterogeneous KGs.

Supports two architecture variants:
  - R-GCN  (Schlichtkrull et al., ESWC 2018)
  - HGT    (Hu et al., WWW 2020)

Both follow the update rule:

  h_v^{(l+1)} = sigma(
      W_0^{tau(v)} h_v^{(l)}
      + sum_{r in R} sum_{u in N_r(v)} alpha_{u,v}^{r,l}
        W_r^{(l)} [h_u^{(l)} || phi_e(x_{(u,r,v)})]
  )

with per-node-type residual connections and layer normalisation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureMLP(nn.Module):
    """Embeds per-edge provenance attributes phi_e(x_{(u,r,v)})."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


class RGCNLayer(nn.Module):
    """
    Single R-GCN message-passing layer with relation-specific weights.

    Per-node-type residual connection and layer normalisation are applied
    after aggregation, independently within each node-type partition so that
    nodes with structurally different neighbourhoods are normalised within
    their own type-specific distributions.
    """

    def __init__(
        self,
        node_types: List[str],
        relation_types: List[str],
        hidden_dim: int,
        edge_feat_dim: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.relation_types = relation_types
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # W_0^{tau(v)}: per-type self-loop weight
        self.W_self = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for nt in node_types
        })

        # W_r^{(l)}: relation-specific neighbour weight
        msg_in = hidden_dim + edge_feat_dim
        self.W_rel = nn.ModuleDict({
            rt: nn.Linear(msg_in, hidden_dim, bias=False)
            for rt in relation_types
        })

        # Per-node-type layer normalisation
        self.layer_norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in node_types
        })

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        adj: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_feat: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h:         {node_type: (N_t, D)} node embeddings
            adj:       {relation: (src_idx, dst_idx)} edge index per relation
            edge_feat: {relation: (E_r, F)} optional edge feature tensors

        Returns:
            Updated node embeddings with same structure as h.
        """
        out: Dict[str, torch.Tensor] = {}

        for nt in self.node_types:
            out[nt] = self.W_self[nt](h[nt])

        for rt, (src_idx, dst_idx) in adj.items():
            # determine source and destination node types from relation schema
            src_type, dst_type = self._relation_node_types(rt)

            msg = h[src_type][src_idx]                        # (E, D)

            if edge_feat is not None and rt in edge_feat:
                msg = torch.cat([msg, edge_feat[rt]], dim=-1)  # (E, D+F)

            msg = self.W_rel[rt](msg)                          # (E, D)
            msg = F.gelu(msg)

            # scatter-add into destination
            out[dst_type].index_add_(0, dst_idx, msg)

        # residual + layer-norm per node type
        h_new: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            aggregated = out[nt]
            aggregated = self.dropout(aggregated)
            # residual: add input before returning
            aggregated = aggregated + h[nt]
            aggregated = self.layer_norms[nt](aggregated)
            h_new[nt] = aggregated

        return h_new

    def _relation_node_types(self, relation: str) -> Tuple[str, str]:
        """Override or extend to encode the KG schema."""
        _schema: Dict[str, Tuple[str, str]] = {
            "DRUG_CAUSES_AE":   ("Drug", "AdverseEffect"),
            "HAS_TARGET":       ("Drug", "Target"),
            "ASSOCIATED_WITH":  ("AdverseEffect", "ClinicalFactor"),
            "MENTIONS":         ("MedicalLiterature", "AdverseEffect"),
            "EVIDENCE_FOR":     ("BiomedicalReport", "AdverseEffect"),
        }
        return _schema[relation]


class HGTLayer(nn.Module):
    """
    Single Heterogeneous Graph Transformer layer (Hu et al., WWW 2020).

    Relation-type-aware multi-head attention; per-node-type residual
    and layer normalisation applied after aggregation.
    """

    def __init__(
        self,
        node_types: List[str],
        relation_types: List[str],
        hidden_dim: int,
        num_heads: int = 8,
        edge_feat_dim: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.node_types = node_types
        self.relation_types = relation_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        self.W_K = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types})
        self.W_Q = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types})
        self.W_V = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types})

        # Relation-specific attention bias
        self.W_attn = nn.ModuleDict({rt: nn.Linear(self.head_dim, self.head_dim, bias=False)
                                     for rt in relation_types})

        msg_in = hidden_dim + edge_feat_dim
        self.W_msg = nn.ModuleDict({rt: nn.Linear(msg_in, hidden_dim, bias=False)
                                    for rt in relation_types})

        self.W_out = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types})
        self.layer_norms = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        adj: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_feat: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        B = self.num_heads
        D = self.head_dim

        # Initialise accumulators
        agg: Dict[str, torch.Tensor] = {nt: torch.zeros_like(h[nt]) for nt in self.node_types}
        cnt: Dict[str, torch.Tensor] = {nt: torch.zeros(h[nt].size(0), 1,
                                                         device=h[nt].device)
                                        for nt in self.node_types}

        for rt, (src_idx, dst_idx) in adj.items():
            src_type, dst_type = self._relation_node_types(rt)

            K = self.W_K[src_type](h[src_type][src_idx]).view(-1, B, D)  # (E, B, D)
            Q = self.W_Q[dst_type](h[dst_type][dst_idx]).view(-1, B, D)
            V = self.W_V[src_type](h[src_type][src_idx]).view(-1, B, D)

            # Relation-aware attention
            attn = (Q * self.W_attn[rt](K)).sum(-1) / self.scale  # (E, B)
            attn = torch.softmax(attn, dim=0)                       # normalise over src nodes

            msg = h[src_type][src_idx]
            if edge_feat is not None and rt in edge_feat:
                msg = torch.cat([msg, edge_feat[rt]], dim=-1)
            msg = self.W_msg[rt](msg).view(-1, B, D)                # (E, B, D)

            weighted = (attn.unsqueeze(-1) * msg).view(-1, self.hidden_dim)  # (E, D)
            agg[dst_type].index_add_(0, dst_idx, weighted)
            cnt[dst_type].index_add_(0, dst_idx, torch.ones(src_idx.size(0), 1,
                                                             device=src_idx.device))

        h_new: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            denom = cnt[nt].clamp(min=1.0)
            out = self.W_out[nt](agg[nt] / denom)
            out = self.dropout(out)
            out = out + h[nt]                          # residual per node type
            h_new[nt] = self.layer_norms[nt](out)     # layer-norm per node type

        return h_new

    def _relation_node_types(self, relation: str) -> Tuple[str, str]:
        _schema: Dict[str, Tuple[str, str]] = {
            "DRUG_CAUSES_AE":  ("Drug", "AdverseEffect"),
            "HAS_TARGET":      ("Drug", "Target"),
            "ASSOCIATED_WITH": ("AdverseEffect", "ClinicalFactor"),
            "MENTIONS":        ("MedicalLiterature", "AdverseEffect"),
            "EVIDENCE_FOR":    ("BiomedicalReport", "AdverseEffect"),
        }
        return _schema[relation]


class HeterogeneousGNN(nn.Module):
    """
    Multi-layer heterogeneous GNN stacking either RGCNLayer or HGTLayer.
    Produces final node embeddings z_v for all node types.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.node_types = cfg["node_types"]
        self.relation_types = cfg["relation_types"]
        hidden_dim = cfg["gnn"]["hidden_dim"]
        num_layers = cfg["gnn"]["num_layers"]
        num_heads = cfg["gnn"]["num_heads"]
        dropout = cfg["gnn"]["dropout"]
        arch = cfg["gnn"]["architecture"]
        edge_feat_dim = cfg["decoder"].get("edge_mlp_hidden", 0)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if arch == "rgcn":
                layer = RGCNLayer(
                    node_types=self.node_types,
                    relation_types=self.relation_types,
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_feat_dim,
                    dropout=dropout,
                )
            elif arch == "hgt":
                layer = HGTLayer(
                    node_types=self.node_types,
                    relation_types=self.relation_types,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_feat_dim=edge_feat_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown architecture: {arch}")
            self.layers.append(layer)

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        adj: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_feat: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        for layer in self.layers:
            h = layer(h, adj, edge_feat)
        return h
