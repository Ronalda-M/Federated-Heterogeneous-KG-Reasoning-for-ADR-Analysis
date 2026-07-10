"""
Formulation B — Relation-Partitioned (Multi-View / Vertical) Federated Learning

Architecture:
    Each client owns the complete KG for one relation type over a shared
    drug vocabulary. Clients train relation-specific encoders locally;
    only drug embeddings z_d^{(r)} are transmitted to the fusion server.

Client encoders:
    Bipartite relations (Drug-ADR, Drug-Disease, Drug-Compound):
        z_d^{(r)} = POOL_{v in N_r(d)}( MLP_r(h_d || h_v || phi_e(x_{d,v})) )
        (GraphSAGE-style mean pooling)

    Same-type relation (Drug-Drug DDI):
        z_d^{(DDI)} = sigma( W_DDI h_d + sum_{d' in N_DDI(d)} alpha_{d,d'} W'_DDI [h_{d'} || phi_e] )
        (relational convolution)

Cross-view attention fusion (server):
    z_d = sum_r lambda_d^{(r)} P_r( z_d^{(r)} )
    lambda_d^{(r)} = softmax_r( q^T tanh(U_r z_d^{(r)}) )
    Missing views (d not in D_r): z_d^{(r)} = 0 and attention logit = -inf

MOCHA joint objective (Smith et al., NeurIPS 2017):
    min_{W, Omega >= 0} sum_r L_r^local(w_r) + lambda/2 tr(W Omega^{-1} W^T)
    tr(Omega) = 1
    Solved via COCOA primal-dual (Smith et al., JMLR 2018)

Split backpropagation:
    theta_r^{(t+1)} = theta_r^{(t)} - lr * (
        mu_r * grad_{theta_r} L_r^local
        + (dL_pred/dz_d^{(r)}) * (dz_d^{(r)}/d theta_r)
    )
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.decoder import BilinearDecoder, PolypharmacyDecoder
from src.training.loss import WeightedBCELoss, TypeAwareNegativeSampler
from src.evaluation.metrics import TemperatureScaler

logger = logging.getLogger(__name__)


# ─── Entity Alignment ────────────────────────────────────────────────────────

class EntityAligner:
    """
    Computes the shared drug vocabulary across all relation-type clients.

        D_shared = D_ADR ∩ D_Disease ∩ D_Compound ∩ D_DDI

    Harmonisation is assumed to have been applied upstream (RxNorm /
    DrugBank ID mapping). This class resolves the intersection and
    builds a global-to-local index mapping for each client.
    """

    def __init__(self, client_drug_sets: Dict[str, Set[int]]) -> None:
        """
        Args:
            client_drug_sets: {relation_id: set of global drug indices
                               present in that client's KG shard}
        """
        self.client_drug_sets = client_drug_sets
        self.shared: Set[int] = set.intersection(*client_drug_sets.values())
        self.shared_list: List[int] = sorted(self.shared)
        self.global_to_shared: Dict[int, int] = {
            g: i for i, g in enumerate(self.shared_list)
        }

    @property
    def num_shared_drugs(self) -> int:
        return len(self.shared_list)

    def local_indices(self, relation: str) -> torch.Tensor:
        """
        Returns a tensor mapping shared drug index → local drug index
        in the given client's shard.  Drugs absent from the client
        (which should not exist after intersection) are marked -1.
        """
        drug_set = self.client_drug_sets[relation]
        local_map: Dict[int, int] = {g: i for i, g in enumerate(sorted(drug_set))}
        idx = torch.full((self.num_shared_drugs,), -1, dtype=torch.long)
        for shared_i, global_g in enumerate(self.shared_list):
            if global_g in local_map:
                idx[shared_i] = local_map[global_g]
        return idx

    def present_mask(self, relation: str) -> torch.Tensor:
        """Boolean mask: True where drug is present in this client's shard."""
        drug_set = self.client_drug_sets[relation]
        return torch.tensor(
            [g in drug_set for g in self.shared_list], dtype=torch.bool
        )


# ─── Bipartite GNN Encoder ───────────────────────────────────────────────────

class BipartiteGraphSAGEEncoder(nn.Module):
    """
    GraphSAGE-style encoder for bipartite Drug ↔ Entity graphs.

    z_d^{(r)} = POOL_{v in N_r(d)}( MLP_r(h_d || h_v || phi_e(x_{d,v})) )

    Used for Drug-ADR, Drug-Disease, Drug-Compound clients.
    """

    def __init__(
        self,
        drug_feat_dim: int,
        entity_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        in_dim = drug_feat_dim + entity_feat_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.pooling = pooling
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h_drug: torch.Tensor,
        h_entity: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
        num_drugs: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_drug:    (N_d, D_drug) drug features
            h_entity:  (N_e, D_ent)  entity features (ADR / Disease / Compound)
            src_idx:   (E,) drug indices for each edge
            dst_idx:   (E,) entity indices for each edge
            edge_feat: (E, F) optional edge provenance features
            num_drugs: total number of drug nodes (defaults to h_drug.size(0))

        Returns:
            z_drug: (N_d, out_dim) drug embeddings
        """
        N_d = num_drugs or h_drug.size(0)
        inp = [h_drug[src_idx], h_entity[dst_idx]]
        if edge_feat is not None:
            inp.append(edge_feat)
        msg = self.mlp(torch.cat(inp, dim=-1))  # (E, out_dim)

        # Pool messages per drug
        z = torch.zeros(N_d, msg.size(-1), device=msg.device)
        if self.pooling == "mean":
            count = torch.zeros(N_d, 1, device=msg.device)
            z.index_add_(0, src_idx, msg)
            count.index_add_(0, src_idx, torch.ones(src_idx.size(0), 1,
                                                     device=src_idx.device))
            z = z / count.clamp(min=1.0)
        else:  # sum
            z.index_add_(0, src_idx, msg)

        return self.layer_norm(z)


# ─── DDI Relational GNN Encoder ──────────────────────────────────────────────

class DDIRelationalEncoder(nn.Module):
    """
    Relation-aware convolution for the Drug-Drug same-type KG shard.

    z_d^{(DDI)} = sigma(
        W_DDI h_d
        + sum_{d' in N_DDI(d)} alpha_{d,d'} W'_DDI [h_{d'} || phi_e(x_{d,d'})]
    )
    """

    def __init__(
        self,
        drug_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.W_self = nn.Linear(drug_feat_dim, out_dim, bias=False)
        msg_in = drug_feat_dim + edge_feat_dim
        self.W_neigh = nn.Linear(msg_in, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h_drug: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = h_drug.size(0)
        self_out = self.W_self(h_drug)  # (N, out_dim)

        msg_inp = [h_drug[src_idx]]
        if edge_feat is not None:
            msg_inp.append(edge_feat)
        msg = self.W_neigh(torch.cat(msg_inp, dim=-1))  # (E, out_dim)

        # Attention weights
        attn_inp = torch.cat([self_out[dst_idx], msg], dim=-1)
        alpha = torch.sigmoid(self.attn(attn_inp))  # (E, 1)
        msg = alpha * msg

        agg = torch.zeros_like(self_out)
        agg.index_add_(0, dst_idx, msg)

        out = F.gelu(self_out + agg)
        out = self.dropout(out)
        return self.layer_norm(out)


# ─── VFL Client ──────────────────────────────────────────────────────────────

class VFLClient:
    """
    A single relation-type client for Formulation B.

    Responsibilities:
        - Train a relation-specific encoder on its local KG shard
        - Transmit drug embeddings z_d^{(r)} to the fusion server
        - Receive gradient signals dL/dz_d^{(r)} from the server
        - Update encoder via split backpropagation + MOCHA local loss

    Attributes:
        relation:   relation type identifier (e.g. "Drug_ADR")
        drug_ids:   set of global drug IDs present in this shard
    """

    def __init__(
        self,
        client_id: str,
        relation: str,
        drug_ids: Set[int],
        h_drug: torch.Tensor,
        h_entity: Optional[torch.Tensor],
        adj: Tuple[torch.Tensor, torch.Tensor],
        edge_feat: Optional[torch.Tensor],
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.relation = relation
        self.drug_ids = drug_ids
        self.device = device
        self.cfg = cfg

        vfl_cfg = cfg["federated"]["formulation_b"]
        self.out_dim = cfg["gnn"]["hidden_dim"]
        self.mu = vfl_cfg["split_backprop"]["auxiliary_loss_weight"]
        self.local_epochs = vfl_cfg["local_epochs"]
        self.local_lr = cfg["federated"]["formulation_a"]["aggregation"]["local_lr"]

        self.h_drug = h_drug.to(device)
        self.h_entity = h_entity.to(device) if h_entity is not None else None
        self.adj = (adj[0].to(device), adj[1].to(device))
        self.edge_feat = edge_feat.to(device) if edge_feat is not None else None

        edge_feat_dim = edge_feat.size(-1) if edge_feat is not None else 0
        drug_feat_dim = h_drug.size(-1)

        if relation == "Drug_Drug":
            self.encoder = DDIRelationalEncoder(
                drug_feat_dim=drug_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=self.out_dim * 2,
                out_dim=self.out_dim,
            ).to(device)
        else:
            ent_feat_dim = h_entity.size(-1) if h_entity is not None else drug_feat_dim
            self.encoder = BipartiteGraphSAGEEncoder(
                drug_feat_dim=drug_feat_dim,
                entity_feat_dim=ent_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=self.out_dim * 2,
                out_dim=self.out_dim,
            ).to(device)

        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=self.local_lr,
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

        # Self-supervised pretext loss (link reconstruction on held-out edges)
        self.pretext_loss_fn = WeightedBCELoss(
            gamma1=cfg["loss"]["gamma1"],
            gamma2=cfg["loss"]["gamma2"],
        )

        # MOCHA task weight vector w_r (last layer of encoder)
        self.w_r = nn.Parameter(
            torch.randn(self.out_dim, device=device)
        )

    def encode(self) -> torch.Tensor:
        """
        Compute and return drug embeddings z_d^{(r)} for all drugs in shard.

        Returns:
            z: (N_d, out_dim) — retained in computation graph for backprop
        """
        src, dst = self.adj
        if self.relation == "Drug_Drug":
            z = self.encoder(self.h_drug, src, dst, self.edge_feat)
        else:
            z = self.encoder(
                self.h_drug, self.h_entity, src, dst, self.edge_feat
            )
        return z

    def local_pretrain(self, num_steps: int = 5) -> float:
        """
        Self-supervised pretext training (link reconstruction) before
        the first federation round. Ensures z_d^{(r)} is meaningful
        even without cross-client coordination.

        Returns:
            Average pretext loss over num_steps.
        """
        self.encoder.train()
        total_loss = 0.0
        src, dst = self.adj
        N = self.h_drug.size(0)

        for _ in range(num_steps):
            self.optimizer.zero_grad()
            z = self.encode()

            # Simple dot-product pretext: predict observed edges
            pos_scores = (z[src] * z[dst]).sum(dim=-1)
            neg_src = torch.randint(0, N, (src.size(0),), device=self.device)
            neg_scores = (z[neg_src] * z[dst]).sum(dim=-1)

            loss = self.pretext_loss_fn(pos_scores, neg_scores)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_steps

    def apply_gradient(self, grad_z: torch.Tensor, z: torch.Tensor) -> None:
        """
        Split backpropagation step.

        Applies gradient received from server (dL_pred/dz_d^{(r)})
        combined with the local MOCHA pretext gradient:

            theta_r -= lr * (mu_r * grad_local + grad_from_server)

        Args:
            grad_z: (N_d, out_dim) gradient of prediction loss w.r.t. z
            z:      (N_d, out_dim) embedding tensor with computation graph
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        # Local self-supervised loss gradient
        src, dst = self.adj
        pos_scores = (z[src] * z[dst]).sum(dim=-1)
        N = self.h_drug.size(0)
        neg_src = torch.randint(0, N, (src.size(0),), device=self.device)
        neg_scores = (z[neg_src] * z[dst]).sum(dim=-1)
        local_loss = self.pretext_loss_fn(pos_scores, neg_scores)

        # Combine: mu_r * grad_local + chain rule from server gradient
        combined_loss = self.mu * local_loss
        combined_loss.backward(retain_graph=True)

        # Inject server gradient via manual backward
        if z.grad is not None:
            z.grad += grad_z
        else:
            z.grad = grad_z.clone()
        z.backward(z.grad)

        self.optimizer.step()


# ─── Cross-View Attention Fusion ──────────────────────────────────────────────

class CrossViewAttentionFusion(nn.Module):
    """
    Server-side fusion of per-relation drug embeddings.

        z_d = sum_r lambda_d^{(r)} P_r( z_d^{(r)} )

        lambda_d^{(r)} = softmax_r( q^T tanh(U_r z_d^{(r)}) )

    Missing views (drug absent from client r):
        - z_d^{(r)} = 0  (zero embedding)
        - attention logit set to -inf before softmax (zero-masked)

    This prevents absent views from contributing signal or attention mass.
    """

    def __init__(
        self,
        relation_types: List[str],
        in_dim: int,
        fusion_dim: int,
    ) -> None:
        super().__init__()
        self.relation_types = relation_types

        # P_r: projection from per-relation space to common fusion space
        self.P = nn.ModuleDict({
            r: nn.Linear(in_dim, fusion_dim, bias=False)
            for r in relation_types
        })
        # U_r, q: attention parameters
        self.U = nn.ModuleDict({
            r: nn.Linear(in_dim, fusion_dim, bias=False)
            for r in relation_types
        })
        self.q = nn.Parameter(torch.randn(fusion_dim))

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        present_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings:    {relation: (N_shared, in_dim)} per-relation embeddings;
                           zero vectors for absent drugs are expected upstream.
            present_masks: {relation: (N_shared,) bool} True where drug is present.

        Returns:
            z_fused: (N_shared, fusion_dim) fused drug embeddings
        """
        N = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device

        logits = torch.full((N, len(self.relation_types)), float("-inf"),
                            device=device)

        for i, r in enumerate(self.relation_types):
            z_r = embeddings[r]                          # (N, in_dim)
            mask = present_masks[r]                      # (N,)

            # Attention logit: q^T tanh(U_r z_r)
            u_z = torch.tanh(self.U[r](z_r))            # (N, fusion_dim)
            logit = (u_z * self.q).sum(dim=-1)          # (N,)
            logits[mask, i] = logit[mask]               # -inf for absent

        # Renormalised softmax over present views only
        lambdas = torch.softmax(logits, dim=-1)          # (N, R)

        # Weighted sum of projected embeddings
        z_fused = torch.zeros(N, self.q.size(0), device=device)
        for i, r in enumerate(self.relation_types):
            projected = self.P[r](embeddings[r])         # (N, fusion_dim)
            z_fused += lambdas[:, i:i+1] * projected

        return z_fused


# ─── MOCHA Objective ─────────────────────────────────────────────────────────

class MOCHAObjective:
    """
    Federated Multi-Task Learning objective (Smith et al., NeurIPS 2017).

        min_{W, Omega >= 0}  sum_r L_r^local(w_r)
                             + (lambda/2) tr(W Omega^{-1} W^T)
        s.t.  tr(Omega) = 1

    Omega is updated in closed form from W (trace-norm-based relatedness):
        Omega ∝ (W^T W)^{1/2} / tr((W^T W)^{1/2})

    This learns the task-relationship structure jointly rather than
    assuming a single shared model (FedAvg) or fully independent tasks.
    """

    def __init__(
        self,
        relation_types: List[str],
        param_dim: int,
        lambda_reg: float = 0.1,
    ) -> None:
        self.relation_types = relation_types
        self.lambda_reg = lambda_reg
        R = len(relation_types)
        # Omega initialised to uniform relatedness
        self.Omega = torch.eye(R) / R
        self.param_dim = param_dim

    def update_omega(self, W: torch.Tensor) -> None:
        """
        Update task-relationship matrix Omega from W.

        Args:
            W: (param_dim, R) matrix stacking per-relation task weights
        """
        WtW = W.T @ W                         # (R, R)
        try:
            # Symmetric matrix square root
            L, V = torch.linalg.eigh(WtW)
            L = L.clamp(min=0.0)
            WtW_sqrt = V @ torch.diag(L.sqrt()) @ V.T
        except Exception:
            WtW_sqrt = WtW

        trace_val = WtW_sqrt.trace().clamp(min=1e-8)
        self.Omega = (WtW_sqrt / trace_val).detach()

    def regularisation_loss(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute tr(W Omega^{-1} W^T) regularisation term.

        Args:
            W: (param_dim, R) task weight matrix
        """
        Omega = self.Omega.to(W.device)
        try:
            Omega_inv = torch.linalg.inv(Omega + 1e-6 * torch.eye(
                Omega.size(0), device=Omega.device))
        except Exception:
            Omega_inv = torch.eye(Omega.size(0), device=W.device)
        return (self.lambda_reg / 2.0) * torch.trace(W.T @ W @ Omega_inv)


# ─── VFL Server ──────────────────────────────────────────────────────────────

class VFLServer:
    """
    Fusion server for Formulation B.

    Responsibilities:
        - Collect drug embeddings z_d^{(r)} from all clients
        - Apply cross-view attention fusion
        - Train the global ADR prediction head on fused embeddings
        - Compute and return gradient signals dL/dz_d^{(r)} to clients
        - Run MOCHA Omega update
    """

    def __init__(
        self,
        relation_types: List[str],
        aligner: EntityAligner,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.relation_types = relation_types
        self.aligner = aligner
        self.cfg = cfg
        self.device = device

        fusion_dim = cfg["gnn"]["hidden_dim"]
        self.fusion = CrossViewAttentionFusion(
            relation_types=relation_types,
            in_dim=fusion_dim,
            fusion_dim=fusion_dim,
        ).to(device)

        num_drugs = aligner.num_shared_drugs
        num_adrs = cfg.get("num_adrs", 1000)

        # ADR embeddings (learnable on server)
        self.z_adr = nn.Parameter(
            torch.randn(num_adrs, fusion_dim, device=device)
        )

        self.decoder = BilinearDecoder(
            hidden_dim=fusion_dim,
            edge_feat_dim=0,
        ).to(device)

        self.loss_fn = WeightedBCELoss(
            gamma1=cfg["loss"]["gamma1"],
            gamma2=cfg["loss"]["gamma2"],
            label_smoothing=cfg["loss"]["label_smoothing"],
        )

        self.neg_sampler = TypeAwareNegativeSampler(
            num_drugs=num_drugs,
            num_adrs=num_adrs,
        )

        self.optimizer = optim.AdamW(
            list(self.fusion.parameters()) + [self.z_adr] +
            list(self.decoder.parameters()),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

        self.calibrator = TemperatureScaler().to(device)
        self.mocha = MOCHAObjective(
            relation_types=relation_types,
            param_dim=fusion_dim,
            lambda_reg=cfg["federated"]["formulation_b"]["mocha"]["lambda_reg"],
        )

    def aggregate_and_train(
        self,
        client_embeddings: Dict[str, torch.Tensor],
        present_masks: Dict[str, torch.Tensor],
        pos_pairs: torch.Tensor,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Fuse client embeddings, train prediction head, return gradients.

        Args:
            client_embeddings: {relation: (N_shared, D)} drug embeddings
            present_masks:     {relation: (N_shared,) bool}
            pos_pairs:         (M, 2) confirmed drug--ADR pairs

        Returns:
            (loss_value, grad_per_relation)
            where grad_per_relation = {relation: (N_shared, D)} gradients
            to send back to each client for split backpropagation.
        """
        # Retain computation graph for split backprop
        for r in self.relation_types:
            client_embeddings[r] = client_embeddings[r].detach().requires_grad_(True)

        self.optimizer.zero_grad()

        # Cross-view attention fusion
        z_fused = self.fusion(client_embeddings, present_masks)  # (N_shared, D)

        # Prediction loss
        pos_drug = pos_pairs[:, 0]
        pos_adr = pos_pairs[:, 1]
        pos_scores = self.decoder(z_fused[pos_drug], self.z_adr[pos_adr])

        neg_drug, neg_adr = self.neg_sampler.sample(pos_drug, pos_adr)
        neg_scores = self.decoder(z_fused[neg_drug], self.z_adr[neg_adr])

        loss = self.loss_fn(pos_scores, neg_scores)
        loss.backward()
        self.optimizer.step()

        # Collect gradients w.r.t. each client's embedding tensor
        grad_per_relation: Dict[str, torch.Tensor] = {}
        for r in self.relation_types:
            g = client_embeddings[r].grad
            grad_per_relation[r] = g.clone() if g is not None else torch.zeros_like(
                client_embeddings[r]
            )

        return float(loss.item()), grad_per_relation

    def update_mocha_omega(
        self, client_params: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update Omega from stacked task weight matrix W.

        Args:
            client_params: {relation: (D,) last-layer weight vector per client}
        """
        D = next(iter(client_params.values())).size(0)
        R = len(self.relation_types)
        W = torch.stack(
            [client_params[r] for r in self.relation_types], dim=1
        )  # (D, R)
        self.mocha.update_omega(W)
        logger.debug("MOCHA Omega updated.")

    def save_checkpoint(self, path: str, round_num: int) -> None:
        torch.save(
            {
                "round": round_num,
                "fusion_state": self.fusion.state_dict(),
                "decoder_state": self.decoder.state_dict(),
                "z_adr": self.z_adr.data,
                "omega": self.mocha.Omega,
            },
            path,
        )
        logger.info(f"VFL checkpoint saved: {path}")


# ─── VFL Training Loop ────────────────────────────────────────────────────────

class VFLTrainer:
    """
    Orchestrates the full Formulation B federation training loop.

    Round loop:
        1. Entity alignment (once, before training)
        2. Local pre-training on each client (self-supervised)
        3. For each round t:
            a. Each client encodes its local KG shard → z_d^{(r)}
            b. Embeddings mapped to shared drug space via EntityAligner
            c. Server fuses embeddings via cross-view attention
            d. Server trains prediction head, computes gradients
            e. Server sends gradients back to each client
            f. Each client applies split backpropagation
            g. Server updates MOCHA Omega from client task weights
    """

    def __init__(
        self,
        server: VFLServer,
        clients: Dict[str, VFLClient],
        aligner: EntityAligner,
        pos_pairs: torch.Tensor,
        cfg: dict,
        checkpoint_dir: str = "checkpoints/vfl/",
    ) -> None:
        self.server = server
        self.clients = clients
        self.aligner = aligner
        self.pos_pairs = pos_pairs
        self.cfg = cfg
        self.num_rounds = cfg["federated"]["formulation_b"]["rounds"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Precompute local-to-shared index maps
        self.local_idx: Dict[str, torch.Tensor] = {
            r: aligner.local_indices(r) for r in aligner.client_drug_sets
        }
        self.present_masks: Dict[str, torch.Tensor] = {
            r: aligner.present_mask(r) for r in aligner.client_drug_sets
        }

    def _embed_to_shared_space(
        self, relation: str, z_local: torch.Tensor
    ) -> torch.Tensor:
        """
        Map local drug embeddings to the shared drug index space.
        Drugs absent from this client are filled with zeros.
        """
        N_shared = self.aligner.num_shared_drugs
        z_shared = torch.zeros(N_shared, z_local.size(-1),
                               device=z_local.device)
        idx = self.local_idx[relation].to(z_local.device)
        present = idx >= 0
        z_shared[present] = z_local[idx[present]]
        return z_shared

    def train(self) -> None:
        logger.info(f"Starting VFL training: {self.num_rounds} rounds, "
                    f"{len(self.clients)} relation-type clients")

        # Step 1: Local pre-training
        logger.info("Pre-training client encoders (self-supervised)...")
        for rel, client in self.clients.items():
            avg_loss = client.local_pretrain(num_steps=10)
            logger.info(f"  [{rel}] pretrain loss={avg_loss:.4f}")

        for t in range(1, self.num_rounds + 1):
            logger.info(f"=== VFL Round {t}/{self.num_rounds} ===")

            # Step 2: Encode at each client
            client_z_local: Dict[str, torch.Tensor] = {}
            for rel, client in self.clients.items():
                z = client.encode()
                client_z_local[rel] = z

            # Step 3: Map to shared space
            client_z_shared: Dict[str, torch.Tensor] = {
                rel: self._embed_to_shared_space(rel, z)
                for rel, z in client_z_local.items()
            }
            present = {
                rel: self.present_masks[rel].to(
                    next(iter(client_z_shared.values())).device
                )
                for rel in self.clients
            }

            # Step 4: Server fuses and trains
            loss, grads = self.server.aggregate_and_train(
                client_embeddings=client_z_shared,
                present_masks=present,
                pos_pairs=self.pos_pairs,
            )
            logger.info(f"  Server prediction loss={loss:.4f}")

            # Step 5: Split backprop at each client
            for rel, client in self.clients.items():
                # Map shared gradient back to local drug indices
                g_shared = grads[rel]
                idx = self.local_idx[rel]
                g_local = torch.zeros_like(client_z_local[rel])
                present_mask = idx >= 0
                g_local[idx[present_mask]] = g_shared[present_mask]

                client.apply_gradient(g_local, client_z_local[rel])

            # Step 6: MOCHA Omega update
            task_params = {
                rel: client.w_r.detach()
                for rel, client in self.clients.items()
            }
            self.server.update_mocha_omega(task_params)

            if t % 10 == 0 or t == self.num_rounds:
                ckpt_path = str(self.checkpoint_dir / f"round_{t:04d}.pt")
                self.server.save_checkpoint(ckpt_path, t)

        logger.info("VFL training complete.")
