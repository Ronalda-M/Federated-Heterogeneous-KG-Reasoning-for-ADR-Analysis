"""
HFL — Data source-Partitioned (Horizontal) Federated Learning

Architecture:
    Each client is an institution holding a local slice of all relation
    types. Clients train a shared-architecture HeterogeneousGNN locally
    and upload model parameters to the server. The server performs:
        1. KG merging and ontology alignment
        2. Reliability-weighted FedAvg aggregation
           beta_i = softmax(-gamma * val_loss_i)
        3. Broadcast of the updated global model back to all clients.

Round loop:
    for t in 1..T:
        for each client i:
            download global params theta^{(t)}
            train locally for E epochs on G^{(i)}
            compute val_loss^{(i)} on local held-out set
            upload theta^{(i,t+1)} and val_loss^{(i)}
        server aggregates:
            beta_i = softmax(-gamma * val_loss_i)
            theta^{(t+1)} = sum_i beta_i * theta^{(i,t+1)}
        broadcast theta^{(t+1)}
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.gnn import HeterogeneousGNN
from src.models.decoder import BilinearDecoder
from src.training.loss import WeightedBCELoss, TypeAwareNegativeSampler
from src.evaluation.metrics import compute_classification_metrics, TemperatureScaler

logger = logging.getLogger(__name__)


# ─── Data container ──────────────────────────────────────────────────────────

class LocalKGFragment:
    """
    Holds a single client's local KG fragment.

    Attributes:
        h:          {node_type: (N_t, D)} initial node feature tensors
        adj:        {relation: (src_idx, dst_idx)} edge index per relation
        edge_feat:  {relation: (E_r, F)} edge provenance attributes
        pos_pairs:  (M, 2) confirmed positive drug--ADR pairs [drug_idx, adr_idx]
        hard_neg:   (H, 2) co-mentioned but unconfirmed pairs (hard negatives)
        val_pairs:  (V, 2) validation positive pairs
        val_labels: (V,) 1.0 for all val_pairs (used with sampled negatives)
    """

    def __init__(
        self,
        h: Dict[str, torch.Tensor],
        adj: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_feat: Optional[Dict[str, torch.Tensor]] = None,
        pos_pairs: Optional[torch.Tensor] = None,
        hard_neg: Optional[torch.Tensor] = None,
        val_pairs: Optional[torch.Tensor] = None,
    ) -> None:
        self.h = h
        self.adj = adj
        self.edge_feat = edge_feat
        self.pos_pairs = pos_pairs
        self.hard_neg = hard_neg
        self.val_pairs = val_pairs

    def to(self, device: torch.device) -> "LocalKGFragment":
        self.h = {k: v.to(device) for k, v in self.h.items()}
        self.adj = {k: (s.to(device), d.to(device)) for k, (s, d) in self.adj.items()}
        if self.edge_feat:
            self.edge_feat = {k: v.to(device) for k, v in self.edge_feat.items()}
        if self.pos_pairs is not None:
            self.pos_pairs = self.pos_pairs.to(device)
        if self.hard_neg is not None:
            self.hard_neg = self.hard_neg.to(device)
        if self.val_pairs is not None:
            self.val_pairs = self.val_pairs.to(device)
        return self


# ─── Local model wrapper ─────────────────────────────────────────────────────

class LocalModel(nn.Module):
    """
    Full local model: HeterogeneousGNN encoder + BilinearDecoder.
    Shared architecture across all clients in Formulation A.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.gnn = HeterogeneousGNN(cfg)
        self.decoder = BilinearDecoder(
            hidden_dim=cfg["gnn"]["hidden_dim"],
            edge_feat_dim=cfg["decoder"].get("edge_mlp_hidden", 0),
            edge_mlp_hidden=cfg["decoder"].get("edge_mlp_hidden", 64),
            edge_mlp_layers=cfg["decoder"].get("edge_mlp_layers", 2),
        )

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        adj: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        drug_idx: torch.Tensor,
        adr_idx: torch.Tensor,
        edge_feat: Optional[Dict[str, torch.Tensor]] = None,
        pair_edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.gnn(h, adj, edge_feat)
        z_d = embeddings["Drug"][drug_idx]
        z_a = embeddings["AdverseEffect"][adr_idx]
        return self.decoder(z_d, z_a, pair_edge_attr)


# ─── Federated Client (Formulation A) ────────────────────────────────────────

class HFLClient:
    """
    A single institution-type federated client.

    Responsibilities:
        - Receive global parameters from server
        - Train locally for E epochs on the local KG fragment
        - Report updated parameters and local validation loss
    """

    def __init__(
        self,
        client_id: str,
        fragment: LocalKGFragment,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.fragment = fragment.to(device)
        self.cfg = cfg
        self.device = device

        self.model = LocalModel(cfg).to(device)

        train_cfg = cfg["training"]
        self.loss_fn = WeightedBCELoss(
            gamma1=cfg["loss"]["gamma1"],
            gamma2=cfg["loss"]["gamma2"],
            label_smoothing=cfg["loss"]["label_smoothing"],
        )

        num_drugs = fragment.h["Drug"].size(0)
        num_adrs = fragment.h["AdverseEffect"].size(0)
        self.neg_sampler = TypeAwareNegativeSampler(
            num_drugs=num_drugs,
            num_adrs=num_adrs,
            hard_negative_ratio=cfg["negative_sampling"]["hard_negative_ratio"],
            negatives_per_positive=cfg["negative_sampling"]["negatives_per_positive"],
        )
        if fragment.hard_neg is not None:
            self.neg_sampler.register_hard_negatives(fragment.hard_neg.cpu())

        self.local_lr = cfg["federated"]["formulation_a"]["aggregation"]["local_lr"]
        self.local_epochs = cfg["federated"]["formulation_a"]["aggregation"]["local_epochs"]

    def set_parameters(self, state_dict: Dict) -> None:
        """Download global model parameters from the server."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_parameters(self) -> Dict:
        """Upload local model parameters to the server."""
        return copy.deepcopy(self.model.state_dict())

    def train_round(self) -> float:
        """
        Perform one federation round of local training.

        Returns:
            val_loss: validation loss after local training (used for
                      reliability weighting at the server).
        """
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.local_lr,
            weight_decay=self.cfg["optimizer"]["weight_decay"],
        )

        f = self.fragment
        pos_drug = f.pos_pairs[:, 0]
        pos_adr = f.pos_pairs[:, 1]

        for _ in range(self.local_epochs):
            optimizer.zero_grad()

            # Positive scores
            pos_scores = self.model(
                f.h, f.adj, pos_drug, pos_adr, f.edge_feat
            )

            # Negative sampling
            neg_drug, neg_adr = self.neg_sampler.sample(pos_drug, pos_adr)
            neg_scores = self.model(
                f.h, f.adj, neg_drug, neg_adr, f.edge_feat
            )

            loss = self.loss_fn(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

        val_loss = self._compute_val_loss()
        logger.info(f"[{self.client_id}] val_loss={val_loss:.4f}")
        return val_loss

    def _compute_val_loss(self) -> float:
        self.model.eval()
        f = self.fragment
        if f.val_pairs is None:
            return 1.0

        with torch.no_grad():
            val_drug = f.val_pairs[:, 0]
            val_adr = f.val_pairs[:, 1]
            pos_scores = self.model(f.h, f.adj, val_drug, val_adr, f.edge_feat)

            neg_drug, neg_adr = self.neg_sampler.sample(val_drug, val_adr)
            neg_scores = self.model(f.h, f.adj, neg_drug, neg_adr, f.edge_feat)

            loss = self.loss_fn(pos_scores, neg_scores)
        return float(loss.item())


# ─── Federated Server (Formulation A) ────────────────────────────────────────

class HFLServer:
    """
    Centralised aggregation server for Formulation A.

    Responsibilities:
        - Maintain the global model
        - Perform reliability-weighted FedAvg aggregation
        - Broadcast updated parameters to all clients
        - Train the GNN on the merged global KG after each round

    Reliability-weighted FedAvg:
        beta_i = softmax(-gamma * val_loss_i)
        theta^{(t+1)} = sum_i beta_i * theta^{(i,t+1)}
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.gamma = cfg["federated"]["formulation_a"]["aggregation"]["temperature_gamma"]
        self.global_model = LocalModel(cfg).to(device)
        self.calibrator = TemperatureScaler().to(device)

    def get_global_parameters(self) -> Dict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(
        self,
        client_params: List[Dict],
        val_losses: List[float],
    ) -> None:
        """
        Reliability-weighted FedAvg.

        Args:
            client_params: list of state_dicts from each client
            val_losses:    corresponding validation losses
        """
        losses_t = torch.tensor(val_losses, dtype=torch.float32)
        weights = torch.softmax(-self.gamma * losses_t, dim=0)

        logger.info(f"Aggregation weights: { {i: f'{w:.3f}' for i, w in enumerate(weights.tolist())} }")

        global_sd = self.global_model.state_dict()
        for key in global_sd:
            global_sd[key] = torch.zeros_like(global_sd[key], dtype=torch.float32)
            for w, params in zip(weights, client_params):
                global_sd[key] += w * params[key].float()

        self.global_model.load_state_dict(global_sd)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Fit temperature scaling on global validation logits."""
        T = self.calibrator.fit(logits, labels)
        logger.info(f"Calibration: T={T:.4f}")
        return T

    def save_checkpoint(self, path: str, round_num: int) -> None:
        torch.save(
            {
                "round": round_num,
                "model_state_dict": self.global_model.state_dict(),
                "calibrator_state_dict": self.calibrator.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(ckpt["model_state_dict"])
        self.calibrator.load_state_dict(ckpt["calibrator_state_dict"])
        return int(ckpt["round"])


# ─── Federation Round Loop ────────────────────────────────────────────────────

class HFLTrainer:
    """
    Orchestrates the full Formulation A federation training loop.

    Round loop:
        for t in 1..num_rounds:
            server broadcasts global params to all clients
            each client trains locally for local_epochs
            each client reports params + val_loss
            server performs reliability-weighted FedAvg
            server saves checkpoint
    """

    def __init__(
        self,
        server: HFLServer,
        clients: List[HFLClient],
        cfg: dict,
        checkpoint_dir: str = "checkpoints/hfl/",
    ) -> None:
        self.server = server
        self.clients = clients
        self.cfg = cfg
        self.num_rounds = cfg["federated"]["formulation_a"]["aggregation"]["rounds"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        logger.info(f"Starting HFL training: {self.num_rounds} rounds, "
                    f"{len(self.clients)} clients")

        for t in range(1, self.num_rounds + 1):
            logger.info(f"=== Round {t}/{self.num_rounds} ===")

            # Broadcast global parameters
            global_params = self.server.get_global_parameters()
            for client in self.clients:
                client.set_parameters(global_params)

            # Local training
            client_params: List[Dict] = []
            val_losses: List[float] = []
            for client in self.clients:
                val_loss = client.train_round()
                client_params.append(client.get_parameters())
                val_losses.append(val_loss)

            # Aggregation
            self.server.aggregate(client_params, val_losses)

            # Checkpoint every 10 rounds
            if t % 10 == 0 or t == self.num_rounds:
                ckpt_path = str(self.checkpoint_dir / f"round_{t:04d}.pt")
                self.server.save_checkpoint(ckpt_path, t)

        logger.info("HFL training complete.")
