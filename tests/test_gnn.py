"""Tests for GNN model components."""
import pytest
import torch
from src.models.gnn import RGCNLayer, HGTLayer, HeterogeneousGNN

NODE_TYPES = ["Drug", "AdverseEffect", "ClinicalFactor", "Target"]
RELATION_TYPES = ["DRUG_CAUSES_AE", "HAS_TARGET", "ASSOCIATED_WITH"]
D = 64

@pytest.fixture
def small_graph():
    h = {nt: torch.randn(10, D) for nt in NODE_TYPES}
    adj = {
        "DRUG_CAUSES_AE":  (torch.randint(0, 10, (20,)),
                            torch.randint(0, 10, (20,))),
        "HAS_TARGET":      (torch.randint(0, 10, (15,)),
                            torch.randint(0, 10, (15,))),
        "ASSOCIATED_WITH": (torch.randint(0, 10, (12,)),
                            torch.randint(0, 10, (12,))),
    }
    return h, adj

def test_rgcn_forward(small_graph):
    h, adj = small_graph
    layer = RGCNLayer(NODE_TYPES, RELATION_TYPES, D)
    h_new = layer(h, adj)
    for nt in NODE_TYPES:
        assert h_new[nt].shape == (10, D)
        assert not torch.isnan(h_new[nt]).any()

def test_hgt_forward(small_graph):
    h, adj = small_graph
    layer = HGTLayer(NODE_TYPES, RELATION_TYPES, D, num_heads=4)
    h_new = layer(h, adj)
    for nt in NODE_TYPES:
        assert h_new[nt].shape == (10, D)

def test_residual_applied(small_graph):
    """Residual connection: output should differ from zero-init aggregation."""
    h, adj = small_graph
    layer = RGCNLayer(NODE_TYPES, RELATION_TYPES, D)
    h_new = layer(h, adj)
    # With residual, output must not collapse to zero
    for nt in NODE_TYPES:
        assert h_new[nt].abs().mean().item() > 1e-6

def test_heterogeneous_gnn_layers():
    cfg = {
        "node_types": NODE_TYPES,
        "relation_types": RELATION_TYPES,
        "gnn": {"hidden_dim": D, "num_layers": 2, "num_heads": 4,
                "dropout": 0.0, "architecture": "hgt"},
        "decoder": {"edge_mlp_hidden": 0},
    }
    model = HeterogeneousGNN(cfg)
    h = {nt: torch.randn(10, D) for nt in NODE_TYPES}
    adj = {
        "DRUG_CAUSES_AE":  (torch.randint(0,10,(15,)), torch.randint(0,10,(15,))),
        "HAS_TARGET":      (torch.randint(0,10,(10,)), torch.randint(0,10,(10,))),
        "ASSOCIATED_WITH": (torch.randint(0,10,(8,)),  torch.randint(0,10,(8,))),
    }
    out = model(h, adj)
    assert "Drug" in out
    assert out["Drug"].shape == (10, D)
