"""Tests for decoder components."""
import pytest
import torch
from src.models.decoder import BilinearDecoder, PolypharmacyDecoder

D = 64

def test_bilinear_decoder_shape():
    dec = BilinearDecoder(hidden_dim=D, edge_feat_dim=0)
    z_d = torch.randn(32, D)
    z_a = torch.randn(32, D)
    scores = dec(z_d, z_a)
    assert scores.shape == (32,)

def test_bilinear_decoder_with_edge_feat():
    dec = BilinearDecoder(hidden_dim=D, edge_feat_dim=8, edge_mlp_hidden=32)
    z_d = torch.randn(32, D)
    z_a = torch.randn(32, D)
    ea  = torch.randn(32, 8)
    scores = dec(z_d, z_a, ea)
    assert scores.shape == (32,)
    assert not torch.isnan(scores).any()

def test_predict_range():
    dec = BilinearDecoder(hidden_dim=D, edge_feat_dim=0)
    z_d = torch.randn(16, D)
    z_a = torch.randn(16, D)
    probs = dec.predict(z_d, z_a)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_polypharmacy_decoder():
    dec = PolypharmacyDecoder(hidden_dim=D)
    z_drugs = [torch.randn(8, D), torch.randn(8, D)]
    z_a = torch.randn(8, D)
    scores = dec(z_drugs, z_a)
    assert scores.shape == (8,)
