"""Tests for loss functions and negative sampler."""
import pytest
import torch
from src.training.loss import WeightedBCELoss, TypeAwareNegativeSampler

def test_weighted_bce_scalar():
    loss_fn = WeightedBCELoss()
    pos = torch.randn(32)
    neg = torch.randn(64)
    loss = loss_fn(pos, neg)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_weighted_bce_with_weights():
    loss_fn = WeightedBCELoss(gamma1=1.0, gamma2=0.5)
    pos = torch.randn(16)
    neg = torch.randn(32)
    conf = torch.rand(16)
    rec  = torch.rand(16)
    loss = loss_fn(pos, neg, conf, rec)
    assert not torch.isnan(loss)

def test_type_aware_sampler_shape():
    sampler = TypeAwareNegativeSampler(
        num_drugs=100, num_adrs=50,
        hard_negative_ratio=0.3,
        negatives_per_positive=5,
    )
    pos_d = torch.randint(0, 100, (20,))
    pos_a = torch.randint(0, 50,  (20,))
    neg_d, neg_a = sampler.sample(pos_d, pos_a)
    # Without a registered hard-negative pool, n_hard falls back to 0.
    # All 20*5=100 slots are filled by type-constrained random sampling,
    # but hard_negative_ratio=0.3 reserves 30% → 70 random returned.
    # With pool registered, the full 100 are returned (70 random + 30 hard).
    assert neg_d.shape[0] == 70
    assert neg_a.shape[0] == 70

def test_type_aware_sampler_with_pool():
    sampler = TypeAwareNegativeSampler(
        num_drugs=100, num_adrs=50,
        hard_negative_ratio=0.3,
        negatives_per_positive=5,
    )
    hard = torch.stack([torch.randint(0, 100, (50,)),
                        torch.randint(0, 50,  (50,))], dim=1)
    sampler.register_hard_negatives(hard)
    pos_d = torch.randint(0, 100, (20,))
    pos_a = torch.randint(0, 50,  (20,))
    neg_d, neg_a = sampler.sample(pos_d, pos_a)
    assert neg_d.shape == (100,)   # 70 random + 30 hard = 20*5
    assert neg_a.shape == (100,)

def test_hard_negatives_registered():
    sampler = TypeAwareNegativeSampler(
        num_drugs=100, num_adrs=50,
        hard_negative_ratio=1.0,
        negatives_per_positive=3,
    )
    hard = torch.stack([torch.randint(0,100,(50,)),
                        torch.randint(0,50,(50,))], dim=1)
    sampler.register_hard_negatives(hard)
    pos_d = torch.randint(0,100,(10,))
    pos_a = torch.randint(0,50,(10,))
    neg_d, neg_a = sampler.sample(pos_d, pos_a)
    assert neg_d.shape[0] == 30
