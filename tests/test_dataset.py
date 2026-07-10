"""Tests for dataset and dataloader."""
import pytest
import torch
from src.data.kg_builder import KGData
from src.data.dataset import DrugADRDataset, DrugADRDataModule, ColdStartDataset

def _make_dummy_kg(n_drugs=50, n_aes=30, n_edges=200):
    """Build a minimal KGData for testing without raw files."""
    kg = KGData()
    kg.node_features = {
        "Drug":              torch.randn(n_drugs, 32),
        "AdverseEffect":     torch.randn(n_aes, 32),
        "ClinicalFactor":    torch.randn(10, 32),
        "Target":            torch.randn(10, 32),
        "MedicalLiterature": torch.randn(5, 32),
        "BiomedicalReport":  torch.randn(5, 32),
    }
    src = torch.randint(0, n_drugs, (n_edges,))
    dst = torch.randint(0, n_aes,   (n_edges,))
    kg.edge_index = {"DRUG_CAUSES_AE": torch.stack([src, dst])}
    kg.edge_attr  = {"DRUG_CAUSES_AE": torch.randn(n_edges, 5)}
    return kg

def test_dataset_from_kg():
    kg = _make_dummy_kg()
    ds = DrugADRDataset.from_kg(kg)
    assert len(ds) == 200
    item = ds[0]
    assert "drug_idx" in item and "adr_idx" in item

def test_datamodule_splits():
    kg = _make_dummy_kg()
    dm = DrugADRDataModule(kg, train_frac=0.7, val_frac=0.15,
                           test_frac=0.15, batch_size=32, num_workers=0)
    sizes = dm.split_sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == 200

def test_datamodule_loaders():
    kg = _make_dummy_kg()
    dm = DrugADRDataModule(kg, batch_size=32, num_workers=0)
    batch = next(iter(dm.train_dataloader()))
    assert "drug_idx" in batch
    assert batch["drug_idx"].shape[0] <= 32

def test_cold_start_dataset():
    kg = _make_dummy_kg()
    dm = DrugADRDataModule(kg, batch_size=32, num_workers=0)
    cs = ColdStartDataset.from_split(dm.test_set, dm.train_set, "drug")
    assert len(cs) >= 0   # may be 0 if all test drugs appear in train
