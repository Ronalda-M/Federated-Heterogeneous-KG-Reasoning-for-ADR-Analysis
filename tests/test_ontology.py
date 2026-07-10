"""Tests for ontology normalisation."""
import pytest
from src.data.ontology_normalizer import (
    _canonicalise,
    DrugNormalizer,
    AdverseEffectNormalizer,
    EntityDeduplicator,
    OntologyNormalizer,
)

def test_canonicalise():
    assert _canonicalise("  Aspirin  ") == "aspirin"
    assert _canonicalise("Café") == "cafe"
    assert _canonicalise("Naproxén Sodium") == "naproxen sodium"

def test_drug_normalizer_no_file():
    norm = DrugNormalizer()   # no vocab files → returns None
    assert norm.normalise("aspirin") is None

def test_entity_deduplicator():
    dedup = EntityDeduplicator()
    i1 = dedup.add_drug("DB00001")
    i2 = dedup.add_drug("DB00002")
    i3 = dedup.add_drug("DB00001")  # duplicate
    assert i1 != i2
    assert i1 == i3
    assert dedup.num_drugs == 2

def test_dedup_report_keys():
    dedup = EntityDeduplicator()
    dedup.add_drug("D1"); dedup.add_ae("PT001")
    report = dedup.dedup_report()
    assert "unique_drugs" in report
    assert "drug_duplicates_merged" in report

def test_ontology_normalizer_no_files():
    norm = OntologyNormalizer()
    # Without vocab files, all lookups return None
    assert norm.process_drug("metformin") is None
    assert norm.process_ae("nausea") is None
