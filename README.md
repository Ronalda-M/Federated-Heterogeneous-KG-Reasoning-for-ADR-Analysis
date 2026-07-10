# Federated Heterogeneous Knowledge Graph Reasoning for Adverse Drug Reaction Prediction

## Overview

This framework integrates distributed pharmacovigilance data sources — hospital EHRs, FAERS spontaneous reports, biomedical literature, and curated databases — into an ontology-aligned heterogeneous knowledge graph (KG) without centralising raw patient records. A MedGemma-4B (LoRA fine-tuned) module performs ontology-constrained biomedical named entity recognition and relation extraction. A relation-aware GNN (HGT with R-GCN as a variant) performs typed link prediction over the resulting KG for Drug–ADR association scoring. Two federated learning formulations are supported.

| Formulation | Description | Aggregation |
|---|---|---|
| **HFL — Horizontal (data source-partitioned)** | Each client is an institution holding all relation types | Reliability-weighted FedAvg |
| **VFL — Vertical (relation-partitioned multi-view)** | Each client owns one relation type over a shared drug vocabulary | Cross-view attention fusion + MOCHA |

---

## Repository Structure

```
fedkg-adr/
├── configs/
│   ├── model.yaml              # GNN architecture, node types, input features
│   ├── training.yaml           # optimiser, loss, negative sampling, calibration, splits
│   ├── federated.yaml          # FL formulations A and B
│   ├── dataset.yaml            # data sources, KG schema, ontology alignment
│   └── ablation/
│       ├── full_model.yaml     # baseline (full model)
│       ├── no_edge_feat.yaml   # −EdgeFeat ablation
│       ├── no_llm_links.yaml   # −LLMLinks ablation
│       └── no_onto_norm.yaml   # −OntoNorm ablation
├── src/
│   ├── data/
│   │   ├── dataset.py          # dataset loading, stratified splitting
│   │   ├── kg_builder.py       # KG construction, cross-source deduplication
│   │   └── ontology_normalizer.py  # entity standardisation to canonical IDs
│   ├── models/
│   │   ├── gnn.py              # RGCNLayer, HGTLayer, HeterogeneousGNN
│   │   ├── decoder.py          # BilinearDecoder, PolypharmacyDecoder (TWOSIDES)
│   │   └── llm_extractor.py    # MedGemma-4B + LoRA NER/RE pipeline
│   ├── federated/
│   │   ├── HFL.py    # Horizontal FL: HFLClient, HFLServer, HFLTrainer
│   │   └── VFL.py    # Vertical FL: VFLClient, FusionServer, VFLTrainer
│   ├── training/
│   │   └── loss.py             # WeightedBCELoss, TypeAwareNegativeSampler
│   └── evaluation/
│       ├── metrics.py          # Precision/Recall/F1, ROC-AUC, PR-AUC, MRR, Hits@K, TemperatureScaler
│       ├── ablation.py         # ablation study runner
│       └── significance.py     # paired bootstrap + Holm-Bonferroni correction
├── scripts/
│   ├── train.py                # main training entry point
│   ├── evaluate.py             # evaluation on held-out test sets
│   ├── run_ablation.py         # run all ablation variants
│   └── run_significance_tests.py  # statistical significance testing
├── tests/
│   ├── test_gnn.py
│   ├── test_decoder.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_metrics.py
│   ├── test_ontology.py
│   └── test_significance.py
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/<your-org>/fedkg-adr.git
cd fedkg-adr
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

**Python version**: 3.9 or above.

---

## Configuration

All hyperparameters are specified in `configs/`. The exact values used in the paper:

| Parameter | Value | Config key |
|---|---|---|
| Hidden dimension *k* | 256 | `model.yaml → gnn.hidden_dim` |
| GNN layers *L* | 3 | `model.yaml → gnn.num_layers` |
| Attention heads *H* | 8 | `model.yaml → gnn.num_heads` |
| Dropout | 0.3 | `model.yaml → gnn.dropout` |
| Neighbourhood fanouts | [20, 10] | `model.yaml → gnn.fanouts` |
| Optimiser | AdamW | `training.yaml → optimizer.name` |
| Learning rate | 2 × 10⁻³ | `training.yaml → optimizer.lr` |
| Weight decay | 10⁻⁴ | `training.yaml → optimizer.weight_decay` |
| Label smoothing | 0.05 | `training.yaml → loss.label_smoothing` |
| Early stopping patience | 10 epochs | `training.yaml → early_stopping.patience` |
| Max epochs | 200 | `training.yaml → training.max_epochs` |
| Train / Val / Test split | 70 / 15 / 15 | `training.yaml → splits` |
| FL rounds | 50 | `federated.yaml → formulation_a.aggregation.rounds` |
| Local epochs per round | 5 | `federated.yaml → formulation_a.aggregation.local_epochs` |
| Reliability weighting γ | 2.0 | `federated.yaml → formulation_a.aggregation.temperature_gamma` |
| Hard negative ratio | 0.3 | `training.yaml → negative_sampling.hard_negative_ratio` |
| Negatives per positive | 5 | `training.yaml → negative_sampling.negatives_per_positive` |

---

## Training

### Formulation A — Horizontal Federated Learning

```bash
python scripts/train.py \
    --model_config   configs/model.yaml \
    --train_config   configs/training.yaml \
    --fed_config     configs/federated.yaml \
    --dataset_config configs/dataset.yaml \
    --HFL \
    --output_dir     results/formulation_a/
```

### Formulation B — Vertical Federated Learning

```bash
python scripts/train.py \
    --model_config   configs/model.yaml \
    --train_config   configs/training.yaml \
    --fed_config     configs/federated.yaml \
    --dataset_config configs/dataset.yaml \
    --VFL \
    --output_dir     results/formulation_b/
```

### Ablation variants

```bash
# Without edge provenance features
python scripts/train.py \
    --model_config configs/ablation/no_edge_feat.yaml \
    --train_config configs/training.yaml \
    --output_dir   results/no_edge_feat/

# Without LLM-augmented links
python scripts/train.py \
    --model_config configs/ablation/no_llm_links.yaml \
    --train_config configs/training.yaml \
    --output_dir   results/no_llm_links/

# Without ontology normalisation
python scripts/train.py \
    --model_config configs/ablation/no_onto_norm.yaml \
    --train_config configs/training.yaml \
    --output_dir   results/no_onto_norm/
```

---

## Evaluation

```bash
# Evaluate on a specific dataset (neonatal / offsides / adrecs / twosides)
python scripts/evaluate.py \
    --checkpoint results/formulation_a/best.pt \
    --dataset    neonatal \
    --output     results/metrics_neonatal.json

# Run all ablation variants and aggregate results
python scripts/run_ablation.py \
    --data_dir results/ \
    --output   results/ablation_summary.json

# Statistical significance testing (paired bootstrap + Holm-Bonferroni)
python scripts/run_significance_tests.py \
    --scores_dir results/scores/ \
    --output     results/significance.json
```

---

## Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Neonatal Drug ADR Association | [Mendeley Data](https://doi.org/10.17632/ppd2c7sz8j.2) | Primary paediatric pharmacovigilance evaluation |
| OFFSIDES | TWOSIDES project | Off-label side-effect detection |
| ADReCS | [ADReCS](http://bioinf.xmu.edu.cn/ADReCS) | Ontology-structured ADR retrieval |
| TWOSIDES | TWOSIDES project | Polypharmacy drug-pair interactions |
| FAERS | [FDA](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html) | Spontaneous adverse event reports |
| SIDER 4.1 | [EMBL-EBI](http://sideeffects.embl.de) | Curated drug side effects |
| DrugBank 5.0 | [DrugBank](https://go.drugbank.com) | Drug molecular properties |
| CHEMDNER | BioCreative IV | Chemical NER fine-tuning |
| CDR | BioCreative V | Chemical-disease relation extraction |

Place downloaded datasets under `data/<dataset_name>/` before running training.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All 32 unit tests cover the GNN forward pass, decoder scoring, loss functions, negative sampling, ontology normalisation, evaluation metrics, and significance testing.

---

## Reproducibility

- All hyperparameters are specified exactly in `configs/`.
- Data splits are stratified by drug identity (not by drug–ADR pair) with `seed: 42`, preventing leakage of drug embeddings from train to test.
- De-duplication is applied both within each source and across all sources (`cross_source_dedup: true` in `configs/dataset.yaml`); edge provenance attributes are retained and consolidated during deduplication.
- Temperature scaling calibration is applied post-training on validation predictions; the learned temperature *T* is stored in checkpoints alongside model weights.
- Trained model checkpoints, configuration files, and evaluation outputs are stored under `results/` and available for download at the repository releases page.

---


