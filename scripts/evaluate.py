"""
Evaluation entry-point.

Usage:
    python scripts/evaluate.py \\
        --checkpoint results/full_model/best.pt \\
        --dataset neonatal \\
        --data_dir data/raw/ \\
        --output   results/metrics_neonatal.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import torch, yaml

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(p):
    with open(p) as f: return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--model_config",   default="configs/model.yaml")
    p.add_argument("--train_config",   default="configs/training.yaml")
    p.add_argument("--fed_config",     default="configs/federated.yaml")
    p.add_argument("--dataset_config", default="configs/dataset.yaml")
    p.add_argument("--data_dir",       default="data/raw/")
    p.add_argument("--dataset",        default="neonatal",
                   choices=["neonatal","offsides","adrcecs"])
    p.add_argument("--output",         default="results/metrics.json")
    p.add_argument("--cold_start",     action="store_true")
    p.add_argument("--subgroup_col",   default=None,
                   help="Column name in dataset for subgroup stratification")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = {}
    for k in ["model_config","train_config","fed_config","dataset_config"]:
        cfg.update(load_yaml(getattr(args, k)))

    device = torch.device(args.device)

    # Build KG and dataset
    from src.data.ontology_normalizer import OntologyNormalizer
    from src.data.kg_builder import KGBuilder
    from src.data.dataset import DrugADRDataModule, ColdStartDataset
    from src.federated.formulation_a import LocalModel
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_ranking_metrics,
        evaluate_subgroup,
        TemperatureScaler,
    )

    norm = OntologyNormalizer(
        rxnorm_path=cfg.get("rxnorm_path"),
        drugbank_path=cfg.get("drugbank_vocab_path"),
        meddra_path=cfg.get("meddra_path"),
        mesh_path=cfg.get("mesh_path"),
    )
    kg = KGBuilder(cfg, norm).build(args.data_dir)
    kg.to(device)

    dm = DrugADRDataModule(kg=kg,
        train_frac=cfg["splits"]["train"],
        val_frac=cfg["splits"]["val"],
        test_frac=cfg["splits"]["test"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"])

    # Load model
    model = LocalModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load calibrator if present
    calibrator = TemperatureScaler().to(device)
    if "calibrator_state_dict" in ckpt:
        calibrator.load_state_dict(ckpt["calibrator_state_dict"])

    # Evaluate on test set
    test_dl = dm.test_dataloader()
    adj = {r: (v[0], v[1]) for r, v in kg.edge_index.items()}

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            d_idx = batch["drug_idx"].to(device)
            a_idx = batch["adr_idx"].to(device)
            logits = model(kg.node_features, adj, d_idx, a_idx, kg.edge_attr)
            all_logits.append(logits.cpu())
            all_labels.append(torch.ones(logits.size(0)))

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.sigmoid(all_logits).numpy()

    results = {}
    results["classification"] = compute_classification_metrics(all_labels, all_scores)
    results["ranking"] = compute_ranking_metrics(
        all_scores, all_labels, k_values=[1, 5, 10])

    logger.info(f"Test F1:     {results['classification']['f1']:.4f}")
    logger.info(f"Test PR-AUC: {results['classification']['pr_auc']:.4f}")
    logger.info(f"Test ROC-AUC:{results['classification']['roc_auc']:.4f}")
    logger.info(f"Test MRR:    {results['ranking']['mrr']:.4f}")

    # Cold-start evaluation
    if args.cold_start:
        for unseen in ["drug", "adr"]:
            cs = ColdStartDataset.from_split(dm.test_set, dm.train_set, unseen)
            if len(cs) == 0:
                continue
            cs_logits = []
            cs_labels = cs.labels.numpy()
            cs_dl = torch.utils.data.DataLoader(cs, batch_size=512)
            with torch.no_grad():
                for b in cs_dl:
                    lg = model(kg.node_features, adj,
                               b["drug_idx"].to(device),
                               b["adr_idx"].to(device), kg.edge_attr)
                    cs_logits.append(torch.sigmoid(lg).cpu())
            cs_scores = torch.cat(cs_logits).numpy()
            results[f"cold_start_{unseen}"] = compute_classification_metrics(
                cs_labels, cs_scores)
            logger.info(f"Cold-start ({unseen}) PR-AUC: "
                        f"{results[f'cold_start_{unseen}']['pr_auc']:.4f}")

    results["dataset"] = args.dataset
    results["checkpoint"] = args.checkpoint
    results["split_sizes"] = dm.split_sizes()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
