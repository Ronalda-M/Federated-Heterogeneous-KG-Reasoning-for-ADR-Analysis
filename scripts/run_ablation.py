"""
Ablation study entry-point.

Usage:
    python scripts/run_ablation.py \\
        --data_dir data/raw/ \\
        --output   results/ablation.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

ABLATION_CONFIGS = {
    "full_model":    "configs/ablation/full_model.yaml",
    "no_edge_feat":  "configs/ablation/no_edge_feat.yaml",
    "no_llm_links":  "configs/ablation/no_llm_links.yaml",
    "no_onto_norm":  "configs/ablation/no_onto_norm.yaml",
}
DATASETS = ["neonatal", "offsides", "adrcecs"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data/raw/")
    p.add_argument("--results_dir",  default="results/")
    p.add_argument("--output",       default="results/ablation.json")
    args = p.parse_args()

    from src.evaluation.ablation import run_ablation, format_ablation_table

    # Build score_files dict from pre-computed score arrays
    # (each variant must have been trained and evaluated first)
    score_files = {}
    for variant in ABLATION_CONFIGS:
        score_files[variant] = {}
        for ds in DATASETS:
            p_ = Path(args.results_dir) / variant / f"{ds}.npy"
            if p_.exists():
                score_files[variant][ds] = str(p_)
            else:
                logger.warning(f"Missing: {p_} — run train.py + evaluate.py first")

    results = run_ablation(score_files, DATASETS)
    print(format_ablation_table(results))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results saved to {args.output}")

if __name__ == "__main__":
    main()
