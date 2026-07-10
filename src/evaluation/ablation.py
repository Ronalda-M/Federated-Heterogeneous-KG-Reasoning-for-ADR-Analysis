"""
Ablation study runner for the Federated Heterogeneous KG ADR model.

Ablation variants (defined in configs/ablation/):
    full_model   — all components enabled
    no_edge_feat — edge provenance attributes zeroed (−EdgeFeat)
    no_llm_links — MENTIONS + EVIDENCE_FOR edges excluded (−LLMLinks)
    no_onto_norm — ontology normalisation disabled (−OntoNorm)

Usage:
    python scripts/run_ablation.py \\
        --data_dir data/processed/ \\
        --output results/ablation.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.evaluation.metrics import compute_classification_metrics, compute_ranking_metrics


ABLATION_CONFIGS = [
    "full_model",
    "no_edge_feat",
    "no_llm_links",
    "no_onto_norm",
]


def run_ablation(
    score_files: Dict[str, Dict[str, str]],
    datasets: List[str],
    k_values: List[int] = (1, 5, 10),
) -> Dict:
    """
    Aggregate ablation results across all variants and datasets.

    Args:
        score_files: {variant: {dataset: path_to_scores_npy}}
        datasets:    list of dataset names (e.g. ["neonatal", "offsides", "adrcecs"])
        k_values:    K values for Hits@K

    Returns:
        Nested dict: results[variant][dataset] = metric_dict
    """
    results: Dict = {}

    for variant in ABLATION_CONFIGS:
        results[variant] = {}
        for ds in datasets:
            path = score_files.get(variant, {}).get(ds)
            if path is None:
                results[variant][ds] = "not_available"
                continue

            data = np.load(path, allow_pickle=True).item()
            y_true = data["labels"]
            y_score = data["scores"]

            clf = compute_classification_metrics(y_true, y_score)
            rank = compute_ranking_metrics(y_score, y_true, k_values=list(k_values))

            results[variant][ds] = {**clf, **rank}

    return results


def format_ablation_table(results: Dict) -> str:
    """
    Render a plain-text summary table of ablation results.
    Suitable for logging or quick inspection.
    """
    header = f"{'Variant':<18} {'Dataset':<14} {'F1':>6} {'PR-AUC':>8} {'ROC-AUC':>9}"
    lines = [header, "-" * len(header)]
    for variant, dsets in results.items():
        for ds, metrics in dsets.items():
            if isinstance(metrics, str):
                lines.append(f"{variant:<18} {ds:<14} {'N/A':>6} {'N/A':>8} {'N/A':>9}")
            else:
                lines.append(
                    f"{variant:<18} {ds:<14} "
                    f"{metrics['f1']:>6.3f} "
                    f"{metrics['pr_auc']:>8.3f} "
                    f"{metrics['roc_auc']:>9.3f}"
                )
    return "\n".join(lines)


def main(data_dir: str, output: str) -> None:
    """
    Auto-discover score files under data_dir and run ablation suite.

    Expected layout:
        data_dir/
            full_model/
                neonatal.npy    # .npy dict with keys 'labels', 'scores'
                offsides.npy
                adrcecs.npy
            no_edge_feat/
                ...
    """
    base = Path(data_dir)
    datasets = ["neonatal", "offsides", "adrcecs"]

    score_files: Dict[str, Dict[str, str]] = {}
    for variant in ABLATION_CONFIGS:
        score_files[variant] = {}
        for ds in datasets:
            p = base / variant / f"{ds}.npy"
            if p.exists():
                score_files[variant][ds] = str(p)

    results = run_ablation(score_files, datasets)
    print(format_ablation_table(results))

    with open(output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nAblation results saved to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()

    main(args.data_dir, args.output)
