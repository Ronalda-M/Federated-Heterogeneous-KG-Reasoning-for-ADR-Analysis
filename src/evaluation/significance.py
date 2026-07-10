"""
Statistical significance testing for ADR prediction experiments.

Methods:
    - Paired bootstrap resampling (2000 iterations)
    - Two-tailed p-value estimation for PR-AUC differences
    - Holm-Bonferroni correction for multiple comparisons
      (one test per baseline model)

Usage:
    python scripts/run_significance_tests.py \\
        --scores_dir results/scores/ \\
        --output results/significance.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


# ─── Bootstrap ───────────────────────────────────────────────────────────────

def bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    scores_proposed: np.ndarray,
    scores_baseline: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap test for PR-AUC difference (proposed - baseline).

    Args:
        y_true:           (N,) ground-truth labels
        scores_proposed:  (N,) predicted scores from proposed model
        scores_baseline:  (N,) predicted scores from baseline model
        n_bootstrap:      number of resampling iterations
        seed:             random seed for reproducibility

    Returns:
        (observed_diff, mean_boot_diff, ci_lower_95, p_value)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed_diff = (
        average_precision_score(y_true, scores_proposed)
        - average_precision_score(y_true, scores_baseline)
    )

    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        diff = (
            average_precision_score(yt, scores_proposed[idx])
            - average_precision_score(yt, scores_baseline[idx])
        )
        boot_diffs.append(diff)

    boot_arr = np.array(boot_diffs)
    mean_diff = float(boot_arr.mean())
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))

    # Two-tailed p-value: proportion of bootstrap diffs on wrong side of 0
    p_value = float(
        min(
            2 * (boot_arr <= 0).mean(),
            2 * (boot_arr >= 0).mean(),
        )
    )

    return observed_diff, mean_diff, ci_lower, ci_upper, p_value


# ─── Holm-Bonferroni Correction ──────────────────────────────────────────────

def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: list of raw p-values (one per comparison)
        alpha:    family-wise error rate threshold

    Returns:
        (adjusted_p_values, reject_null_flags)
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    adjusted = np.zeros(m)
    reject = np.zeros(m, dtype=bool)

    for i, (p, orig_idx) in enumerate(zip(sorted_p, order)):
        adj = p * (m - i)
        adjusted[orig_idx] = min(adj, 1.0)
        if adj <= alpha:
            reject[orig_idx] = True
        else:
            # Once we fail to reject, all subsequent are also kept
            break

    return adjusted.tolist(), reject.tolist()


# ─── Full Significance Suite ──────────────────────────────────────────────────

def run_significance_suite(
    y_true: np.ndarray,
    scores: Dict[str, np.ndarray],
    proposed_key: str = "proposed",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Run bootstrap tests between the proposed model and all baselines,
    then apply Holm-Bonferroni correction.

    Args:
        y_true:       (N,) ground-truth labels
        scores:       {model_name: score_array} for all models
        proposed_key: key for the proposed model in scores dict
        n_bootstrap:  bootstrap iterations
        alpha:        family-wise error rate
        seed:         random seed

    Returns:
        Dictionary with per-comparison results and corrected p-values.
    """
    baselines = [k for k in scores if k != proposed_key]
    raw_p: List[float] = []
    comparisons: List[Dict] = []

    for baseline in baselines:
        obs, mean_d, ci_lo, ci_hi, p = bootstrap_pr_auc_diff(
            y_true=y_true,
            scores_proposed=scores[proposed_key],
            scores_baseline=scores[baseline],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        raw_p.append(p)
        comparisons.append({
            "baseline": baseline,
            "observed_pr_auc_diff": round(obs, 4),
            "mean_bootstrap_diff": round(mean_d, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "p_value_raw": round(p, 4),
        })

    adj_p, reject = holm_bonferroni(raw_p, alpha=alpha)

    for i, comp in enumerate(comparisons):
        comp["p_value_holm_bonferroni"] = round(adj_p[i], 4)
        comp["significant"] = bool(reject[i])

    return {
        "proposed_model": proposed_key,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "comparisons": comparisons,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(scores_dir: str, output: str, proposed_key: str = "proposed") -> None:
    """
    Load score arrays from scores_dir, run significance suite, save JSON.

    Expected directory layout:
        scores_dir/
            labels.npy         — (N,) ground-truth labels
            proposed.npy       — (N,) proposed model scores
            distmult.npy
            complex.npy
            biobert.npy
            ...
    """
    scores_path = Path(scores_dir)
    y_true = np.load(scores_path / "labels.npy")

    scores: Dict[str, np.ndarray] = {}
    for f in scores_path.glob("*.npy"):
        if f.stem != "labels":
            scores[f.stem] = np.load(f)

    results = run_significance_suite(
        y_true=y_true,
        scores=scores,
        proposed_key=proposed_key,
    )

    with open(output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposed_key", default="proposed")
    args = parser.parse_args()

    main(args.scores_dir, args.output, args.proposed_key)
