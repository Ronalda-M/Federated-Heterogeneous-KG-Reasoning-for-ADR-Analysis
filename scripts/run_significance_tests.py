"""
Statistical significance testing entry-point.

Usage:
    python scripts/run_significance_tests.py \\
        --scores_dir results/scores/ \\
        --output     results/significance.json
"""
from src.evaluation.significance import main
if __name__ == "__main__":
    main()
