"""
Validation gate for walk-forward fold results.

Checks whether model performance (Sharpe/accuracy/IC) is stable across time,
with minimum combined strength and positive returns in both halves.
"""
import numpy as np


def sharpe_ratio(returns: np.ndarray | list, annualization_factor: float = 252) -> float:
    """
    Annualized Sharpe ratio from return series.

    Args:
        returns: array of returns (daily, quarterly, etc.)
        annualization_factor: sqrt of periods per year (252 for daily, 52 for weekly, 4 for quarterly)

    Returns:
        Annualized Sharpe ratio (mean/std * sqrt(annualization_factor))
    """
    returns = np.asarray(returns)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(annualization_factor))


def passes_validation_gate(
    fold_results: list[dict],
    metric_key: str = "sharpe",
    combined_threshold: float = 1.0,
    half_threshold: float = 0.0,
    min_frac_positive: float = 0.0,
) -> dict:
    """
    Check if model passes validation gate over time.

    Splits fold_results (ordered by time) in half and checks:
    - combined metric > combined_threshold
    - first half metric > half_threshold
    - second half metric > half_threshold
    - fraction of folds with metric > 0 >= min_frac_positive

    Args:
        fold_results: list of fold dicts (from walk_forward CV, ordered by time)
        metric_key: key name in each fold dict (e.g., "sharpe", "accuracy", "spearman_rho")
        combined_threshold: minimum acceptable combined metric
        half_threshold: minimum for each time half
        min_frac_positive: minimum fraction of individual folds with metric > 0 (0 = no requirement)

    Returns:
        {
            "passed": bool,
            "combined_metric": float (mean across all folds),
            "first_half_metric": float,
            "second_half_metric": float,
            "frac_positive": float,
            "reason": str (explanation)
        }
    """
    if not fold_results:
        return {
            "passed": False,
            "combined_metric": 0.0,
            "first_half_metric": 0.0,
            "second_half_metric": 0.0,
            "frac_positive": 0.0,
            "reason": "No fold results",
        }

    # Extract metric from each fold
    metrics = []
    missing = []
    for i, fold in enumerate(fold_results):
        if metric_key in fold:
            metrics.append(fold[metric_key])
        else:
            missing.append(i)

    if missing:
        return {
            "passed": False,
            "combined_metric": 0.0,
            "first_half_metric": 0.0,
            "second_half_metric": 0.0,
            "frac_positive": 0.0,
            "reason": f"Missing {metric_key} in folds: {missing}",
        }

    metrics = np.asarray(metrics)
    combined = float(metrics.mean())
    frac_positive = float((metrics > 0).mean())

    # Split by time (first half of folds vs second half)
    n = len(metrics)
    mid = n // 2
    first_half = float(metrics[:mid].mean()) if mid > 0 else 0.0
    second_half = float(metrics[mid:].mean()) if n > mid else 0.0

    # Check gates
    combined_ok = combined > combined_threshold
    first_ok = first_half > half_threshold
    second_ok = second_half > half_threshold
    frac_positive_ok = frac_positive >= min_frac_positive
    passed = combined_ok and first_ok and second_ok and frac_positive_ok

    reason = ""
    if not combined_ok:
        reason += f"{metric_key} combined {combined:.4f} <= {combined_threshold:.4f}. "
    if not first_ok:
        reason += f"First half {first_half:.4f} <= {half_threshold:.4f}. "
    if not second_ok:
        reason += f"Second half {second_half:.4f} <= {half_threshold:.4f}. "
    if not frac_positive_ok:
        reason += f"Only {frac_positive:.0%} of folds positive (need >= {min_frac_positive:.0%}). "
    if not reason:
        reason = (f"PASS: {metric_key} combined {combined:.4f}, halves {first_half:.4f}/{second_half:.4f}, "
                   f"{frac_positive:.0%} folds positive")

    return {
        "passed": passed,
        "combined_metric": combined,
        "first_half_metric": first_half,
        "second_half_metric": second_half,
        "frac_positive": frac_positive,
        "reason": reason,
    }


if __name__ == "__main__":
    # Self-check: synthetic fold results
    print("Running validation gate self-checks...\n")

    # ── Test 1: Good model (passes gate on accuracy) ──
    fold_results_good = [
        {"fold": 1, "accuracy": 0.58},
        {"fold": 2, "accuracy": 0.60},
        {"fold": 3, "accuracy": 0.59},
        {"fold": 4, "accuracy": 0.61},
    ]
    result = passes_validation_gate(fold_results_good, metric_key="accuracy", combined_threshold=0.55, half_threshold=0.5)
    assert result["passed"], f"Good model should pass. Result: {result}"
    print("✓ Test 1 PASS: Good accuracy model")
    print(f"  Combined: {result['combined_metric']:.4f}, Halves: {result['first_half_metric']:.4f}/{result['second_half_metric']:.4f}\n")

    # ── Test 2: Poor model (fails gate on accuracy) ──
    fold_results_poor = [
        {"fold": 1, "accuracy": 0.48},
        {"fold": 2, "accuracy": 0.50},
        {"fold": 3, "accuracy": 0.51},
        {"fold": 4, "accuracy": 0.49},
    ]
    result = passes_validation_gate(fold_results_poor, metric_key="accuracy", combined_threshold=0.55, half_threshold=0.5)
    assert not result["passed"], f"Poor model should fail. Result: {result}"
    print("✓ Test 2 PASS: Poor accuracy model correctly rejected")
    print(f"  Reason: {result['reason']}\n")

    # ── Test 3: Degrading model (passes combined but fails time split) ──
    fold_results_degrade = [
        {"fold": 1, "accuracy": 0.62},
        {"fold": 2, "accuracy": 0.61},
        {"fold": 3, "accuracy": 0.48},  # model breaks in second half
        {"fold": 4, "accuracy": 0.47},
    ]
    result = passes_validation_gate(fold_results_degrade, metric_key="accuracy", combined_threshold=0.55, half_threshold=0.5)
    assert not result["passed"], f"Degrading model should fail. Result: {result}"
    print("✓ Test 3 PASS: Degrading model correctly rejected")
    print(f"  Reason: {result['reason']}\n")

    # ── Test 4: Sharpe ratio on correlation-like metric ──
    fold_results_corr = [
        {"fold": 1, "spearman_rho": 0.15},
        {"fold": 2, "spearman_rho": 0.18},
        {"fold": 3, "spearman_rho": 0.12},
        {"fold": 4, "spearman_rho": 0.14},
    ]
    result = passes_validation_gate(fold_results_corr, metric_key="spearman_rho", combined_threshold=0.1, half_threshold=0.0)
    assert result["passed"], f"Good correlation model should pass. Result: {result}"
    print("✓ Test 4 PASS: Spearman rho gate works")
    print(f"  Combined: {result['combined_metric']:.4f}, Halves: {result['first_half_metric']:.4f}/{result['second_half_metric']:.4f}\n")

    # ── Test 5: Sharpe ratio utility function ──
    returns = np.array([0.001, 0.002, -0.0005, 0.0015, 0.0012])
    sharpe = sharpe_ratio(returns, annualization_factor=252)
    assert isinstance(sharpe, float) and sharpe > 0, f"Sharpe calculation failed: {sharpe}"
    print("✓ Test 5 PASS: Sharpe ratio utility function")
    print(f"  Sharpe from 5 daily returns: {sharpe:.4f}\n")

    print("All validation checks passed!")
