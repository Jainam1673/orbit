"""Statistical significance testing, Pass@k, and bootstrap estimation for ORBIT."""

import math
from collections.abc import Sequence

import numpy as np


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Computes unbiased estimator for Pass@k.

    Formula: 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    Args:
        n: Total number of generated candidate samples.
        c: Number of correct/successful samples.
        k: Pass@k threshold.
    """
    if n < k:
        raise ValueError(f"Total samples n={n} must be >= k={k}")
    if c < 0 or c > n:
        raise ValueError(f"Correct samples c={c} must be in range [0, {n}]")
    if n - c < k:
        return 1.0

    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def compute_bootstrap_ci(
    data: Sequence[float],
    num_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap (1 - alpha)% confidence interval of the mean."""
    if not data:
        return (0.0, 0.0)
    if len(data) == 1:
        return (float(data[0]), float(data[0]))

    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=np.float64)
    boot_means = np.zeros(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        resampled = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(resampled)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    low = float(np.percentile(boot_means, low_pct))
    high = float(np.percentile(boot_means, high_pct))
    return (low, high)


def compute_cohens_d(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> float:
    """Computes Cohen's d effect size between two independent samples."""
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0.0:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def compute_welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> dict[str, float]:
    """Computes Welch's t-test for samples with unequal variances and sample sizes."""
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"t_stat": 0.0, "mean_diff": 0.0, "df": 0.0}

    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0.0:
        return {"t_stat": 0.0, "mean_diff": m1 - m2, "df": float(n1 + n2 - 2)}

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = num / max(1e-12, den)

    return {
        "t_stat": t_stat,
        "mean_diff": m1 - m2,
        "df": df,
    }
