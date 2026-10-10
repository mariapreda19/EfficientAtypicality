from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import betaln, logsumexp

@dataclass(frozen=True)
class Config:

    N: int = 10_000

    P0: float = 0.5

    EPSILON: float = 0.10
    MIN_LENGTH: int = 20
    MAX_LENGTH: int = 1000

    ALPHAS: Tuple[float, ...] = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
    )

    N_REPETITIONS: int = 5000

    MASTER_SEED: int = 20260918

    OUTPUT_DIR: str = "experiment4_results"

CFG = Config()

LOG2 = math.log(2.0)

def kt_log2_probability_counts(n1, n0):

    n1 = np.asarray(n1, dtype=np.float64)

    n0 = np.asarray(n0, dtype=np.float64)

    return (betaln(n1 + 0.5, n0 + 0.5) - betaln(0.5, 0.5)) / LOG2

def multiscale_lengths(min_length: int, max_length: int, epsilon: float) -> List[int]:

    if not 0 < epsilon <= 0.5:
        raise ValueError("epsilon trebuie sa fie in (0, 1/2].")

    lengths = []

    m = 1

    while m <= max_length:

        if m >= min_length:
            lengths.append(m)

        new_m = math.ceil((1.0 + epsilon) * m)

        if new_m <= m:
            new_m = m + 1

        m = new_m

    return lengths

def build_multiscale_family(
    N: int,
    min_length: int,
    max_length: int,
    epsilon: float,
) -> pd.DataFrame:

    records = []

    lengths = multiscale_lengths(min_length, max_length, epsilon)

    for scale, m in enumerate(lengths):

        h = max(1, math.floor(epsilon * m))

        starts = np.arange(0, N - m + 1, h, dtype=np.int64)

        for start in starts:

            records.append((scale, m, h, int(start), int(start + m)))

    return pd.DataFrame(records, columns=["scale", "length", "step", "start", "end"])

def generate_null_sequence(rng: np.random.Generator, cfg: Config = CFG) -> np.ndarray:
    """
    Entire sequence is generated under P0.
    There is no anomaly.
    """

    return (rng.random(cfg.N) < cfg.P0).astype(np.int8)

def prefix_ones(X: np.ndarray) -> np.ndarray:

    return np.concatenate([np.array([0], dtype=np.int64), np.cumsum(X, dtype=np.int64)])

def raw_scores_all_intervals(
    X: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:

    pref = prefix_ones(X)

    lengths = ends - starts

    n1 = pref[ends] - pref[starts]

    n0 = lengths - n1

    log_q = kt_log2_probability_counts(n1, n0)

    log_p0 = n1 * math.log2(cfg.P0) + n0 * math.log2(1.0 - cfg.P0)

    raw = log_q - log_p0

    return raw

def log2_total_evalue_uniform(raw_scores: np.ndarray) -> float:

    M = len(raw_scores)

    log_sum_nats = logsumexp(raw_scores * LOG2)

    log2_sum = log_sum_nats / LOG2

    return log2_sum - math.log2(M)

def run_replicate(
    seed: np.random.SeedSequence,
    starts: np.ndarray,
    ends: np.ndarray,
    cfg: Config = CFG,
):

    rng = np.random.default_rng(seed)

    X = generate_null_sequence(rng, cfg)

    t0 = time.perf_counter()

    raw = raw_scores_all_intervals(X, starts, ends, cfg)

    elapsed = time.perf_counter() - t0

    M = len(raw)

    penalty = math.log2(M)

    penalized = raw - penalty

    max_raw = float(np.max(raw))

    max_penalized = float(np.max(penalized))

    best_idx = int(np.argmax(raw))

    log2_e_total = log2_total_evalue_uniform(raw)

    if log2_e_total < 1023:
        e_total = float(np.exp2(log2_e_total))
    else:
        e_total = np.inf

    return {
        "max_raw_score": max_raw,

        "max_penalized_score": max_penalized,

        "best_start": int(starts[best_idx]),

        "best_end": int(ends[best_idx]),

        "best_length": int(ends[best_idx] - starts[best_idx]),

        "log2_e_total": log2_e_total,

        "e_total": e_total,

        "runtime": elapsed,
    }

def run_experiment(cfg: Config = CFG):

    print("=" * 72)

    print("EXPERIMENT 4")

    print("Global false-alarm calibration")

    print("=" * 72)

    print("\nBuilding multiscale family...")

    family = build_multiscale_family(
        N=cfg.N,
        min_length=cfg.MIN_LENGTH,
        max_length=cfg.MAX_LENGTH,
        epsilon=cfg.EPSILON,
    )

    starts = family["start"].to_numpy(dtype=np.int64)

    ends = family["end"].to_numpy(dtype=np.int64)

    M = len(family)

    penalty = math.log2(M)

    print(f"Number of candidates: {M:,}")

    print(f"Number of scales: {family['scale'].nunique()}")

    print(f"Uniform penalty log2(M): {penalty:.4f} bits")

    print(f"Monte Carlo repetitions: {cfg.N_REPETITIONS:,}")

    root = np.random.SeedSequence(cfg.MASTER_SEED)

    seeds = root.spawn(cfg.N_REPETITIONS)

    rows = []

    t_total = time.perf_counter()

    for r in range(cfg.N_REPETITIONS):

        row = run_replicate(
            seed=seeds[r],
            starts=starts,
            ends=ends,
            cfg=cfg,
        )

        row["replicate"] = r

        rows.append(row)

        if r == 0 or (r + 1) % 250 == 0:

            elapsed = time.perf_counter() - t_total

            print(f"  {r + 1:5d}/{cfg.N_REPETITIONS}   elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - t_total

    print(f"\nTotal runtime: {elapsed:.2f}s")

    return pd.DataFrame(rows), family

def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1.0 + z**2 / n

    centre = (p + z**2 / (2.0 * n)) / denominator

    half_width = z * math.sqrt(
        p * (1.0 - p) / n + z**2 / (4.0 * n**2)
    ) / denominator

    low = max(0.0, centre - half_width)

    high = min(1.0, centre + half_width)

    if successes == 0:
        low = 0.0

    if successes == n:
        high = 1.0

    return low, high

def summarize_false_alarms(
    results: pd.DataFrame,
    family: pd.DataFrame,
    cfg: Config = CFG,
):

    M = len(family)

    penalty = math.log2(M)

    n = len(results)

    rows = []

    for alpha in cfg.ALPHAS:

        threshold_penalized = math.log2(1.0 / alpha)

        threshold_raw_global = penalty + threshold_penalized

        global_alarm = (
            results[
                "max_penalized_score"
            ]
            >=
            threshold_penalized
        )

        global_alarm_raw_equivalent = (
            results[
                "max_raw_score"
            ]
            >=
            threshold_raw_global
        )

        assert np.array_equal(
            global_alarm.to_numpy(),
            global_alarm_raw_equivalent.to_numpy(),
        )

        naive_alarm = (
            results[
                "max_raw_score"
            ]
            >=
            threshold_penalized
        )

        k_global = int(global_alarm.sum())

        k_naive = int(naive_alarm.sum())

        global_low, global_high = (
            wilson_interval(
                k_global,
                n,
            )
        )

        naive_low, naive_high = (
            wilson_interval(
                k_naive,
                n,
            )
        )

        rows.append(
            {
                "alpha":
                    alpha,

                "n":
                    n,

                "global_false_alarms":
                    k_global,

                "global_false_alarm_rate":
                    k_global / n,

                "global_ci_low":
                    global_low,

                "global_ci_high":
                    global_high,

                "naive_false_alarms":
                    k_naive,

                "naive_false_alarm_rate":
                    k_naive / n,

                "naive_ci_low":
                    naive_low,

                "naive_ci_high":
                    naive_high,

                "penalized_threshold_bits":
                    threshold_penalized,

                "raw_global_threshold_bits":
                    threshold_raw_global,

                "penalty_bits":
                    penalty,
            }
        )

    return pd.DataFrame(rows)

def summarize_evalues(results: pd.DataFrame):

    finite = results[
        np.isfinite(
            results[
                "e_total"
            ]
        )
    ]

    e = finite["e_total"].to_numpy()

    return pd.DataFrame(
        [
            {
                "n":
                    len(results),

                "finite_n":
                    len(e),

                "mean_e_total":
                    float(
                        np.mean(e)
                    ),

                "median_e_total":
                    float(
                        np.median(e)
                    ),

                "q90_e_total":
                    float(
                        np.quantile(
                            e,
                            0.90,
                        )
                    ),

                "q95_e_total":
                    float(
                        np.quantile(
                            e,
                            0.95,
                        )
                    ),

                "q99_e_total":
                    float(
                        np.quantile(
                            e,
                            0.99,
                        )
                    ),

                "max_e_total":
                    float(
                        np.max(e)
                    ),

                "mean_log2_e_total":
                    float(
                        results[
                            "log2_e_total"
                        ].mean()
                    ),
            }
        ]
    )

def plot_calibration(summary: pd.DataFrame, output_dir: Path):

    alpha = summary[
        "alpha"
    ].to_numpy()

    observed = summary[
        "global_false_alarm_rate"
    ].to_numpy()

    lower = np.maximum(
        0.0,
        observed
        -
        summary[
            "global_ci_low"
        ].to_numpy()
    )

    upper = np.maximum(
        0.0,
        summary[
            "global_ci_high"
        ].to_numpy()
        -
        observed
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.errorbar(
        alpha,
        observed,
        yerr=np.vstack(
            [
                lower,
                upper,
            ]
        ),
        marker="o",
        capsize=4,
        label="Penalized UAD",
    )

    max_alpha = float(
        max(alpha)
    )

    ax.plot(
        [
            0,
            max_alpha,
        ],
        [
            0,
            max_alpha,
        ],
        linestyle="--",
        label="y = alpha",
    )

    ax.set_xlabel(
        "Nominal alpha"
    )

    ax.set_ylabel(
        "Empirical global false-alarm rate"
    )

    ax.set_title(
        "Global false-alarm calibration"
    )

    ax.set_xlim(
        0,
        max_alpha * 1.05,
    )

    ax.set_ylim(
        0,
        max_alpha * 1.20,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "figure1_global_false_alarm_calibration.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def plot_penalized_vs_naive(summary: pd.DataFrame, output_dir: Path):

    alpha = summary[
        "alpha"
    ].to_numpy()

    penalized = summary[
        "global_false_alarm_rate"
    ].to_numpy()

    naive = summary[
        "naive_false_alarm_rate"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        alpha,
        penalized,
        marker="o",
        label="Penalized UAD",
    )

    ax.plot(
        alpha,
        naive,
        marker="o",
        label="Unpenalized scan",
    )

    ax.plot(
        alpha,
        alpha,
        linestyle="--",
        label="Nominal alpha",
    )

    ax.set_xlabel(
        "Nominal alpha"
    )

    ax.set_ylabel(
        "Global false-alarm rate"
    )

    ax.set_title(
        "Effect of multiple-testing penalization"
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "figure2_penalized_vs_unpenalized.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def plot_max_score_distribution(
    results: pd.DataFrame, output_dir: Path, cfg: Config = CFG
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        results[
            "max_penalized_score"
        ],
        bins=60,
        density=True,
        alpha=0.75,
    )

    for alpha in (
        0.10,
        0.05,
        0.01,
    ):

        threshold = math.log2(
            1.0 / alpha
        )

        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=1.2,
            label=(
                f"alpha={alpha:g}"
            ),
        )

    ax.set_xlabel(
        "Maximum penalized score"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Null distribution of maximum penalized atypicality"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "figure3_max_penalized_score_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def plot_total_evalue_distribution(results: pd.DataFrame, output_dir: Path):

    log_e = results[
        "log2_e_total"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        log_e,
        bins=60,
        density=True,
        alpha=0.75,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="E_total = 1",
    )

    ax.set_xlabel(
        "log2(E_total)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Distribution of the total e-value under the null"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "figure4_total_evalue_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def false_alarm_scale_summary(
    results: pd.DataFrame, family: pd.DataFrame, alpha: float = 0.05
):

    threshold = math.log2(1.0 / alpha)

    alarms = results[
        results[
            "max_penalized_score"
        ]
        >=
        threshold
    ].copy()

    if len(alarms) == 0:

        return pd.DataFrame(
            columns=[
                "best_length",
                "count",
                "fraction",
            ]
        )

    table = (
        alarms
        .groupby(
            "best_length"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    table[
        "fraction"
    ] = (
        table[
            "count"
        ]
        /
        len(alarms)
    )

    return table.sort_values("best_length")

def run_unit_tests(cfg: Config = CFG):

    print("Running unit tests...")

    empty_log_q = float(kt_log2_probability_counts(0, 0))

    assert math.isclose(
        empty_log_q,
        0.0,
        abs_tol=1e-12,
    )

    family = build_multiscale_family(
        N=1000,
        min_length=20,
        max_length=200,
        epsilon=cfg.EPSILON,
    )

    M = len(family)

    w = 1.0 / M

    assert math.isclose(
        M * w,
        1.0,
        abs_tol=1e-12,
    )

    alpha = 0.05

    penalty = math.log2(M)

    pen_threshold = math.log2(1.0 / alpha)

    raw_threshold = math.log2(M / alpha)

    assert math.isclose(
        penalty
        +
        pen_threshold,
        raw_threshold,
        abs_tol=1e-12,
    )

    rng1 = np.random.default_rng(123)

    rng2 = np.random.default_rng(123)

    X1 = generate_null_sequence(rng1, cfg)

    X2 = generate_null_sequence(rng2, cfg)

    assert np.array_equal(X1, X2)

    print("All unit tests passed.")

def main():

    output_dir = Path(CFG.OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            asdict(
                CFG
            )
        ]
    ).to_csv(
        output_dir / "config.csv",
        index=False,
    )

    run_unit_tests(CFG)

    results, family = run_experiment(CFG)

    results.to_csv(
        output_dir / "experiment4_raw_results.csv",
        index=False,
    )

    family.to_csv(
        output_dir / "experiment4_multiscale_family.csv",
        index=False,
    )

    summary = (
        summarize_false_alarms(
            results,
            family,
            CFG,
        )
    )

    summary.to_csv(
        output_dir / "experiment4_false_alarm_summary.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "FALSE-ALARM SUMMARY"
    )

    print("=" * 72)

    print(
        summary[
            [
                "alpha",
                "n",
                "global_false_alarms",
                "global_false_alarm_rate",
                "global_ci_low",
                "global_ci_high",
                "naive_false_alarms",
                "naive_false_alarm_rate",
            ]
        ].to_string(
            index=False
        )
    )

    evalue_summary = summarize_evalues(results)

    evalue_summary.to_csv(
        output_dir / "experiment4_evalue_summary.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "TOTAL E-VALUE SUMMARY"
    )

    print("=" * 72)

    print(
        evalue_summary.to_string(
            index=False
        )
    )

    scale_summary = (
        false_alarm_scale_summary(
            results,
            family,
            alpha=0.05,
        )
    )

    scale_summary.to_csv(
        output_dir / "experiment4_false_alarm_scales_alpha005.csv",
        index=False,
    )

    if len(scale_summary) > 0:

        print(
            "\nFalse alarms by winning interval length "
            "(alpha=0.05):"
        )

        print(
            scale_summary.to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo penalized false alarms "
            "at alpha=0.05."
        )

    plot_calibration(
        summary,
        output_dir,
    )

    plot_penalized_vs_naive(
        summary,
        output_dir,
    )

    plot_max_score_distribution(
        results,
        output_dir,
        CFG,
    )

    plot_total_evalue_distribution(
        results,
        output_dir,
    )

    print("\nExperimentul 4 s-a terminat.")

    print(f"Rezultatele sunt in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()