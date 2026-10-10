from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import betaln

@dataclass(frozen=True)
class Config:

    N: int = 10_000

    TRUE_START: int = 4000
    TRUE_LENGTH: int = 500

    EPSILON: float = 0.10

    MIN_LENGTH: int = 20
    MAX_LENGTH: int = 1000

    ALPHA: float = 0.05

    N_REPETITIONS: int = 500

    MASTER_SEED: int = 20260918

    P_Z1_TYPICAL: float = 0.5
    P_X1_Z0_TYPICAL: float = 0.1
    P_X1_Z1_TYPICAL: float = 0.9

    P_Z1_CONTEXT_SHIFT: float = 0.8

    P_X1_Z0_VISIBLE: float = 0.1
    P_X1_Z1_VISIBLE: float = 0.6

    P_X1_Z0_HIDDEN: float = 0.9
    P_X1_Z1_HIDDEN: float = 0.1

    IOU_THRESHOLD: float = 0.5

    OUTPUT_DIR: str = "experiment1_results"

    @property
    def TRUE_END(self) -> int:
        return self.TRUE_START + self.TRUE_LENGTH

CFG = Config()

SCENARIOS = [
    "null",
    "context_shift",
    "visible_mechanism_shift",
    "hidden_mechanism_shift",
]

LOG2 = math.log(2.0)

def log2_safe(x: float) -> float:
    return math.log(x) / LOG2

def bernoulli_kl_bits(q: float, p: float) -> float:
    """
    D(Bern(q) || Bern(p)) in bits.
    """
    if not (0 <= q <= 1):
        raise ValueError("q must lie in [0,1].")
    if not (0 < p < 1):
        raise ValueError("p must lie strictly in (0,1).")

    ans = 0.0

    if q > 0:
        ans += q * math.log(q / p)

    if q < 1:
        ans += (1 - q) * math.log((1 - q) / (1 - p))

    return ans / LOG2

def conditional_kl_bits(
    p_z1: float, q_x1_z0: float, q_x1_z1: float,
    p0_x1_z0: float, p0_x1_z1: float,
) -> float:
    """
    D(Q_X|Z || P0_X|Z | Q_Z)
    """
    return (
        (1 - p_z1) * bernoulli_kl_bits(q_x1_z0, p0_x1_z0)
        + p_z1 * bernoulli_kl_bits(q_x1_z1, p0_x1_z1)
    )

def marginal_x1_probability(p_z1: float, p_x1_z0: float, p_x1_z1: float) -> float:
    return (1.0 - p_z1) * p_x1_z0 + p_z1 * p_x1_z1

def kt_log2_probability_from_counts(n1: np.ndarray | int, n0: np.ndarray | int) -> np.ndarray:
    n1 = np.asarray(n1, dtype=np.float64)
    n0 = np.asarray(n0, dtype=np.float64)

    log_prob_nats = betaln(n1 + 0.5, n0 + 0.5) - betaln(0.5, 0.5)

    return log_prob_nats / LOG2

def kt_codelength_from_counts(n1: np.ndarray | int, n0: np.ndarray | int) -> np.ndarray:
    return -kt_log2_probability_from_counts(n1, n0)

def scenario_parameters(scenario: str, cfg: Config = CFG) -> Tuple[float, float, float]:

    if scenario == "null":
        return (
            cfg.P_Z1_TYPICAL,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )

    if scenario == "context_shift":
        return (
            cfg.P_Z1_CONTEXT_SHIFT,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )

    if scenario == "visible_mechanism_shift":
        return (
            cfg.P_Z1_TYPICAL,
            cfg.P_X1_Z0_VISIBLE,
            cfg.P_X1_Z1_VISIBLE,
        )

    if scenario == "hidden_mechanism_shift":
        return (
            cfg.P_Z1_TYPICAL,
            cfg.P_X1_Z0_HIDDEN,
            cfg.P_X1_Z1_HIDDEN,
        )

    raise ValueError(f"Unknown scenario: {scenario}")

def generate_sequence(
    scenario: str, rng: np.random.Generator, cfg: Config = CFG
) -> Tuple[np.ndarray, np.ndarray]:

    N = cfg.N

    Z = rng.binomial(
        1,
        cfg.P_Z1_TYPICAL,
        size=N,
    ).astype(np.int8)

    p_z1_alt, p_x1_z0_alt, p_x1_z1_alt = (
        scenario_parameters(scenario, cfg)
    )

    s = cfg.TRUE_START
    e = cfg.TRUE_END

    Z[s:e] = rng.binomial(
        1,
        p_z1_alt,
        size=e - s,
    ).astype(np.int8)

    p_x_typical = np.where(
        Z == 0,
        cfg.P_X1_Z0_TYPICAL,
        cfg.P_X1_Z1_TYPICAL,
    )

    X = (
        rng.random(N) < p_x_typical
    ).astype(np.int8)

    p_x_alt = np.where(
        Z[s:e] == 0,
        p_x1_z0_alt,
        p_x1_z1_alt,
    )

    X[s:e] = (
        rng.random(e - s) < p_x_alt
    ).astype(np.int8)

    return X, Z

def multiscale_lengths(min_length: int, max_length: int, epsilon: float) -> List[int]:

    if not (0 < epsilon <= 0.5):
        raise ValueError("epsilon must satisfy 0 < epsilon <= 1/2.")

    if min_length < 1:
        raise ValueError("min_length must be >= 1.")

    lengths = []

    m = 1

    while m <= max_length:
        if m >= min_length:
            lengths.append(m)

        next_m = math.ceil((1.0 + epsilon) * m)

        if next_m <= m:
            next_m = m + 1

        m = next_m

    return lengths

def build_multiscale_family(
    N: int,
    min_length: int,
    max_length: int,
    epsilon: float,
) -> pd.DataFrame:

    records = []

    lengths = multiscale_lengths(
        min_length=min_length,
        max_length=max_length,
        epsilon=epsilon,
    )

    for scale_idx, m in enumerate(lengths):

        h = max(
            1,
            math.floor(epsilon * m),
        )

        starts = np.arange(
            0,
            N - m + 1,
            h,
            dtype=np.int64,
        )

        for start in starts:
            records.append(
                (
                    scale_idx,
                    m,
                    h,
                    int(start),
                    int(start + m),
                )
            )

    family = pd.DataFrame(
        records,
        columns=[
            "scale",
            "length",
            "step",
            "start",
            "end",
        ],
    )

    return family

@dataclass
class PrefixCounts:

    x1: np.ndarray

    z1: np.ndarray

    x0z0: np.ndarray
    x1z0: np.ndarray
    x0z1: np.ndarray
    x1z1: np.ndarray

def prefix_sum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.array([0], dtype=np.int64), np.cumsum(a, dtype=np.int64)])

def build_prefix_counts(
    X: np.ndarray,
    Z: np.ndarray,
) -> PrefixCounts:

    x1 = prefix_sum(X == 1)
    z1 = prefix_sum(Z == 1)

    x0z0 = prefix_sum((X == 0) & (Z == 0))
    x1z0 = prefix_sum((X == 1) & (Z == 0))
    x0z1 = prefix_sum((X == 0) & (Z == 1))
    x1z1 = prefix_sum((X == 1) & (Z == 1))

    return PrefixCounts(
        x1=x1,
        z1=z1,
        x0z0=x0z0,
        x1z0=x1z0,
        x0z1=x0z1,
        x1z1=x1z1,
    )

def interval_count(prefix: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:

    return prefix[ends] - prefix[starts]

def ordinary_scores_all_intervals(
    prefixes: PrefixCounts,
    starts: np.ndarray,
    ends: np.ndarray,
    cfg: Config = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    m = ends - starts

    n1 = interval_count(
        prefixes.x1,
        starts,
        ends,
    )

    n0 = m - n1

    p0_x1 = marginal_x1_probability(
        cfg.P_Z1_TYPICAL,
        cfg.P_X1_Z0_TYPICAL,
        cfg.P_X1_Z1_TYPICAL,
    )

    typical_length = (
        -n1 * np.log2(p0_x1)
        -n0 * np.log2(1.0 - p0_x1)
    )

    universal_length = kt_codelength_from_counts(
        n1,
        n0,
    )

    raw_score = (
        typical_length
        -
        universal_length
    )

    return (
        raw_score,
        typical_length,
        universal_length,
    )

def conditional_scores_all_intervals(
    prefixes: PrefixCounts,
    starts: np.ndarray,
    ends: np.ndarray,
    cfg: Config = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n00 = interval_count(
        prefixes.x0z0,
        starts,
        ends,
    )

    n10 = interval_count(
        prefixes.x1z0,
        starts,
        ends,
    )

    n01 = interval_count(
        prefixes.x0z1,
        starts,
        ends,
    )

    n11 = interval_count(
        prefixes.x1z1,
        starts,
        ends,
    )

    p0_0 = cfg.P_X1_Z0_TYPICAL
    p0_1 = cfg.P_X1_Z1_TYPICAL

    typical_length = (
        -n10 * np.log2(p0_0)
        -n00 * np.log2(1.0 - p0_0)
        -n11 * np.log2(p0_1)
        -n01 * np.log2(1.0 - p0_1)
    )

    universal_length = (
        kt_codelength_from_counts(
            n10,
            n00,
        )
        +
        kt_codelength_from_counts(
            n11,
            n01,
        )
    )

    raw_score = (
        typical_length
        -
        universal_length
    )

    return (
        raw_score,
        typical_length,
        universal_length,
    )

def candidate_log2_penalty(n_intervals: int, n_model_classes: int = 2) -> float:

    total_candidates = n_intervals * n_model_classes

    return math.log2(total_candidates)

def detection_threshold(alpha: float) -> float:

    return math.log2(1.0 / alpha)

def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:

    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))

    union = (end_a - start_a) + (end_b - start_b) - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def scan_sequence(
    X: np.ndarray,
    Z: np.ndarray,
    family: pd.DataFrame,
    cfg: Config = CFG,
    return_full_scores: bool = False,
) -> Dict:

    prefixes = build_prefix_counts(
        X,
        Z,
    )

    starts = family["start"].to_numpy(
        dtype=np.int64,
    )

    ends = family["end"].to_numpy(
        dtype=np.int64,
    )

    M = len(family)

    penalty = candidate_log2_penalty(
        M,
        n_model_classes=2,
    )

    threshold = detection_threshold(
        cfg.ALPHA
    )

    t0 = time.perf_counter()

    ordinary_raw, ordinary_lt, ordinary_la = (
        ordinary_scores_all_intervals(
            prefixes,
            starts,
            ends,
            cfg,
        )
    )

    ordinary_pen = (
        ordinary_raw
        -
        penalty
    )

    ordinary_time = time.perf_counter() - t0

    t0 = time.perf_counter()

    conditional_raw, conditional_lt, conditional_la = (
        conditional_scores_all_intervals(
            prefixes,
            starts,
            ends,
            cfg,
        )
    )

    conditional_pen = (
        conditional_raw
        -
        penalty
    )

    conditional_time = time.perf_counter() - t0

    ord_idx = int(
        np.argmax(ordinary_pen)
    )

    cond_idx = int(
        np.argmax(conditional_pen)
    )

    ord_start = int(starts[ord_idx])
    ord_end = int(ends[ord_idx])

    cond_start = int(starts[cond_idx])
    cond_end = int(ends[cond_idx])

    ord_best_pen = float(
        ordinary_pen[ord_idx]
    )

    cond_best_pen = float(
        conditional_pen[cond_idx]
    )

    result = {
        "ordinary_detected":
            ord_best_pen >= threshold,

        "ordinary_best_start":
            ord_start,

        "ordinary_best_end":
            ord_end,

        "ordinary_best_length":
            ord_end - ord_start,

        "ordinary_best_score_raw":
            float(ordinary_raw[ord_idx]),

        "ordinary_best_score_penalized":
            ord_best_pen,

        "ordinary_iou":
            interval_iou(
                ord_start,
                ord_end,
                cfg.TRUE_START,
                cfg.TRUE_END,
            ),

        "conditional_detected":
            cond_best_pen >= threshold,

        "conditional_best_start":
            cond_start,

        "conditional_best_end":
            cond_end,

        "conditional_best_length":
            cond_end - cond_start,

        "conditional_best_score_raw":
            float(conditional_raw[cond_idx]),

        "conditional_best_score_penalized":
            cond_best_pen,

        "conditional_iou":
            interval_iou(
                cond_start,
                cond_end,
                cfg.TRUE_START,
                cfg.TRUE_END,
            ),

        "runtime_ordinary":
            ordinary_time,

        "runtime_conditional":
            conditional_time,

        "num_intervals_scanned":
            M,

        "penalty_bits":
            penalty,

        "threshold_bits":
            threshold,
    }

    if return_full_scores:
        result["_ordinary_raw"] = ordinary_raw
        result["_ordinary_pen"] = ordinary_pen
        result["_conditional_raw"] = conditional_raw
        result["_conditional_pen"] = conditional_pen

    return result

def run_replicate(
    scenario: str,
    replicate: int,
    family: pd.DataFrame,
    seed_sequence: np.random.SeedSequence,
    cfg: Config = CFG,
) -> Dict:

    rng = np.random.default_rng(seed_sequence)

    X, Z = generate_sequence(
        scenario,
        rng,
        cfg,
    )

    result = scan_sequence(
        X,
        Z,
        family,
        cfg,
    )

    pz, px0, px1 = scenario_parameters(
        scenario,
        cfg,
    )

    marginal_x = (
        marginal_x1_probability(
            pz,
            px0,
            px1,
        )
    )

    cond_kl = (
        conditional_kl_bits(
            pz,
            px0,
            px1,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )
    )

    p0_x = marginal_x1_probability(
        cfg.P_Z1_TYPICAL,
        cfg.P_X1_Z0_TYPICAL,
        cfg.P_X1_Z1_TYPICAL,
    )

    marginal_kl = (
        bernoulli_kl_bits(
            marginal_x,
            p0_x,
        )
    )

    result.update(
        {
            "scenario":
                scenario,

            "replicate":
                replicate,

            "N":
                cfg.N,

            "true_start":
                cfg.TRUE_START,

            "true_end":
                cfg.TRUE_END,

            "true_length":
                cfg.TRUE_LENGTH,

            "p_z1_anomaly":
                pz,

            "p_x1_z0_anomaly":
                px0,

            "p_x1_z1_anomaly":
                px1,

            "p_x1_marginal_anomaly":
                marginal_x,

            "theoretical_marginal_kl_bits":
                marginal_kl,

            "theoretical_conditional_kl_bits":
                cond_kl,
        }
    )

    return result

def run_experiment(cfg: Config = CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 72)
    print("EXPERIMENT 1")
    print("Conditional vs ordinary atypicality")
    print("=" * 72)

    print("\nBuilding multiscale family...")

    family = build_multiscale_family(
        N=cfg.N,
        min_length=cfg.MIN_LENGTH,
        max_length=cfg.MAX_LENGTH,
        epsilon=cfg.EPSILON,
    )

    print(
        f"Number of intervals: {len(family):,}"
    )

    print(
        f"Number of scales: "
        f"{family['scale'].nunique()}"
    )

    print(
        f"Smallest scale: "
        f"{family['length'].min()}"
    )

    print(
        f"Largest scale: "
        f"{family['length'].max()}"
    )

    penalty = candidate_log2_penalty(
        len(family),
        2,
    )

    threshold = detection_threshold(
        cfg.ALPHA
    )

    print(
        f"Candidate penalty: "
        f"{penalty:.4f} bits"
    )

    print(
        f"Penalized detection threshold: "
        f"{threshold:.4f} bits"
    )

    print(
        f"Equivalent raw-score threshold: "
        f"{penalty + threshold:.4f} bits"
    )

    root = np.random.SeedSequence(
        cfg.MASTER_SEED
    )

    total_runs = (
        len(SCENARIOS)
        *
        cfg.N_REPETITIONS
    )

    child_seeds = root.spawn(
        total_runs
    )

    results = []

    seed_idx = 0

    experiment_start = time.perf_counter()

    for scenario in SCENARIOS:

        print(
            f"\nScenario: {scenario}"
        )

        scenario_start = (
            time.perf_counter()
        )

        for r in range(
            cfg.N_REPETITIONS
        ):

            row = run_replicate(
                scenario=scenario,
                replicate=r,
                family=family,
                seed_sequence=child_seeds[
                    seed_idx
                ],
                cfg=cfg,
            )

            seed_idx += 1

            results.append(
                row
            )

            if (
                (r + 1) % 50 == 0
                or
                r == 0
            ):
                print(
                    f"  {r + 1:4d}/"
                    f"{cfg.N_REPETITIONS}"
                )

        scenario_elapsed = time.perf_counter() - scenario_start

        print(
            f"  elapsed: "
            f"{scenario_elapsed:.2f}s"
        )

    elapsed = time.perf_counter() - experiment_start

    print(
        f"\nTotal Monte Carlo time: "
        f"{elapsed:.2f}s"
    )

    results_df = pd.DataFrame(
        results
    )

    return results_df, family

def wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> Tuple[float, float]:

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = (
        1
        +
        z**2 / n
    )

    centre = (
        p
        +
        z**2 / (2 * n)
    ) / denominator

    half_width = (
        z
        *
        math.sqrt(
            p * (1 - p) / n
            +
            z**2 / (4 * n**2)
        )
        /
        denominator
    )

    low = centre - half_width
    high = centre + half_width

    low = max(0.0, low)
    high = min(1.0, high)

    if successes == 0:
        low = 0.0

    if successes == n:
        high = 1.0

    return low, high

def summarize_results(results: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:

    records = []

    for scenario in SCENARIOS:

        sub = results[
            results["scenario"]
            ==
            scenario
        ]

        n = len(sub)

        for method in [
            "ordinary",
            "conditional",
        ]:

            detected = (
                sub[
                    f"{method}_detected"
                ]
                .astype(bool)
            )

            localized = (
                detected
                &
                (
                    sub[
                        f"{method}_iou"
                    ]
                    >=
                    cfg.IOU_THRESHOLD
                )
            )

            k_det = int(
                detected.sum()
            )

            k_loc = int(
                localized.sum()
            )

            det_low, det_high = (
                wilson_interval(
                    k_det,
                    n,
                )
            )

            loc_low, loc_high = (
                wilson_interval(
                    k_loc,
                    n,
                )
            )

            records.append(
                {
                    "scenario":
                        scenario,

                    "method":
                        method,

                    "n":
                        n,

                    "detections":
                        k_det,

                    "detection_rate":
                        k_det / n,

                    "detection_ci_low":
                        det_low,

                    "detection_ci_high":
                        det_high,

                    "correct_localizations":
                        k_loc,

                    "localization_rate":
                        k_loc / n,

                    "localization_ci_low":
                        loc_low,

                    "localization_ci_high":
                        loc_high,

                    "median_iou":
                        sub[
                            f"{method}_iou"
                        ].median(),

                    "mean_iou":
                        sub[
                            f"{method}_iou"
                        ].mean(),

                    "median_best_penalized_score":
                        sub[
                            f"{method}_best_score_penalized"
                        ].median(),

                    "mean_best_penalized_score":
                        sub[
                            f"{method}_best_score_penalized"
                        ].mean(),
                }
            )

    return pd.DataFrame(records)

def plot_detection_rates(summary: pd.DataFrame, output_dir: Path):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    scenario_labels = [
        "Null",
        "Context shift",
        "Visible mechanism",
        "Hidden mechanism",
    ]

    x = np.arange(
        len(SCENARIOS)
    )

    width = 0.36

    for idx, method in enumerate(
        ["ordinary", "conditional"]
    ):

        sub = (
            summary[
                summary["method"]
                ==
                method
            ]
            .set_index("scenario")
            .loc[SCENARIOS]
        )

        y = sub[
            "detection_rate"
        ].to_numpy()

        lower = np.maximum(
            0.0,
            y - sub["detection_ci_low"].to_numpy()
        )

        upper = np.maximum(
            0.0,
            sub["detection_ci_high"].to_numpy() - y
        )

        if idx == 0:
            offset = -width / 2
        else:
            offset = width / 2

        ax.bar(
            x + offset,
            y,
            width,
            label=method.capitalize(),
            yerr=np.vstack(
                [lower, upper]
            ),
            capsize=4,
        )

    ax.axhline(
        CFG.ALPHA,
        linestyle="--",
        linewidth=1,
        label=(
            f"alpha = {CFG.ALPHA}"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        scenario_labels
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Detection probability"
    )

    ax.set_title(
        "Ordinary vs conditional atypicality"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "figure1_detection_probability.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def plot_localization_rates(summary: pd.DataFrame, output_dir: Path):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    scenario_labels = [
        "Null",
        "Context shift",
        "Visible mechanism",
        "Hidden mechanism",
    ]

    x = np.arange(
        len(SCENARIOS)
    )

    width = 0.36

    for idx, method in enumerate(
        ["ordinary", "conditional"]
    ):

        sub = (
            summary[
                summary["method"]
                ==
                method
            ]
            .set_index("scenario")
            .loc[SCENARIOS]
        )

        y = sub[
            "localization_rate"
        ].to_numpy()

        lower = np.maximum(
            0.0,
            y - sub["localization_ci_low"].to_numpy()
        )

        upper = np.maximum(
            0.0,
            sub["localization_ci_high"].to_numpy() - y
        )

        if idx == 0:
            offset = -width / 2
        else:
            offset = width / 2

        ax.bar(
            x + offset,
            y,
            width,
            label=method.capitalize(),
            yerr=np.vstack(
                [lower, upper]
            ),
            capsize=4,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        scenario_labels
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        f"P(detected and IoU ≥ "
        f"{CFG.IOU_THRESHOLD})"
    )

    ax.set_title(
        "Correct localization probability"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "figure2_localization_probability.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def plot_score_distributions(results: pd.DataFrame, output_dir: Path):

    important_scenarios = [
        "context_shift",
        "hidden_mechanism_shift",
    ]

    for scenario in important_scenarios:

        sub = results[
            results["scenario"]
            ==
            scenario
        ]

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        values = [
            sub[
                "ordinary_best_score_penalized"
            ].to_numpy(),

            sub[
                "conditional_best_score_penalized"
            ].to_numpy(),
        ]

        ax.boxplot(
            values,
            tick_labels=[
                "Ordinary",
                "Conditional",
            ],
            showfliers=False,
        )

        ax.axhline(
            detection_threshold(
                CFG.ALPHA
            ),
            linestyle="--",
            linewidth=1.5,
            label="Detection threshold",
        )

        ax.set_ylabel(
            "Maximum penalized atypicality score (bits)"
        )

        ax.set_title(
            scenario.replace(
                "_",
                " ",
            ).title()
        )

        ax.legend()

        fig.tight_layout()

        path = (
            output_dir
            /
            f"figure3_scores_{scenario}.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()

def plot_representative_hidden_example(
    family: pd.DataFrame, output_dir: Path, cfg: Config = CFG
):

    rng = np.random.default_rng(
        cfg.MASTER_SEED + 999_999
    )

    X, Z = generate_sequence(
        "hidden_mechanism_shift",
        rng,
        cfg,
    )

    scan = scan_sequence(
        X,
        Z,
        family,
        cfg,
        return_full_scores=True,
    )

    unique_lengths = np.sort(
        family["length"].unique()
    )

    chosen_length = int(
        unique_lengths[
            np.argmin(
                np.abs(
                    unique_lengths
                    -
                    cfg.TRUE_LENGTH
                )
            )
        ]
    )

    mask = (
        family["length"].to_numpy()
        ==
        chosen_length
    )

    starts = (
        family.loc[
            mask,
            "start",
        ]
        .to_numpy()
    )

    mids = (
        starts
        +
        chosen_length / 2
    )

    ord_scores = (
        scan[
            "_ordinary_pen"
        ][mask]
    )

    cond_scores = (
        scan[
            "_conditional_pen"
        ][mask]
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        mids,
        ord_scores,
        label="Ordinary",
    )

    ax.plot(
        mids,
        cond_scores,
        label="Conditional",
    )

    ax.axvspan(
        cfg.TRUE_START,
        cfg.TRUE_END,
        alpha=0.15,
        label="True anomaly",
    )

    ax.axhline(
        detection_threshold(
            cfg.ALPHA
        ),
        linestyle="--",
        linewidth=1,
        label="Detection threshold",
    )

    ax.set_xlabel(
        "Interval midpoint"
    )

    ax.set_ylabel(
        "Penalized score (bits)"
    )

    ax.set_title(
        "Representative hidden mechanism shift "
        f"(candidate length = {chosen_length})"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "figure4_hidden_example_profile.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def conditional_raw_score_known_segment(
    X: np.ndarray, Z: np.ndarray, cfg: Config = CFG
) -> float:

    n00 = int(
        np.sum(
            (X == 0)
            &
            (Z == 0)
        )
    )

    n10 = int(
        np.sum(
            (X == 1)
            &
            (Z == 0)
        )
    )

    n01 = int(
        np.sum(
            (X == 0)
            &
            (Z == 1)
        )
    )

    n11 = int(
        np.sum(
            (X == 1)
            &
            (Z == 1)
        )
    )

    typical_length = (
        -n10
        * math.log2(
            cfg.P_X1_Z0_TYPICAL
        )
        -
        n00
        * math.log2(
            1
            -
            cfg.P_X1_Z0_TYPICAL
        )
        -
        n11
        * math.log2(
            cfg.P_X1_Z1_TYPICAL
        )
        -
        n01
        * math.log2(
            1
            -
            cfg.P_X1_Z1_TYPICAL
        )
    )

    universal_length = float(
        kt_codelength_from_counts(
            n10,
            n00,
        )
        +
        kt_codelength_from_counts(
            n11,
            n01,
        )
    )

    return (
        typical_length
        -
        universal_length
    )

def run_kl_convergence_experiment(
    cfg: Config = CFG, repetitions: int = 500
) -> pd.DataFrame:

    sample_sizes = [
        25,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
    ]

    root = np.random.SeedSequence(
        cfg.MASTER_SEED + 123456
    )

    child_seeds = root.spawn(
        len(sample_sizes)
        *
        repetitions
    )

    rows = []

    seed_idx = 0

    p_z1 = cfg.P_Z1_TYPICAL
    p_x_z0 = cfg.P_X1_Z0_HIDDEN
    p_x_z1 = cfg.P_X1_Z1_HIDDEN

    theoretical_d = (
        conditional_kl_bits(
            p_z1,
            p_x_z0,
            p_x_z1,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )
    )

    print("\nKL convergence experiment")
    print(
        f"Theoretical conditional KL: "
        f"{theoretical_d:.6f} bits/sample"
    )

    for n in sample_sizes:

        for r in range(
            repetitions
        ):

            rng = np.random.default_rng(
                child_seeds[
                    seed_idx
                ]
            )

            seed_idx += 1

            Z = rng.binomial(
                1,
                p_z1,
                size=n,
            ).astype(np.int8)

            p_x = np.where(
                Z == 0,
                p_x_z0,
                p_x_z1,
            )

            X = (
                rng.random(n)
                <
                p_x
            ).astype(np.int8)

            score = (
                conditional_raw_score_known_segment(
                    X,
                    Z,
                    cfg,
                )
            )

            rows.append(
                {
                    "n":
                        n,

                    "replicate":
                        r,

                    "score":
                        score,

                    "score_per_sample":
                        score / n,

                    "theoretical_conditional_kl":
                        theoretical_d,
                }
            )

    return pd.DataFrame(
        rows
    )

def plot_kl_convergence(convergence: pd.DataFrame, output_dir: Path):

    grouped = (
        convergence
        .groupby("n")[
            "score_per_sample"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "count",
            ]
        )
        .reset_index()
    )

    grouped["se"] = (
        grouped["std"]
        /
        np.sqrt(
            grouped["count"]
        )
    )

    theoretical_d = float(
        convergence[
            "theoretical_conditional_kl"
        ].iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.errorbar(
        grouped["n"],
        grouped["mean"],
        yerr=1.96 * grouped["se"],
        marker="o",
        capsize=4,
        label="Mean A(X|Z) / n",
    )

    ax.axhline(
        theoretical_d,
        linestyle="--",
        label=(
            "Theoretical conditional KL "
            f"= {theoretical_d:.4f}"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "n"
    )

    ax.set_ylabel(
        "Normalized conditional atypicality "
        "(bits/sample)"
    )

    ax.set_title(
        "Convergence of normalized conditional atypicality"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "figure5_conditional_KL_convergence.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

def raw_kt_score_z(Z: np.ndarray, p_z1_typical: float) -> float:

    n1 = int(
        np.sum(Z == 1)
    )

    n0 = len(Z) - n1

    lt = (
        -n1
        * math.log2(
            p_z1_typical
        )
        -
        n0
        * math.log2(
            1 - p_z1_typical
        )
    )

    la = float(
        kt_codelength_from_counts(
            n1,
            n0,
        )
    )

    return lt - la

def joint_decomposition_sanity_check(cfg: Config = CFG):

    rng = np.random.default_rng(
        cfg.MASTER_SEED + 777
    )

    n = 1000

    Z = rng.binomial(
        1,
        0.5,
        size=n,
    ).astype(np.int8)

    p_x = np.where(
        Z == 0,
        0.9,
        0.1,
    )

    X = (
        rng.random(n)
        <
        p_x
    ).astype(np.int8)

    A_z = raw_kt_score_z(
        Z,
        cfg.P_Z1_TYPICAL,
    )

    A_x_given_z = (
        conditional_raw_score_known_segment(
            X,
            Z,
            cfg,
        )
    )

    A_joint = (
        A_z
        +
        A_x_given_z
    )

    residual = (
        A_joint
        -
        A_z
        -
        A_x_given_z
    )

    assert abs(residual) < 1e-10

    print(
        "Joint decomposition sanity check passed."
    )

    print(
        f"A(Z)       = {A_z:.6f}"
    )

    print(
        f"A(X|Z)     = {A_x_given_z:.6f}"
    )

    print(
        f"A(X,Z)     = {A_joint:.6f}"
    )

    print(
        f"Residual   = {residual:.3e}"
    )

def run_unit_tests(cfg: Config = CFG):

    print("\nRunning unit tests...")

    p0_x = marginal_x1_probability(
        cfg.P_Z1_TYPICAL,
        cfg.P_X1_Z0_TYPICAL,
        cfg.P_X1_Z1_TYPICAL,
    )

    assert math.isclose(
        p0_x,
        0.5,
        abs_tol=1e-12,
    )

    pz, px0, px1 = (
        scenario_parameters(
            "context_shift",
            cfg,
        )
    )

    p_x_context = (
        marginal_x1_probability(
            pz,
            px0,
            px1,
        )
    )

    assert math.isclose(
        p_x_context,
        0.74,
        abs_tol=1e-12,
    )

    d_context = conditional_kl_bits(
        pz,
        px0,
        px1,
        cfg.P_X1_Z0_TYPICAL,
        cfg.P_X1_Z1_TYPICAL,
    )

    assert math.isclose(
        d_context,
        0.0,
        abs_tol=1e-12,
    )

    pz, px0, px1 = (
        scenario_parameters(
            "hidden_mechanism_shift",
            cfg,
        )
    )

    p_x_hidden = (
        marginal_x1_probability(
            pz,
            px0,
            px1,
        )
    )

    assert math.isclose(
        p_x_hidden,
        0.5,
        abs_tol=1e-12,
    )

    marginal_d_hidden = (
        bernoulli_kl_bits(
            p_x_hidden,
            p0_x,
        )
    )

    assert math.isclose(
        marginal_d_hidden,
        0.0,
        abs_tol=1e-12,
    )

    conditional_d_hidden = (
        conditional_kl_bits(
            pz,
            px0,
            px1,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )
    )

    assert math.isclose(
        conditional_d_hidden,
        2.53594000115385,
        rel_tol=1e-10,
    )

    rng1 = np.random.default_rng(
        12345
    )

    rng2 = np.random.default_rng(
        12345
    )

    X1, Z1 = generate_sequence(
        "hidden_mechanism_shift",
        rng1,
        cfg,
    )

    X2, Z2 = generate_sequence(
        "hidden_mechanism_shift",
        rng2,
        cfg,
    )

    assert np.array_equal(
        X1,
        X2,
    )

    assert np.array_equal(
        Z1,
        Z2,
    )

    empty_length = float(
        kt_codelength_from_counts(
            0,
            0,
        )
    )

    assert math.isclose(
        empty_length,
        0.0,
        abs_tol=1e-12,
    )

    print(
        "All unit tests passed."
    )

def print_scenario_table(cfg: Config = CFG):

    rows = []

    p0_x = (
        marginal_x1_probability(
            cfg.P_Z1_TYPICAL,
            cfg.P_X1_Z0_TYPICAL,
            cfg.P_X1_Z1_TYPICAL,
        )
    )

    for scenario in SCENARIOS:

        pz, px0, px1 = (
            scenario_parameters(
                scenario,
                cfg,
            )
        )

        px = (
            marginal_x1_probability(
                pz,
                px0,
                px1,
            )
        )

        d_marginal = (
            bernoulli_kl_bits(
                px,
                p0_x,
            )
        )

        d_conditional = (
            conditional_kl_bits(
                pz,
                px0,
                px1,
                cfg.P_X1_Z0_TYPICAL,
                cfg.P_X1_Z1_TYPICAL,
            )
        )

        rows.append(
            {
                "scenario":
                    scenario,

                "P(Z=1)":
                    pz,

                "P(X=1|Z=0)":
                    px0,

                "P(X=1|Z=1)":
                    px1,

                "P(X=1)":
                    px,

                "marginal_KL_bits":
                    d_marginal,

                "conditional_KL_bits":
                    d_conditional,
            }
        )

    df = pd.DataFrame(
        rows
    )

    print("\nTheoretical scenario table:")
    print(
        df.to_string(
            index=False
        )
    )

    return df

def save_config(cfg: Config, output_dir: Path):

    config_df = pd.DataFrame(
        [
            asdict(cfg)
        ]
    )

    config_df.to_csv(
        output_dir
        /
        "config.csv",
        index=False,
    )

def main():

    output_dir = Path(CFG.OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(CFG, output_dir)

    run_unit_tests(CFG)

    joint_decomposition_sanity_check(CFG)

    scenario_table = (
        print_scenario_table(
            CFG
        )
    )

    scenario_table.to_csv(
        output_dir
        /
        "theoretical_scenarios.csv",
        index=False,
    )

    results, family = (
        run_experiment(
            CFG
        )
    )

    results.to_csv(
        output_dir
        /
        "experiment1_raw_results.csv",
        index=False,
    )

    family.to_csv(
        output_dir
        /
        "multiscale_family.csv",
        index=False,
    )

    try:
        results.to_parquet(
            output_dir
            /
            "experiment1_raw_results.parquet",
            index=False,
        )
    except Exception:
        pass

    summary = summarize_results(
        results,
        CFG,
    )

    summary.to_csv(
        output_dir / "experiment1_summary.csv",
        index=False,
    )

    plot_detection_rates(
        summary,
        output_dir,
    )

    plot_localization_rates(
        summary,
        output_dir,
    )

    plot_score_distributions(
        results,
        output_dir,
    )

    plot_representative_hidden_example(
        family,
        output_dir,
        CFG,
    )

    convergence = run_kl_convergence_experiment(
        CFG,
        repetitions=500,
    )

    convergence.to_csv(
        output_dir / "conditional_KL_convergence.csv",
        index=False,
    )

    plot_kl_convergence(
        convergence,
        output_dir,
    )

if __name__ == "__main__":
    main()