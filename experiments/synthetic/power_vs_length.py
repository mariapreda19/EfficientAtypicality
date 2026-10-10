














from __future__ import annotations
from project_paths import PROJECT_ROOT, DATA_ROOT, DATA_FILES, TRAINING_BASED, KNOWN_TYPICAL, RESULTS_ROOT

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import betaln






@dataclass(frozen=True)
class Config:


    N: int = 10_000


    ANOMALY_CENTER: int = 5_000


    ANOMALY_LENGTHS: Tuple[int, ...] = (
        25,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
    )



    P0: float = 0.5






    Q_VALUES: Tuple[float, ...] = (
        0.55,
        0.60,
        0.65,
        0.70,
        0.80,
        0.90,
        0.95,
    )


    EPSILON: float = 0.10
    MIN_LENGTH: int = 10
    MAX_LENGTH: int = 1000


    ALPHA: float = 0.05



    DELTA: float = 0.10


    N_REPETITIONS: int = 200


    IOU_THRESHOLD: float = 0.5


    MASTER_SEED: int = 20260918

    OUTPUT_DIR: str = "experiment2_results"


CFG = Config()


LOG2 = math.log(2.0)






def bernoulli_kl_bits(q: float, p: float) -> float:
    """
    D(Bern(q) || Bern(p)), in bits.
    """

    ans = 0.0

    if q > 0:
        ans += q * math.log(q / p)

    if q < 1:
        ans += (
            (1.0 - q)
            *
            math.log(
                (1.0 - q)
                /
                (1.0 - p)
            )
        )

    return ans / LOG2


def kt_log2_probability_counts(
    n1,
    n0,
):


    n1 = np.asarray(
        n1,
        dtype=float,
    )

    n0 = np.asarray(
        n0,
        dtype=float,
    )

    return (
        betaln(
            n1 + 0.5,
            n0 + 0.5,
        )
        -
        betaln(
            0.5,
            0.5,
        )
    ) / LOG2


def kt_codelength_counts(
    n1,
    n0,
):
    return -kt_log2_probability_counts(
        n1,
        n0,
    )






def kt_redundancy_bound(
    m: int,
    q: float,
) -> float:

    k = np.arange(
        m + 1,
        dtype=float,
    )

    n0 = m - k

    log_q = (
        k * math.log2(q)
        +
        n0 * math.log2(
            1.0 - q
        )
    )

    log_kt = (
        kt_log2_probability_counts(
            k,
            n0,
        )
    )

    regret = (
        log_q
        -
        log_kt
    )

    return float(
        np.max(regret)
    )






def finite_sample_B(
    q: float,
    p0: float,
) -> float:
    """
    Y = log2(Q(X)/P0(X)).

    B = max(Y) - min(Y).
    """

    y1 = math.log2(
        q / p0
    )

    y0 = math.log2(
        (1.0 - q)
        /
        (1.0 - p0)
    )

    return abs(
        y1 - y0
    )






def multiscale_lengths(
    min_length: int,
    max_length: int,
    epsilon: float,
) -> List[int]:

    if not (
        0 < epsilon <= 0.5
    ):
        raise ValueError(
            "epsilon trebuie sa fie in (0, 1/2]."
        )

    lengths = []


    m = 1

    while m <= max_length:

        if m >= min_length:
            lengths.append(m)

        new_m = math.ceil(
            (1.0 + epsilon)
            *
            m
        )

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

    lengths = multiscale_lengths(
        min_length,
        max_length,
        epsilon,
    )

    for scale, m in enumerate(
        lengths
    ):

        h = max(
            1,
            math.floor(
                epsilon * m
            ),
        )

        starts = np.arange(
            0,
            N - m + 1,
            h,
            dtype=np.int64,
        )

        for s in starts:

            records.append(
                (
                    scale,
                    m,
                    h,
                    int(s),
                    int(s + m),
                )
            )

    return pd.DataFrame(
        records,
        columns=[
            "scale",
            "length",
            "step",
            "start",
            "end",
        ],
    )






def anomaly_interval(
    ell: int,
    cfg: Config = CFG,
) -> Tuple[int, int]:

    start = (
        cfg.ANOMALY_CENTER
        -
        ell // 2
    )

    end = start + ell

    if start < 0 or end > cfg.N:
        raise ValueError(
            "Anomalia nu incape in secventa."
        )

    return start, end


def generate_sequence(
    ell: int,
    q: float,
    rng: np.random.Generator,
    cfg: Config = CFG,
) -> Tuple[np.ndarray, int, int]:


    X = (
        rng.random(cfg.N)
        <
        cfg.P0
    ).astype(np.int8)

    start, end = anomaly_interval(
        ell,
        cfg,
    )

    X[start:end] = (
        rng.random(ell)
        <
        q
    ).astype(np.int8)

    return X, start, end






def prefix_ones(
    X: np.ndarray,
) -> np.ndarray:

    return np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                X,
                dtype=np.int64,
            ),
        ]
    )






def raw_scores_all_intervals(
    X: np.ndarray,
    family: pd.DataFrame,
    cfg: Config = CFG,
) -> np.ndarray:

    pref = prefix_ones(
        X
    )

    starts = family[
        "start"
    ].to_numpy(
        dtype=np.int64
    )

    ends = family[
        "end"
    ].to_numpy(
        dtype=np.int64
    )

    m = (
        ends - starts
    )

    n1 = (
        pref[ends]
        -
        pref[starts]
    )

    n0 = (
        m - n1
    )


    L0 = (
        -n1
        * math.log2(
            cfg.P0
        )
        -
        n0
        * math.log2(
            1.0 - cfg.P0
        )
    )


    LU = kt_codelength_counts(
        n1,
        n0,
    )

    return (
        L0 - LU
    )






def global_penalty_bits(
    n_candidates: int,
) -> float:


    return math.log2(
        n_candidates
    )


def detection_threshold_bits(
    alpha: float,
) -> float:

    return math.log2(
        1.0 / alpha
    )






def interval_iou(
    a0: int,
    a1: int,
    b0: int,
    b1: int,
) -> float:

    inter = max(
        0,
        min(a1, b1)
        -
        max(a0, b0),
    )

    union = (
        (a1 - a0)
        +
        (b1 - b0)
        -
        inter
    )

    if union <= 0:
        return 0.0

    return (
        inter / union
    )






def largest_contained_candidate(
    family: pd.DataFrame,
    true_start: int,
    true_end: int,
) -> Dict:

    contained = family[
        (family["start"] >= true_start)
        &
        (family["end"] <= true_end)
    ]

    if len(contained) == 0:
        raise RuntimeError(
            "Nu exista candidat multiscale continut in anomalie."
        )

    idx = contained[
        "length"
    ].idxmax()

    row = family.loc[idx]

    return {
        "family_index":
            int(idx),

        "start":
            int(row["start"]),

        "end":
            int(row["end"]),

        "length":
            int(row["length"]),

        "scale":
            int(row["scale"]),
    }






def finite_sample_condition(
    m: int,
    q: float,
    n_candidates: int,
    cfg: Config = CFG,
) -> Dict:

    d = bernoulli_kl_bits(
        q,
        cfg.P0,
    )

    R = kt_redundancy_bound(
        m,
        q,
    )

    B = finite_sample_B(
        q,
        cfg.P0,
    )


    w = 1.0 / n_candidates

    lhs = m * d

    redundancy_term = R

    multiple_testing_term = math.log2(
        1.0
        /
        (
            cfg.ALPHA
            *
            w
        )
    )

    fluctuation_term = (
        B
        *
        math.sqrt(
            m
            /
            2.0
            *
            math.log(
                1.0
                /
                cfg.DELTA
            )
        )
    )

    rhs = (
        redundancy_term
        +
        multiple_testing_term
        +
        fluctuation_term
    )

    return {
        "d_bits":
            d,

        "m":
            m,

        "R_bits":
            R,

        "B":
            B,

        "information_bits":
            lhs,

        "multiple_testing_bits":
            multiple_testing_term,

        "fluctuation_bits":
            fluctuation_term,

        "rhs_bits":
            rhs,

        "margin_bits":
            lhs - rhs,

        "condition_satisfied":
            lhs >= rhs,
    }






def run_replicate(
    ell: int,
    q: float,
    family: pd.DataFrame,
    contained_candidate: Dict,
    seed: np.random.SeedSequence,
    cfg: Config = CFG,
) -> Dict:

    rng = np.random.default_rng(
        seed
    )

    X, true_start, true_end = (
        generate_sequence(
            ell,
            q,
            rng,
            cfg,
        )
    )

    t0 = time.perf_counter()

    raw_scores = (
        raw_scores_all_intervals(
            X,
            family,
            cfg,
        )
    )

    runtime = (
        time.perf_counter()
        -
        t0
    )

    M = len(family)

    penalty = global_penalty_bits(
        M
    )

    threshold = detection_threshold_bits(
        cfg.ALPHA
    )

    penalized_scores = (
        raw_scores
        -
        penalty
    )

    best_idx = int(
        np.argmax(
            penalized_scores
        )
    )

    best_row = family.iloc[
        best_idx
    ]

    best_score = float(
        penalized_scores[
            best_idx
        ]
    )

    detected = (
        best_score
        >=
        threshold
    )

    best_start = int(
        best_row["start"]
    )

    best_end = int(
        best_row["end"]
    )

    iou = interval_iou(
        best_start,
        best_end,
        true_start,
        true_end,
    )


    j_idx = (
        contained_candidate[
            "family_index"
        ]
    )

    j_penalized_score = float(
        penalized_scores[
            j_idx
        ]
    )

    j_detected = (
        j_penalized_score
        >=
        threshold
    )

    return {
        "ell":
            ell,

        "q":
            q,

        "d_bits":
            bernoulli_kl_bits(
                q,
                cfg.P0,
            ),

        "true_start":
            true_start,

        "true_end":
            true_end,

        "detected":
            detected,

        "best_start":
            best_start,

        "best_end":
            best_end,

        "best_length":
            best_end - best_start,

        "best_penalized_score":
            best_score,

        "iou":
            iou,

        "localized":
            (
                detected
                and
                iou >= cfg.IOU_THRESHOLD
            ),

        "contained_candidate_length":
            contained_candidate[
                "length"
            ],

        "contained_candidate_score":
            j_penalized_score,

        "contained_candidate_detected":
            j_detected,

        "runtime":
            runtime,
    }






def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    den = 1 + z**2 / n

    center = (
        p
        +
        z**2
        /
        (2 * n)
    ) / den

    half = (
        z
        *
        math.sqrt(
            p * (1 - p) / n
            +
            z**2 / (4 * n**2)
        )
        /
        den
    )

    low = max(
        0.0,
        center - half,
    )

    high = min(
        1.0,
        center + half,
    )

    if successes == 0:
        low = 0.0

    if successes == n:
        high = 1.0

    return low, high






def run_experiment(
    cfg: Config = CFG,
):

    print("=" * 72)
    print("EXPERIMENT 2")
    print("Power vs anomaly length / strength")
    print("=" * 72)





    print(
        "\nConstruim familia multiscale..."
    )

    family = (
        build_multiscale_family(
            N=cfg.N,
            min_length=cfg.MIN_LENGTH,
            max_length=cfg.MAX_LENGTH,
            epsilon=cfg.EPSILON,
        )
    )

    print(
        f"Numar intervale: "
        f"{len(family):,}"
    )

    print(
        f"Numar scale: "
        f"{family['scale'].nunique()}"
    )

    print(
        f"Lungime minima: "
        f"{family['length'].min()}"
    )

    print(
        f"Lungime maxima: "
        f"{family['length'].max()}"
    )

    penalty = global_penalty_bits(
        len(family)
    )

    threshold = (
        detection_threshold_bits(
            cfg.ALPHA
        )
    )

    print(
        f"Penalizare globala: "
        f"{penalty:.4f} bits"
    )

    print(
        f"Threshold penalizat: "
        f"{threshold:.4f} bits"
    )

    print(
        f"Threshold raw echivalent: "
        f"{penalty + threshold:.4f} bits"
    )





    theory_rows = []

    contained_by_length = {}

    for ell in cfg.ANOMALY_LENGTHS:

        s, e = anomaly_interval(
            ell,
            cfg,
        )

        J = largest_contained_candidate(
            family,
            s,
            e,
        )

        contained_by_length[
            ell
        ] = J

        rho = (
            1.0
            /
            (1.0 + cfg.EPSILON)**2
        )

        guaranteed_lower = (
            rho * ell - 1.0
        )

        for q in cfg.Q_VALUES:

            theory = (
                finite_sample_condition(
                    m=J["length"],
                    q=q,
                    n_candidates=len(family),
                    cfg=cfg,
                )
            )

            theory_rows.append(
                {
                    "ell":
                        ell,

                    "q":
                        q,

                    "actual_contained_m":
                        J["length"],

                    "guaranteed_lower_m":
                        guaranteed_lower,

                    **theory,
                }
            )

    theory_df = pd.DataFrame(
        theory_rows
    )





    total_conditions = (
        len(cfg.ANOMALY_LENGTHS)
        *
        len(cfg.Q_VALUES)
    )

    total_runs = (
        total_conditions
        *
        cfg.N_REPETITIONS
    )

    root = np.random.SeedSequence(
        cfg.MASTER_SEED
    )

    seeds = root.spawn(
        total_runs
    )

    seed_idx = 0

    rows = []

    start_total = (
        time.perf_counter()
    )





    for ell in cfg.ANOMALY_LENGTHS:

        J = contained_by_length[
            ell
        ]

        for q in cfg.Q_VALUES:

            d = bernoulli_kl_bits(
                q,
                cfg.P0,
            )

            print(
                f"\nell={ell:4d}, "
                f"q={q:.2f}, "
                f"d={d:.4f} bits, "
                f"J={J['length']}"
            )

            for r in range(
                cfg.N_REPETITIONS
            ):

                result = run_replicate(
                    ell=ell,
                    q=q,
                    family=family,
                    contained_candidate=J,
                    seed=seeds[
                        seed_idx
                    ],
                    cfg=cfg,
                )

                seed_idx += 1

                result[
                    "replicate"
                ] = r

                rows.append(
                    result
                )

                if (
                    r == 0
                    or
                    (r + 1) % 50 == 0
                ):
                    print(
                        f"  {r+1:4d}/"
                        f"{cfg.N_REPETITIONS}"
                    )

    elapsed = (
        time.perf_counter()
        -
        start_total
    )

    print(
        f"\nTimp total: "
        f"{elapsed:.2f}s"
    )

    return (
        pd.DataFrame(rows),
        theory_df,
        family,
    )






def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = results.groupby(
        ["ell", "q", "d_bits"]
    )

    for (
        ell,
        q,
        d,
    ), sub in grouped:

        n = len(sub)

        n_det = int(
            sub["detected"].sum()
        )

        n_loc = int(
            sub["localized"].sum()
        )

        n_j = int(
            sub[
                "contained_candidate_detected"
            ].sum()
        )

        det_low, det_high = (
            wilson_interval(
                n_det,
                n,
            )
        )

        j_low, j_high = (
            wilson_interval(
                n_j,
                n,
            )
        )

        rows.append(
            {
                "ell":
                    ell,

                "q":
                    q,

                "d_bits":
                    d,

                "n":
                    n,

                "detections":
                    n_det,

                "detection_rate":
                    n_det / n,

                "detection_ci_low":
                    det_low,

                "detection_ci_high":
                    det_high,

                "localization_rate":
                    n_loc / n,

                "contained_candidate_detection_rate":
                    n_j / n,

                "contained_candidate_ci_low":
                    j_low,

                "contained_candidate_ci_high":
                    j_high,

                "mean_iou":
                    sub[
                        "iou"
                    ].mean(),

                "median_iou":
                    sub[
                        "iou"
                    ].median(),

                "mean_best_score":
                    sub[
                        "best_penalized_score"
                    ].mean(),

                "median_best_score":
                    sub[
                        "best_penalized_score"
                    ].median(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["q", "ell"]
        )
        .reset_index(
            drop=True
        )
    )






def plot_power_heatmap(
    summary: pd.DataFrame,
    theory: pd.DataFrame,
    output_dir: Path,
    cfg: Config = CFG,
):

    pivot = summary.pivot(
        index="q",
        columns="ell",
        values="detection_rate",
    )

    q_values = list(
        pivot.index
    )

    ell_values = list(
        pivot.columns
    )

    data = pivot.to_numpy()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(ell_values)
        )
    )

    ax.set_xticklabels(
        ell_values
    )


    y_labels = []

    for q in q_values:

        d = bernoulli_kl_bits(
            q,
            cfg.P0,
        )

        y_labels.append(
            f"{q:.2f}\n(d={d:.3f})"
        )

    ax.set_yticks(
        np.arange(
            len(q_values)
        )
    )

    ax.set_yticklabels(
        y_labels
    )


    for i in range(
        len(q_values)
    ):
        for j in range(
            len(ell_values)
        ):

            ax.text(
                j,
                i,
                f"{data[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )



    for i, q in enumerate(
        q_values
    ):

        for j, ell in enumerate(
            ell_values
        ):

            row = theory[
                (theory["q"] == q)
                &
                (theory["ell"] == ell)
            ]

            if (
                len(row) == 1
                and
                bool(
                    row[
                        "condition_satisfied"
                    ].iloc[0]
                )
            ):

                ax.scatter(
                    j,
                    i,
                    marker="x",
                    s=100,
                    linewidths=2,
                    label=None,
                )

    ax.set_xlabel(
        "Lungimea anomaliei"
    )

    ax.set_ylabel(
        "Intensitate q si KL"
    )

    ax.set_title(
        "Empirical detection power\n"
        "X = conditia finite-sample suficienta este satisfacuta"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Detection probability"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure1_power_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def plot_power_curves(
    summary: pd.DataFrame,
    output_dir: Path,
    cfg: Config = CFG,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for q in cfg.Q_VALUES:

        sub = (
            summary[
                summary["q"] == q
            ]
            .sort_values(
                "ell"
            )
        )

        d = bernoulli_kl_bits(
            q,
            cfg.P0,
        )

        ax.plot(
            sub["ell"],
            sub["detection_rate"],
            marker="o",
            label=(
                f"q={q:.2f}, "
                f"d={d:.3f}"
            ),
        )

    ax.axhline(
        0.90,
        linestyle="--",
        linewidth=1,
        label="Power = 0.90",
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_xlabel(
        "Lungimea anomaliei"
    )

    ax.set_ylabel(
        "Detection probability"
    )

    ax.set_title(
        "Power vs lungimea anomaliei"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure2_power_vs_length.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def plot_localization_heatmap(
    summary: pd.DataFrame,
    output_dir: Path,
):

    pivot = summary.pivot(
        index="q",
        columns="ell",
        values="localization_rate",
    )

    data = pivot.to_numpy()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(pivot.columns)
        )
    )

    ax.set_xticklabels(
        pivot.columns
    )

    ax.set_yticks(
        np.arange(
            len(pivot.index)
        )
    )

    ax.set_yticklabels(
        [
            f"{q:.2f}"
            for q in pivot.index
        ]
    )

    for i in range(
        data.shape[0]
    ):
        for j in range(
            data.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{data[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_xlabel(
        "Lungimea anomaliei"
    )

    ax.set_ylabel(
        "q"
    )

    ax.set_title(
        "Probabilitatea de localizare corecta (IoU >= 0.5)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Localization probability"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure3_localization_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def plot_theory_validation(
    summary: pd.DataFrame,
    theory: pd.DataFrame,
    output_dir: Path,
    cfg: Config = CFG,
):


    merged = summary.merge(
        theory[
            [
                "ell",
                "q",
                "condition_satisfied",
                "margin_bits",
            ]
        ],
        on=[
            "ell",
            "q",
        ],
        how="left",
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for q in cfg.Q_VALUES:

        sub = (
            merged[
                merged["q"] == q
            ]
            .sort_values(
                "ell"
            )
        )

        ax.plot(
            sub["ell"],
            sub[
                "contained_candidate_detection_rate"
            ],
            marker="o",
            label=f"q={q:.2f}",
        )


        guaranteed = sub[
            sub[
                "condition_satisfied"
            ]
        ]

        if len(
            guaranteed
        ) > 0:

            ax.scatter(
                guaranteed["ell"],
                guaranteed[
                    "contained_candidate_detection_rate"
                ],
                marker="x",
                s=100,
            )

    ax.axhline(
        1.0 - cfg.DELTA,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"1-delta = "
            f"{1-cfg.DELTA:.2f}"
        ),
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_xlabel(
        "Lungimea anomaliei"
    )

    ax.set_ylabel(
        "P(candidatul J este detectat)"
    )

    ax.set_title(
        "Validarea empirica a conditiei finite-sample"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure4_finite_sample_validation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def empirical_length_thresholds(
    summary: pd.DataFrame,
    target_power: float = 0.90,
) -> pd.DataFrame:

    rows = []

    for q in sorted(
        summary["q"].unique()
    ):

        sub = (
            summary[
                summary["q"] == q
            ]
            .sort_values(
                "ell"
            )
        )

        hits = sub[
            sub[
                "detection_rate"
            ]
            >=
            target_power
        ]

        if len(hits) > 0:
            empirical_ell = int(
                hits.iloc[0][
                    "ell"
                ]
            )
        else:
            empirical_ell = np.nan

        rows.append(
            {
                "q":
                    q,

                "d_bits":
                    float(
                        sub.iloc[0][
                            "d_bits"
                        ]
                    ),

                "empirical_min_length_for_90_power":
                    empirical_ell,
            }
        )

    return pd.DataFrame(
        rows
    )






def run_unit_tests(
    cfg: Config = CFG,
):

    print(
        "Running unit tests..."
    )


    assert math.isclose(
        bernoulli_kl_bits(
            cfg.P0,
            cfg.P0,
        ),
        0.0,
        abs_tol=1e-12,
    )


    assert (
        bernoulli_kl_bits(
            0.9,
            cfg.P0,
        )
        >
        0
    )


    assert math.isclose(
        float(
            kt_codelength_counts(
                0,
                0,
            )
        ),
        0.0,
        abs_tol=1e-12,
    )


    assert (
        finite_sample_B(
            0.9,
            cfg.P0,
        )
        >
        0
    )


    s, e = anomaly_interval(
        500,
        cfg,
    )

    assert (
        e - s == 500
    )

    print(
        "All unit tests passed."
    )






def main():

    output_dir = Path(
        CFG.OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    pd.DataFrame(
        [asdict(CFG)]
    ).to_csv(
        output_dir
        /
        "config.csv",
        index=False,
    )


    run_unit_tests(
        CFG
    )


    results, theory, family = (
        run_experiment(
            CFG
        )
    )


    results.to_csv(
        output_dir
        /
        "experiment2_raw_results.csv",
        index=False,
    )

    theory.to_csv(
        output_dir
        /
        "experiment2_theory.csv",
        index=False,
    )

    family.to_csv(
        output_dir
        /
        "multiscale_family.csv",
        index=False,
    )


    summary = summarize_results(
        results
    )

    summary.to_csv(
        output_dir
        /
        "experiment2_summary.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        summary.to_string(
            index=False
        )
    )


    threshold_table = (
        empirical_length_thresholds(
            summary,
            target_power=0.90,
        )
    )

    threshold_table.to_csv(
        output_dir
        /
        "empirical_90_power_thresholds.csv",
        index=False,
    )

    print(
        "\nEmpirical minimum lengths "
        "for >= 90% detection:"
    )

    print(
        threshold_table.to_string(
            index=False
        )
    )


    guaranteed = theory[
        theory[
            "condition_satisfied"
        ]
    ].copy()

    print(
        "\nCombinations satisfying "
        "the finite-sample sufficient condition:"
    )

    if len(
        guaranteed
    ) == 0:

        print(
            "None in the tested grid."
        )

    else:

        print(
            guaranteed[
                [
                    "ell",
                    "q",
                    "d_bits",
                    "actual_contained_m",
                    "information_bits",
                    "rhs_bits",
                    "margin_bits",
                ]
            ].to_string(
                index=False
            )
        )


    plot_power_heatmap(
        summary,
        theory,
        output_dir,
        CFG,
    )

    plot_power_curves(
        summary,
        output_dir,
        CFG,
    )

    plot_localization_heatmap(
        summary,
        output_dir,
    )

    plot_theory_validation(
        summary,
        theory,
        output_dir,
        CFG,
    )

    print(
        "\nExperimentul 2 s-a terminat."
    )

    print(
        f"Rezultatele sunt in: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
