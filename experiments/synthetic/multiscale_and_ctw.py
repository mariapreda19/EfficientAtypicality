




















from __future__ import annotations
from project_paths import PROJECT_ROOT, DATA_ROOT, DATA_FILES, TRAINING_BASED, KNOWN_TYPICAL, RESULTS_ROOT

import math
import time
import bisect
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import betaln






@dataclass(frozen=True)
class Config:





    N_SCAN: int = 3000

    P0: float = 0.5


    MAX_LENGTH_SCAN: int = 400

    MIN_LENGTH_SCAN: int = 10

    EPSILON: float = 0.10

    ALPHA: float = 0.05

    N_REPETITIONS_SCAN: int = 100



    SCAN_CONDITIONS: Tuple[Tuple[int, float], ...] = (
        (100, 0.80),
        (200, 0.70),
        (200, 0.80),
        (100, 0.90),
    )

    IOU_THRESHOLD: float = 0.5





    N_CTW: int = 10_000

    D_MAX: int = 5

    N_CTW_QUERIES: int = 1000

    CTW_MIN_LENGTH: int = 20
    CTW_MAX_LENGTH: int = 1000


    CTW_RUNTIME_REPEATS: int = 3

    MASTER_SEED: int = 20260918

    OUTPUT_DIR: str = "experiment3_results"


CFG = Config()

LOG2 = math.log(2.0)






def kt_log2_probability_counts(n1, n0):

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


def kt_codelength_counts(n1, n0):

    return -kt_log2_probability_counts(
        n1,
        n0,
    )






def multiscale_lengths(
    min_length: int,
    max_length: int,
    epsilon: float,
) -> List[int]:

    lengths = []

    m = 1

    while m <= max_length:

        if m >= min_length:
            lengths.append(m)

        new_m = math.ceil(
            (1.0 + epsilon) * m
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

    for scale, m in enumerate(lengths):

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






def exhaustive_candidate_count(
    N: int,
    min_length: int,
    max_length: int,
) -> int:

    count = 0

    for m in range(
        min_length,
        max_length + 1,
    ):

        count += (
            N - m + 1
        )

    return count






def centered_interval(
    N: int,
    ell: int,
) -> Tuple[int, int]:

    center = N // 2

    start = (
        center
        -
        ell // 2
    )

    end = start + ell

    return start, end


def generate_sequence(
    N: int,
    ell: int,
    q: float,
    rng: np.random.Generator,
    p0: float = 0.5,
):

    X = (
        rng.random(N)
        <
        p0
    ).astype(np.int8)

    start, end = centered_interval(
        N,
        ell,
    )

    X[start:end] = (
        rng.random(ell)
        <
        q
    ).astype(np.int8)

    return (
        X,
        start,
        end,
    )






def prefix_ones(
    X: np.ndarray,
):

    return np.concatenate(
        (
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                X,
                dtype=np.int64,
            ),
        )
    )






def interval_scores_vectorized(
    pref: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    p0: float,
):

    lengths = (
        ends - starts
    )

    n1 = (
        pref[ends]
        -
        pref[starts]
    )

    n0 = (
        lengths - n1
    )

    typical_length = (
        -n1
        * math.log2(p0)
        -
        n0
        * math.log2(
            1.0 - p0
        )
    )

    universal_length = (
        kt_codelength_counts(
            n1,
            n0,
        )
    )

    return (
        typical_length
        -
        universal_length
    )






def interval_iou(
    a0,
    a1,
    b0,
    b1,
):

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






def scan_multiscale(
    X: np.ndarray,
    family: pd.DataFrame,
    alpha: float,
    p0: float,
):

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

    t0 = time.perf_counter()

    raw = interval_scores_vectorized(
        pref,
        starts,
        ends,
        p0,
    )

    M = len(family)

    penalty = math.log2(M)

    penalized = (
        raw - penalty
    )

    idx = int(
        np.argmax(
            penalized
        )
    )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    threshold = math.log2(
        1.0 / alpha
    )

    return {
        "best_start":
            int(starts[idx]),

        "best_end":
            int(ends[idx]),

        "best_raw":
            float(raw[idx]),

        "best_penalized":
            float(
                penalized[idx]
            ),

        "detected":
            bool(
                penalized[idx]
                >=
                threshold
            ),

        "runtime":
            elapsed,

        "num_candidates":
            M,

        "penalty":
            penalty,
    }






def scan_exhaustive(
    X: np.ndarray,
    min_length: int,
    max_length: int,
    alpha: float,
    p0: float,
):

    N = len(X)

    pref = prefix_ones(
        X
    )

    M = exhaustive_candidate_count(
        N,
        min_length,
        max_length,
    )

    penalty = math.log2(M)

    best_pen = -np.inf
    best_raw = -np.inf
    best_start = None
    best_end = None

    t0 = time.perf_counter()

    for m in range(
        min_length,
        max_length + 1,
    ):

        starts = np.arange(
            0,
            N - m + 1,
            dtype=np.int64,
        )

        ends = (
            starts + m
        )

        raw = interval_scores_vectorized(
            pref,
            starts,
            ends,
            p0,
        )

        penalized = (
            raw - penalty
        )

        idx = int(
            np.argmax(
                penalized
            )
        )

        candidate_pen = float(
            penalized[idx]
        )

        if (
            candidate_pen
            >
            best_pen
        ):

            best_pen = (
                candidate_pen
            )

            best_raw = float(
                raw[idx]
            )

            best_start = int(
                starts[idx]
            )

            best_end = int(
                ends[idx]
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    threshold = math.log2(
        1.0 / alpha
    )

    return {
        "best_start":
            best_start,

        "best_end":
            best_end,

        "best_raw":
            best_raw,

        "best_penalized":
            best_pen,

        "detected":
            bool(
                best_pen
                >=
                threshold
            ),

        "runtime":
            elapsed,

        "num_candidates":
            M,

        "penalty":
            penalty,
    }






def run_experiment_3a(
    cfg: Config = CFG,
):

    print("=" * 72)
    print("EXPERIMENT 3A")
    print("Exhaustive vs multiscale")
    print("=" * 72)

    family = build_multiscale_family(
        cfg.N_SCAN,
        cfg.MIN_LENGTH_SCAN,
        cfg.MAX_LENGTH_SCAN,
        cfg.EPSILON,
    )

    exhaustive_M = (
        exhaustive_candidate_count(
            cfg.N_SCAN,
            cfg.MIN_LENGTH_SCAN,
            cfg.MAX_LENGTH_SCAN,
        )
    )

    print(
        f"\nExhaustive candidates: "
        f"{exhaustive_M:,}"
    )

    print(
        f"Multiscale candidates: "
        f"{len(family):,}"
    )

    reduction = (
        exhaustive_M
        /
        len(family)
    )

    print(
        f"Candidate reduction factor: "
        f"{reduction:.2f}x"
    )

    total_runs = (
        len(cfg.SCAN_CONDITIONS)
        *
        cfg.N_REPETITIONS_SCAN
    )

    root = np.random.SeedSequence(
        cfg.MASTER_SEED
    )

    seeds = root.spawn(
        total_runs
    )

    seed_idx = 0

    rows = []

    for ell, q in cfg.SCAN_CONDITIONS:

        print(
            f"\nell={ell}, q={q}"
        )

        for r in range(
            cfg.N_REPETITIONS_SCAN
        ):

            rng = np.random.default_rng(
                seeds[
                    seed_idx
                ]
            )

            seed_idx += 1

            X, true_start, true_end = (
                generate_sequence(
                    cfg.N_SCAN,
                    ell,
                    q,
                    rng,
                    cfg.P0,
                )
            )





            ex = scan_exhaustive(
                X,
                cfg.MIN_LENGTH_SCAN,
                cfg.MAX_LENGTH_SCAN,
                cfg.ALPHA,
                cfg.P0,
            )

            ex_iou = interval_iou(
                ex["best_start"],
                ex["best_end"],
                true_start,
                true_end,
            )





            ms = scan_multiscale(
                X,
                family,
                cfg.ALPHA,
                cfg.P0,
            )

            ms_iou = interval_iou(
                ms["best_start"],
                ms["best_end"],
                true_start,
                true_end,
            )

            rows.append(
                {
                    "ell":
                        ell,

                    "q":
                        q,

                    "replicate":
                        r,

                    "true_start":
                        true_start,

                    "true_end":
                        true_end,

                    "exhaustive_detected":
                        ex["detected"],

                    "multiscale_detected":
                        ms["detected"],

                    "exhaustive_iou":
                        ex_iou,

                    "multiscale_iou":
                        ms_iou,

                    "exhaustive_localized":
                        (
                            ex["detected"]
                            and
                            ex_iou
                            >=
                            cfg.IOU_THRESHOLD
                        ),

                    "multiscale_localized":
                        (
                            ms["detected"]
                            and
                            ms_iou
                            >=
                            cfg.IOU_THRESHOLD
                        ),

                    "exhaustive_best_raw":
                        ex["best_raw"],

                    "multiscale_best_raw":
                        ms["best_raw"],

                    "exhaustive_best_penalized":
                        ex[
                            "best_penalized"
                        ],

                    "multiscale_best_penalized":
                        ms[
                            "best_penalized"
                        ],

                    "exhaustive_runtime":
                        ex["runtime"],

                    "multiscale_runtime":
                        ms["runtime"],

                    "exhaustive_candidates":
                        ex[
                            "num_candidates"
                        ],

                    "multiscale_candidates":
                        ms[
                            "num_candidates"
                        ],

                    "exhaustive_penalty":
                        ex["penalty"],

                    "multiscale_penalty":
                        ms["penalty"],
                }
            )

            if (
                r == 0
                or
                (r + 1) % 20 == 0
            ):

                print(
                    f"  {r+1:3d}/"
                    f"{cfg.N_REPETITIONS_SCAN}"
                )

    return (
        pd.DataFrame(rows),
        family,
    )






def summarize_3a(
    results: pd.DataFrame,
):

    rows = []

    for (
        ell,
        q,
    ), sub in results.groupby(
        ["ell", "q"]
    ):

        rows.append(
            {
                "ell":
                    ell,

                "q":
                    q,

                "n":
                    len(sub),

                "exhaustive_detection_rate":
                    sub[
                        "exhaustive_detected"
                    ].mean(),

                "multiscale_detection_rate":
                    sub[
                        "multiscale_detected"
                    ].mean(),

                "exhaustive_localization_rate":
                    sub[
                        "exhaustive_localized"
                    ].mean(),

                "multiscale_localization_rate":
                    sub[
                        "multiscale_localized"
                    ].mean(),

                "exhaustive_mean_iou":
                    sub[
                        "exhaustive_iou"
                    ].mean(),

                "multiscale_mean_iou":
                    sub[
                        "multiscale_iou"
                    ].mean(),

                "exhaustive_mean_runtime":
                    sub[
                        "exhaustive_runtime"
                    ].mean(),

                "multiscale_mean_runtime":
                    sub[
                        "multiscale_runtime"
                    ].mean(),

                "runtime_speedup":
                    (
                        sub[
                            "exhaustive_runtime"
                        ].mean()
                        /
                        sub[
                            "multiscale_runtime"
                        ].mean()
                    ),

                "candidate_reduction":
                    (
                        sub[
                            "exhaustive_candidates"
                        ].iloc[0]
                        /
                        sub[
                            "multiscale_candidates"
                        ].iloc[0]
                    ),

                "mean_raw_score_gap_ex_minus_ms":
                    (
                        sub[
                            "exhaustive_best_raw"
                        ]
                        -
                        sub[
                            "multiscale_best_raw"
                        ]
                    ).mean(),

                "mean_penalized_score_gap_ex_minus_ms":
                    (
                        sub[
                            "exhaustive_best_penalized"
                        ]
                        -
                        sub[
                            "multiscale_best_penalized"
                        ]
                    ).mean(),
            }
        )

    return pd.DataFrame(
        rows
    )






def plot_3a_detection(
    summary: pd.DataFrame,
    output_dir: Path,
):

    labels = [
        f"l={int(r.ell)}, q={r.q:.2f}"
        for r in summary.itertuples()
    ]

    x = np.arange(
        len(summary)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        summary[
            "exhaustive_detection_rate"
        ],
        width,
        label="Exhaustive",
    )

    ax.bar(
        x + width / 2,
        summary[
            "multiscale_detection_rate"
        ],
        width,
        label="Multiscale",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Detection probability"
    )

    ax.set_title(
        "Exhaustive vs multiscale detection"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure3a_detection.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


def plot_3a_runtime(
    summary: pd.DataFrame,
    output_dir: Path,
):

    labels = [
        f"l={int(r.ell)}, q={r.q:.2f}"
        for r in summary.itertuples()
    ]

    x = np.arange(
        len(summary)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        summary[
            "exhaustive_mean_runtime"
        ],
        width,
        label="Exhaustive",
    )

    ax.bar(
        x + width / 2,
        summary[
            "multiscale_mean_runtime"
        ],
        width,
        label="Multiscale",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Mean runtime (seconds)"
    )

    ax.set_title(
        "Runtime: exhaustive vs multiscale"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure3a_runtime.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def context_to_int(
    X: np.ndarray,
    start: int,
    end: int,
) -> int:


    value = 0

    for bit in X[
        start:end
    ]:

        value = (
            value << 1
        ) | int(bit)

    return value


def log2_kt_from_counts(
    zeros: int,
    ones: int,
) -> float:

    return (
        betaln(
            ones + 0.5,
            zeros + 0.5,
        )
        -
        betaln(
            0.5,
            0.5,
        )
    ) / LOG2


def log2sumexp2(
    a: float,
    b: float,
) -> float:
    """
    log2(2^a + 2^b)
    """

    m = max(
        a,
        b,
    )

    return (
        m
        +
        math.log2(
            2**(a - m)
            +
            2**(b - m)
        )
    )






def build_occurrence_lists(
    X: np.ndarray,
    D_max: int,
):

    lists = defaultdict(list)

    N = len(X)

    for t in range(N):

        symbol = int(
            X[t]
        )


        lists[
            (
                0,
                0,
                symbol,
            )
        ].append(t)

        max_j = min(
            D_max,
            t,
        )

        for j in range(
            1,
            max_j + 1,
        ):

            ctx = context_to_int(
                X,
                t - j,
                t,
            )

            lists[
                (
                    j,
                    ctx,
                    symbol,
                )
            ].append(t)

    return lists






def indexed_count(
    occurrence_lists,
    j: int,
    ctx: int,
    symbol: int,
    start: int,
    end: int,
) -> int:

    arr = occurrence_lists.get(
        (
            j,
            ctx,
            symbol,
        ),
        [],
    )

    left = bisect.bisect_left(
        arr,
        start,
    )

    right = bisect.bisect_left(
        arr,
        end,
    )

    return (
        right - left
    )






def direct_context_counts(
    X: np.ndarray,
    start: int,
    end: int,
    d: int,
):


    counts = {}

    for j in range(
        0,
        d + 1,
    ):

        max_contexts = (
            1 << j
        )

        for ctx in range(
            max_contexts
        ):

            counts[
                (
                    j,
                    ctx,
                )
            ] = [
                0,
                0,
            ]

    for t in range(
        start,
        end,
    ):

        symbol = int(
            X[t]
        )

        for j in range(
            0,
            d + 1,
        ):

            if t < j:
                continue

            if j == 0:
                ctx = 0
            else:

                ctx = context_to_int(
                    X,
                    t - j,
                    t,
                )

            counts[
                (
                    j,
                    ctx,
                )
            ][symbol] += 1

    return counts






def indexed_context_counts(
    occurrence_lists,
    start: int,
    end: int,
    d: int,
):

    counts = {}

    for j in range(
        0,
        d + 1,
    ):

        for ctx in range(
            1 << j
        ):

            n0 = indexed_count(
                occurrence_lists,
                j,
                ctx,
                0,
                start,
                end,
            )

            n1 = indexed_count(
                occurrence_lists,
                j,
                ctx,
                1,
                start,
                end,
            )

            counts[
                (
                    j,
                    ctx,
                )
            ] = [
                n0,
                n1,
            ]

    return counts






def ctw_log2_from_counts(
    counts,
    d: int,
) -> float:

    memo = {}

    def recurse(
        j: int,
        ctx: int,
    ) -> float:

        key = (
            j,
            ctx,
        )

        if key in memo:
            return memo[key]

        zeros, ones = counts[
            key
        ]

        log_pe = (
            log2_kt_from_counts(
                zeros,
                ones,
            )
        )

        if j == d:

            memo[key] = (
                log_pe
            )

            return log_pe





        left_ctx = ctx

        right_ctx = (
            (1 << j)
            |
            ctx
        )

        log_left = recurse(
            j + 1,
            left_ctx,
        )

        log_right = recurse(
            j + 1,
            right_ctx,
        )

        log_children = (
            log_left
            +
            log_right
        )

        result = (
            log2sumexp2(
                -1.0 + log_pe,
                -1.0 + log_children,
            )
        )

        memo[key] = result

        return result

    return recurse(
        0,
        0,
    )






def direct_ctw_log2_probability(
    X: np.ndarray,
    start: int,
    end: int,
    d: int,
) -> float:

    counts = direct_context_counts(
        X,
        start,
        end,
        d,
    )

    return ctw_log2_from_counts(
        counts,
        d,
    )






def indexed_ctw_log2_probability(
    occurrence_lists,
    start: int,
    end: int,
    d: int,
) -> float:

    counts = indexed_context_counts(
        occurrence_lists,
        start,
        end,
        d,
    )

    return ctw_log2_from_counts(
        counts,
        d,
    )






def generate_ctw_benchmark_sequence(
    N: int,
    rng: np.random.Generator,
):
    """
    Markov-like binary source so that nonzero CTW depths
    actually have structure to model.
    """

    X = np.zeros(
        N,
        dtype=np.int8,
    )

    X[0] = rng.integers(
        0,
        2,
    )

    for t in range(
        1,
        N,
    ):

        if X[t - 1] == 0:
            p1 = 0.20
        else:
            p1 = 0.80

        X[t] = int(
            rng.random()
            <
            p1
        )

    return X






def run_experiment_3b(
    cfg: Config = CFG,
):

    print("\n")
    print("=" * 72)
    print("EXPERIMENT 3B")
    print("Direct CTW vs indexed CTW")
    print("=" * 72)

    rng = np.random.default_rng(
        cfg.MASTER_SEED + 98765
    )

    X = (
        generate_ctw_benchmark_sequence(
            cfg.N_CTW,
            rng,
        )
    )

    print(
        "\nBuilding CTW occurrence index..."
    )

    t0 = time.perf_counter()

    occurrence_lists = (
        build_occurrence_lists(
            X,
            cfg.D_MAX,
        )
    )

    preprocessing_time = (
        time.perf_counter()
        -
        t0
    )

    print(
        f"Preprocessing time: "
        f"{preprocessing_time:.4f}s"
    )





    queries = []

    for _ in range(
        cfg.N_CTW_QUERIES
    ):

        length = int(
            rng.integers(
                cfg.CTW_MIN_LENGTH,
                cfg.CTW_MAX_LENGTH + 1,
            )
        )


        start = int(
            rng.integers(
                cfg.D_MAX,
                cfg.N_CTW
                -
                length
                +
                1,
            )
        )

        end = (
            start + length
        )

        depth = int(
            rng.integers(
                0,
                cfg.D_MAX + 1,
            )
        )

        queries.append(
            (
                start,
                end,
                depth,
            )
        )





    rows = []

    print(
        f"Checking exactness on "
        f"{len(queries)} queries..."
    )

    for i, (
        start,
        end,
        depth,
    ) in enumerate(
        queries
    ):

        direct = (
            direct_ctw_log2_probability(
                X,
                start,
                end,
                depth,
            )
        )

        indexed = (
            indexed_ctw_log2_probability(
                occurrence_lists,
                start,
                end,
                depth,
            )
        )

        diff = abs(
            direct
            -
            indexed
        )

        rows.append(
            {
                "query":
                    i,

                "start":
                    start,

                "end":
                    end,

                "length":
                    end - start,

                "depth":
                    depth,

                "direct_log2_probability":
                    direct,

                "indexed_log2_probability":
                    indexed,

                "absolute_difference_bits":
                    diff,
            }
        )

    accuracy_df = pd.DataFrame(
        rows
    )





    direct_times = []

    indexed_times = []

    for _ in range(
        cfg.CTW_RUNTIME_REPEATS
    ):

        t0 = time.perf_counter()

        for start, end, depth in queries:

            direct_ctw_log2_probability(
                X,
                start,
                end,
                depth,
            )

        direct_times.append(
            time.perf_counter()
            -
            t0
        )

        t0 = time.perf_counter()

        for start, end, depth in queries:

            indexed_ctw_log2_probability(
                occurrence_lists,
                start,
                end,
                depth,
            )

        indexed_times.append(
            time.perf_counter()
            -
            t0
        )

    direct_time = float(
        np.median(
            direct_times
        )
    )

    indexed_time = float(
        np.median(
            indexed_times
        )
    )

    speedup = (
        direct_time
        /
        indexed_time
    )

    summary = pd.DataFrame(
        [
            {
                "N":
                    cfg.N_CTW,

                "D_max":
                    cfg.D_MAX,

                "queries":
                    cfg.N_CTW_QUERIES,

                "preprocessing_time":
                    preprocessing_time,

                "median_direct_total_time":
                    direct_time,

                "median_indexed_total_time":
                    indexed_time,

                "direct_time_per_query":
                    direct_time
                    /
                    cfg.N_CTW_QUERIES,

                "indexed_time_per_query":
                    indexed_time
                    /
                    cfg.N_CTW_QUERIES,

                "speedup_direct_over_indexed":
                    speedup,

                "max_absolute_difference_bits":
                    accuracy_df[
                        "absolute_difference_bits"
                    ].max(),

                "mean_absolute_difference_bits":
                    accuracy_df[
                        "absolute_difference_bits"
                    ].mean(),
            }
        ]
    )

    print(
        "\nCTW EXACTNESS:"
    )

    print(
        f"Max abs difference: "
        f"{summary['max_absolute_difference_bits'].iloc[0]:.3e} bits"
    )

    print(
        f"Mean abs difference: "
        f"{summary['mean_absolute_difference_bits'].iloc[0]:.3e} bits"
    )

    print(
        "\nCTW RUNTIME:"
    )

    print(
        f"Direct:  "
        f"{direct_time:.4f}s"
    )

    print(
        f"Indexed: "
        f"{indexed_time:.4f}s"
    )

    print(
        f"Speedup: "
        f"{speedup:.2f}x"
    )

    return (
        accuracy_df,
        summary,
        X,
    )






def plot_ctw_accuracy(
    accuracy: pd.DataFrame,
    output_dir: Path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        accuracy["length"],
        accuracy[
            "absolute_difference_bits"
        ],
        s=15,
    )

    ax.set_xlabel(
        "Interval length"
    )

    ax.set_ylabel(
        "|Direct - Indexed| (bits)"
    )

    ax.set_title(
        "Indexed CTW exactness"
    )

    ax.set_yscale(
        "symlog",
        linthresh=1e-15,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure3b_ctw_exactness.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def benchmark_ctw_by_length(
    X,
    occurrence_lists,
    cfg: Config = CFG,
):

    rng = np.random.default_rng(
        cfg.MASTER_SEED + 22222
    )

    lengths = [
        25,
        50,
        100,
        200,
        500,
        1000,
    ]

    queries_per_length = 100

    rows = []

    for length in lengths:

        queries = []

        for _ in range(
            queries_per_length
        ):

            start = int(
                rng.integers(
                    cfg.D_MAX,
                    cfg.N_CTW
                    -
                    length
                    +
                    1,
                )
            )

            end = (
                start + length
            )

            depth = (
                cfg.D_MAX
            )

            queries.append(
                (
                    start,
                    end,
                    depth,
                )
            )


        t0 = time.perf_counter()

        for start, end, d in queries:

            direct_ctw_log2_probability(
                X,
                start,
                end,
                d,
            )

        direct_time = (
            time.perf_counter()
            -
            t0
        )


        t0 = time.perf_counter()

        for start, end, d in queries:

            indexed_ctw_log2_probability(
                occurrence_lists,
                start,
                end,
                d,
            )

        indexed_time = (
            time.perf_counter()
            -
            t0
        )

        rows.append(
            {
                "length":
                    length,

                "direct_time_per_query":
                    direct_time
                    /
                    queries_per_length,

                "indexed_time_per_query":
                    indexed_time
                    /
                    queries_per_length,

                "speedup":
                    direct_time
                    /
                    indexed_time,
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_ctw_runtime_by_length(
    benchmark: pd.DataFrame,
    output_dir: Path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        benchmark["length"],
        benchmark[
            "direct_time_per_query"
        ],
        marker="o",
        label="Direct CTW",
    )

    ax.plot(
        benchmark["length"],
        benchmark[
            "indexed_time_per_query"
        ],
        marker="o",
        label="Indexed CTW",
    )

    ax.set_xlabel(
        "Interval length"
    )

    ax.set_ylabel(
        "Time per query (seconds)"
    )

    ax.set_title(
        "CTW query time vs interval length"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure3b_ctw_runtime_by_length.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()






def run_unit_tests():

    print(
        "Running unit tests..."
    )

    rng = np.random.default_rng(
        123
    )

    X = rng.integers(
        0,
        2,
        size=300,
        dtype=np.int8,
    )

    occ = build_occurrence_lists(
        X,
        D_max=4,
    )


    tests = [
        (10, 50, 0),
        (10, 50, 1),
        (20, 100, 2),
        (30, 150, 3),
        (100, 250, 4),
    ]

    for start, end, d in tests:

        a = direct_ctw_log2_probability(
            X,
            start,
            end,
            d,
        )

        b = indexed_ctw_log2_probability(
            occ,
            start,
            end,
            d,
        )

        assert math.isclose(
            a,
            b,
            rel_tol=0,
            abs_tol=1e-10,
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





    run_unit_tests()





    results_3a, family = (
        run_experiment_3a(
            CFG
        )
    )

    summary_3a = summarize_3a(
        results_3a
    )

    results_3a.to_csv(
        output_dir
        /
        "experiment3a_raw_results.csv",
        index=False,
    )

    summary_3a.to_csv(
        output_dir
        /
        "experiment3a_summary.csv",
        index=False,
    )

    family.to_csv(
        output_dir
        /
        "experiment3a_multiscale_family.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "EXPERIMENT 3A SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        summary_3a.to_string(
            index=False
        )
    )

    plot_3a_detection(
        summary_3a,
        output_dir,
    )

    plot_3a_runtime(
        summary_3a,
        output_dir,
    )





    accuracy_3b, summary_3b, X = (
        run_experiment_3b(
            CFG
        )
    )

    accuracy_3b.to_csv(
        output_dir
        /
        "experiment3b_ctw_accuracy.csv",
        index=False,
    )

    summary_3b.to_csv(
        output_dir
        /
        "experiment3b_ctw_summary.csv",
        index=False,
    )

    plot_ctw_accuracy(
        accuracy_3b,
        output_dir,
    )


    occurrence_lists = (
        build_occurrence_lists(
            X,
            CFG.D_MAX,
        )
    )

    runtime_by_length = (
        benchmark_ctw_by_length(
            X,
            occurrence_lists,
            CFG,
        )
    )

    runtime_by_length.to_csv(
        output_dir
        /
        "experiment3b_runtime_by_length.csv",
        index=False,
    )

    print(
        "\nCTW runtime by length:"
    )

    print(
        runtime_by_length.to_string(
            index=False
        )
    )

    plot_ctw_runtime_by_length(
        runtime_by_length,
        output_dir,
    )

    print(
        "\nExperimentul 3 s-a terminat."
    )

    print(
        f"Rezultatele sunt in: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
