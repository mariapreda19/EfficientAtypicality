from project_paths import DATA_ROOT, RESULTS_ROOT

import numpy as np
import scipy.io as sio
import math
import time

from numba import njit
import os

COIN_FILE = str(DATA_ROOT / "Known Typical Model" / "Data40kTosses.mat")
TEXT_FILE = str(DATA_ROOT / "Known Typical Model" / "the home of abraham.txt")

EPSILON = 0.1
DMAX = 15
LMAX = 1000
TRUE_START = 2001
TRUE_END = 2800
alpha = 0.05
TEXT_TIE_SEED = 7
BASELINE_FILE = str(RESULTS_ROOT / "experiment2_literal_matlab_port.npz")
OUTPUT_FILE = str(RESULTS_ROOT / "experiment2_indexed_multiscale_penalized.npz")
baseline_available = False
baseline_runtime = float("nan")

def logs(n):
    if n <= 1:
        return 0.0
    result = 0.0
    value = math.log2(float(n))
    while value > 0:
        result += value
        value = math.log2(value)
    return result

def text_to_word_len(filename):
    tie_rng = np.random.default_rng(TEXT_TIE_SEED)
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    letters = set(range(65, 91)) | set(range(97, 123))
    wordlen = []

    for line in lines:
        counterl = 0
        for j, char in enumerate(line):
            atl = ord(char)
            if atl in letters:
                counterl += 1
            elif counterl != 0:
                wordlen.append(counterl)
                counterl = 0

            if j == len(line) - 1 and counterl != 0:
                wordlen.append(counterl)
                counterl = 0

    wordlen = np.asarray(wordlen, dtype=int)
    bin_data = np.zeros(len(wordlen) - 1, dtype=np.uint8)

    for i in range(1, len(wordlen)):
        difference = wordlen[i] - wordlen[i - 1]

        if difference > 0:
            bin_data[i - 1] = 1
        elif difference < 0:
            bin_data[i - 1] = 0
        else:
            bin_data[i - 1] = int(tie_rng.random() > 0.5)

    return wordlen, bin_data

mat = sio.loadmat(COIN_FILE)
original_data = mat["data"].squeeze().astype(np.uint8)
p = float(np.sum(original_data) / len(original_data))

print("ORIGINAL COIN DATA")
print("total tosses:", len(original_data))
print("ones:", int(np.sum(original_data)))
print("zeros:", int(len(original_data) - np.sum(original_data)))
print("p:", p)

wordlen, DataP = text_to_word_len(TEXT_FILE)

print()
print("TEXT DATA")
print("Number of words:", len(wordlen))
print("Generated binary values:", len(DataP))
print("First 20 word lengths:", wordlen[:20])
print("First 20 text bits:", DataP[:20])
print("Text tie seed:", TEXT_TIE_SEED)

start = 2000
endpoint = start + len(DataP)

data = original_data[:4000].copy()
data[start:endpoint] = DataP

x = np.asarray(data, dtype=np.uint8).copy()
N = len(x)

assert N == 4000
assert len(DataP) == 800
assert abs(p - 0.505425) < 1e-12

if os.path.isfile(BASELINE_FILE):
    with np.load(BASELINE_FILE, allow_pickle=False) as z:
        if ('DataTest' in z.files and 'total_runtime' in z.files
                and np.array_equal(z['DataTest'].astype(np.uint8), x)
                and ('maxD' not in z.files or int(z['maxD']) == DMAX)
                and ('maxL' not in z.files or int(z['maxL']) == LMAX)):
            baseline_runtime = float(z['total_runtime'])
            baseline_available = True
        else:
            print("Baseline exists but data or parameters differ; runtime comparison skipped")

print()
print("EXPERIMENT")
print("N:", N)
print("text insertion:", f"[{start + 1},{endpoint}]")
print("inserted bits:", len(DataP))
print("Dmax:", DMAX)
print("Lmax:", LMAX)
print("epsilon:", EPSILON)
print("alpha:", alpha)
print("baseline available for direct comparison:", baseline_available)

C0 = np.zeros(N + 1, dtype=np.float64)
c0 = math.log2(1.0 / (1.0 - p))
c1 = math.log2(1.0 / p)

for i in range(N):
    if x[i] == 1:
        C0[i + 1] = C0[i] + c1
    else:
        C0[i + 1] = C0[i] + c0

print()
print("BUILDING CONTEXT IDS")

context_start = time.perf_counter()

context_ids = []
context_ids.append(np.zeros(N, dtype=np.int32))

for d in range(1, DMAX + 1):
    ids = np.full(N, -1, dtype=np.int32)

    if N > d:
        context = 0

        for k in range(d):
            context = (context << 1) | int(x[k])

        ids[d] = context
        mask = (1 << d) - 1

        for t in range(d + 1, N):
            context = ((context << 1) & mask) | int(x[t - 1])
            ids[t] = context

    context_ids.append(ids)

context_time = time.perf_counter() - context_start
print("context construction:", f"{context_time:.6f} s")

print("BUILDING OCCURRENCE INDEX")

occ_start = time.perf_counter()

OFFSETS = np.array([(1 << j) - 1 for j in range(DMAX + 1)], dtype=np.int32)
TOTAL_NODES = (1 << (DMAX + 1)) - 1

occ_lists = {}
for j in range(DMAX + 1):
    for t in range(N):
        c = int(context_ids[j][t])
        if c < 0:
            continue
        node = int(OFFSETS[j]) + c
        key = 2 * node + int(x[t])
        pair = occ_lists.get(key)
        if pair is None:
            pair = []
            occ_lists[key] = pair
        pair.append(t)

occ_lookup = np.full(2 * TOTAL_NODES, -1, dtype=np.int32)
occ_ptr = [0]
occ_values = []

for key in sorted(occ_lists):
    occ_lookup[key] = len(occ_ptr) - 1
    occ_values.extend(occ_lists[key])
    occ_ptr.append(len(occ_values))

occ_ptr = np.asarray(occ_ptr, dtype=np.int32)
occ_values = np.asarray(occ_values, dtype=np.int32)
stored_occurrences = len(occ_values)

del occ_lists

occ_time = time.perf_counter() - occ_start
index_runtime = context_time + occ_time
print("occurrence construction:", f"{occ_time:.6f} s")
print("stored occurrences:", stored_occurrences)

print("BUILDING KT ARRAYS")

kt_start = time.perf_counter()
KT_HALF = np.empty(LMAX + 1, dtype=np.float64)
KT_INT = np.empty(LMAX + 1, dtype=np.float64)
log2_value = math.log(2.0)
log_gamma_half = math.lgamma(0.5)
LOG_HALF_NORM = 2.0 * log_gamma_half

for n in range(LMAX + 1):
    KT_HALF[n] = math.lgamma(n + 0.5)
    KT_INT[n] = math.lgamma(n + 1.0)

kt_time = time.perf_counter() - kt_start
print("KT arrays:", f"{kt_time:.6f} s")

depth_values = []
for d in range(DMAX + 1):
    depth_values.append(logs(d))
depth_normalizer = math.log2(sum(2.0 ** (-v) for v in depth_values))
depth_penalty = np.asarray(depth_values, dtype=np.float64) + depth_normalizer

kraft_sum = float(sum(2.0 ** (-v) for v in depth_penalty))
assert kraft_sum <= 1.0 + 1e-12
print("depth Kraft sum:", f"{kraft_sum:.12f}")

@njit(cache=False)
def lower_bound_numba(arr, lo, hi, value):
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(cache=False)
def indexed_count_numba(node, bit, a, b, occ_lookup, occ_ptr, occ_values):
    idx = occ_lookup[2 * node + bit]
    if idx < 0:
        return 0
    lo = occ_ptr[idx]
    hi = occ_ptr[idx + 1]
    return lower_bound_numba(occ_values, lo, hi, b) - lower_bound_numba(occ_values, lo, hi, a)

@njit(cache=False)
def log2add_numba(a, b):
    if a > b:
        return a + math.log2(1.0 + 2.0 ** (b - a))
    return b + math.log2(1.0 + 2.0 ** (a - b))

@njit(cache=False)
def indexed_ctw_all_depths_numba(a, b, D, offsets, total_nodes, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT):
    pe = np.zeros(total_nodes, dtype=np.float64)
    pw = np.zeros(total_nodes, dtype=np.float64)
    active = np.empty(total_nodes, dtype=np.int32)
    bounds = np.zeros(D + 2, dtype=np.int32)

    active[0] = 0
    bounds[1] = 1
    tail = 1

    for d in range(D + 1):
        for i in range(bounds[d], bounds[d + 1]):
            c = active[i]
            node = offsets[d] + c
            n0 = indexed_count_numba(node, 0, a, b, occ_lookup, occ_ptr, occ_values)
            n1 = indexed_count_numba(node, 1, a, b, occ_lookup, occ_ptr, occ_values)
            if n0 + n1 == 0:
                continue
            pe[node] = (KT_HALF[n0] + KT_HALF[n1] - KT_INT[n0 + n1] - LOG_HALF_NORM) / log2_value
            if d < D:
                active[tail] = c
                active[tail + 1] = (1 << d) + c
                tail += 2
        if d < D:
            bounds[d + 2] = tail

    logQ = np.empty(D + 1, dtype=np.float64)
    logQ[0] = pe[0]

    for target_depth in range(1, D + 1):
        for i in range(bounds[target_depth], bounds[target_depth + 1]):
            c = active[i]
            node = offsets[target_depth] + c
            pw[node] = pe[node]

        for d in range(target_depth - 1, -1, -1):
            for i in range(bounds[d], bounds[d + 1]):
                c = active[i]
                node = offsets[d] + c
                child0 = offsets[d + 1] + c
                child1 = offsets[d + 1] + (1 << d) + c
                pw[node] = log2add_numba(pe[node], pw[child0] + pw[child1]) - 1.0

        logQ[target_depth] = pw[0]

    return logQ

@njit(cache=False)
def evaluate_interval_numba(a, b, D, offsets, total_nodes, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty):
    logQ = indexed_ctw_all_depths_numba(a, b, D, offsets, total_nodes, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT)
    best_length = -logQ[0] + depth_penalty[0]
    best_depth = 0

    for d in range(1, D + 1):
        length = -logQ[d] + depth_penalty[d]
        if length < best_length:
            best_length = length
            best_depth = d

    return best_length, best_depth


def direct_counts(a, b, d):
    counts = {}
    for t in range(a, b):
        if t < d:
            continue
        c = 0
        for q in range(t - d, t):
            c = (c << 1) | int(x[q])
        if c not in counts:
            counts[c] = [0, 0]
        counts[c][int(x[t])] += 1
    return counts


def direct_ctw(a, b, target_depth):
    pe = []
    for d in range(target_depth + 1):
        counts = direct_counts(a, b, d)
        level = {}
        for c, pair in counts.items():
            level[c] = (KT_HALF[pair[0]] + KT_HALF[pair[1]] - KT_INT[pair[0] + pair[1]] - LOG_HALF_NORM) / log2_value
        pe.append(level)
    if target_depth == 0:
        return pe[0].get(0, 0.0)
    pw = dict(pe[target_depth])
    for j in range(target_depth - 1, -1, -1):
        parent = {}
        for c, pe_here in pe[j].items():
            child0 = c
            child1 = (1 << j) + c
            split = pw.get(child0, 0.0) + pw.get(child1, 0.0)
            m = max(pe_here, split)
            parent[c] = m + math.log2(2.0 ** (pe_here - m) + 2.0 ** (split - m)) - 1.0
        pw = parent
    return pw.get(0, 0.0)

print()
print("NUMBA JIT WARM-UP")

jit_start = time.perf_counter()
warm_L, warm_D = evaluate_interval_numba(
    2200, 2500, DMAX, OFFSETS, TOTAL_NODES, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty
)
jit_time = time.perf_counter() - jit_start

print("Compilation + warm-up:", f"{jit_time:.6f} s")
print("Warm-up interval:", f"L={warm_L:.12f}, D={warm_D}")

print()
print("VALIDATION: INDEXED CTW == DIRECT CTW")

test_intervals = [
    (100, 120),
    (500, 600),
    (1900, 2050),
    (2000, 2100),
    (2100, 2300),
    (2200, 2500),
    (2300, 2700),
    (2400, 2800),
    (3200, 3500)
]

max_validation_error = 0.0
max_count_error = 0

for test_start, test_end in test_intervals:
    logQ = indexed_ctw_all_depths_numba(
        test_start, test_end, DMAX, OFFSETS, TOTAL_NODES, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT
    )
    errors = []

    for d in range(DMAX + 1):
        counts = direct_counts(test_start, test_end, d)
        for c, pair in counts.items():
            node = int(OFFSETS[d]) + c
            n0 = indexed_count_numba(node, 0, test_start, test_end, occ_lookup, occ_ptr, occ_values)
            n1 = indexed_count_numba(node, 1, test_start, test_end, occ_lookup, occ_ptr, occ_values)
            max_count_error = max(max_count_error, abs(pair[0] - n0), abs(pair[1] - n1))
        errors.append(abs(float(logQ[d]) - direct_ctw(test_start, test_end, d)))

    local = max(errors)
    max_validation_error = max(max_validation_error, local)
    print(f"[{test_start + 1:4d},{test_end:4d}] indexed/direct error={local:.3e}")

print("maximum count error:", max_count_error)
print("MAX INDEXED/DIRECT CTW ERROR:", f"{max_validation_error:.3e}")
assert max_count_error == 0
assert max_validation_error < 1e-9, "Indexed/direct CTW disagreement; stop before scan"
print("INDEPENDENT EXACTNESS CHECK: PASS")

scales = []
length = 1

while length <= LMAX:
    scales.append(length)
    new_length = int(math.ceil((1.0 + EPSILON) * length))

    if new_length <= length:
        new_length = length + 1

    length = new_length

print()
print("MULTISCALE FAMILY")
print("number of scales:", len(scales))
print("scales:", scales)

K = len(scales)
scale_info = []
for length in scales:
    step = max(1, int(math.floor(EPSILON * length)))
    j_first = (DMAX + step - 1) // step
    j_last = (N - length) // step
    if j_first > j_last:
        continue
    n_here = j_last - j_first + 1
    weight = 1.0 / (K * n_here)
    interval_penalty = math.log2(1.0 / weight)
    scale_info.append((length, step, j_first, j_last, n_here, weight, interval_penalty))

candidate_count = sum(item[4] for item in scale_info)
total_weight = sum(item[4] * item[5] for item in scale_info)
exhaustive_count = sum(N - length + 1 for length in range(1, LMAX + 1))
assert total_weight <= 1.0 + 1e-12

print("Exhaustive candidates:", exhaustive_count)
print("Multiscale candidates:", candidate_count)
print("Sum of weights:", f"{total_weight:.12f}")

@njit(cache=False)
def scan_one_scale_numba(length, step, first_j, last_j, D, offsets, total_nodes, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty, C0, interval_penalty):
    n_here = last_j - first_j + 1
    scores = np.empty(n_here, dtype=np.float64)
    starts = np.empty(n_here, dtype=np.int32)
    ends = np.empty(n_here, dtype=np.int32)
    depths = np.empty(n_here, dtype=np.int32)
    L0s = np.empty(n_here, dtype=np.float64)
    LUs = np.empty(n_here, dtype=np.float64)

    for k in range(n_here):
        start = (first_j + k) * step
        end = start + length
        ctw_length, best_depth = evaluate_interval_numba(
            start, end, D, offsets, total_nodes, occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty
        )

        L0 = C0[end] - C0[start]
        LU = ctw_length
        score = LU - L0 + interval_penalty

        scores[k] = score
        starts[k] = start
        ends[k] = end
        depths[k] = best_depth
        L0s[k] = L0
        LUs[k] = LU

    return scores, starts, ends, depths, L0s, LUs

print()
print("COMPILING SCALE SCANNER")

compile_start = time.perf_counter()
_ = scan_one_scale_numba(
    100, 10, (DMAX + 9) // 10, (N - 100) // 10, DMAX, OFFSETS, TOTAL_NODES,
    occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty, C0, 0.0
)
scanner_compile_time = time.perf_counter() - compile_start
print("Scale scanner compilation:", f"{scanner_compile_time:.6f} s")

print()
print("RUNNING NUMBA INDEXED MULTISCALE SCAN")

scan_start_time = time.perf_counter()

all_scores = []
all_starts = []
all_ends = []
all_lengths = []
all_depths = []
all_L0 = []
all_LU = []

candidate_count_numba = 0

for scale_no, (length, step, j_first, j_last, n_here, weight, interval_penalty) in enumerate(scale_info, start=1):
    scale_start = time.perf_counter()

    scores, starts, ends, depths, L0s, LUs = scan_one_scale_numba(
        length, step, j_first, j_last, DMAX, OFFSETS, TOTAL_NODES,
        occ_lookup, occ_ptr, occ_values, KT_HALF, KT_INT, depth_penalty, C0, interval_penalty
    )

    scale_time = time.perf_counter() - scale_start
    candidate_count_numba += len(scores)

    all_scores.append(scores)
    all_starts.append(starts)
    all_ends.append(ends)
    all_lengths.append(np.full(len(scores), length, dtype=np.int32))
    all_depths.append(depths)
    all_L0.append(L0s)
    all_LU.append(LUs)

    print(
        f"Scale {scale_no:2d}/{len(scales):2d}"
        f" | L={length:3d}"
        f" | h={step:3d}"
        f" | n={len(scores):4d}"
        f" | {scale_time:8.4f} s"
    )

numba_scan_time = time.perf_counter() - scan_start_time
assert candidate_count_numba == candidate_count

scores = np.concatenate(all_scores)
starts = np.concatenate(all_starts)
ends = np.concatenate(all_ends)
lengths = np.concatenate(all_lengths)
depths = np.concatenate(all_depths)
L0s = np.concatenate(all_L0)
LUs = np.concatenate(all_LU)

best_idx = int(np.argmin(scores))
best_score = float(scores[best_idx])
best_start = int(starts[best_idx])
best_end = int(ends[best_idx])
best_length = int(lengths[best_idx])
best_depth = int(depths[best_idx])
best_L0 = float(L0s[best_idx])
best_LU = float(LUs[best_idx])

threshold = math.log2(1.0 / alpha)
global_detection = bool(-best_score >= threshold)

a = TRUE_START - 1
b = TRUE_END
intersection = max(0, min(best_end, b) - max(best_start, a))
union = best_length + (b - a) - intersection
iou_numba = intersection / union

previous_fast_scan = 173.19869039700006
original_baseline = 1175.616066
speedup_vs_previous = previous_fast_scan / numba_scan_time
speedup_vs_original = baseline_runtime / numba_scan_time if baseline_available else float("nan")

total_reduced_runtime = index_runtime + kt_time + numba_scan_time

print()
print("INDEXED MULTISCALE RESULTS")
print("Candidates:", candidate_count_numba)
print("Detected interval (best scanned):", f"{best_start + 1}--{best_end}")
print("Declared interval:", (best_start + 1, best_end) if global_detection else None)
print("Detected length:", best_length)
print("Best depth:", best_depth)
print("L0:", f"{best_L0:.12f}")
print("LU:", f"{best_LU:.12f}")
print("Penalized -A:", f"{best_score:.12f}")
print("Global alpha threshold:", threshold)
print("Global detection:", global_detection)
print("IoU:", f"{iou_numba:.6f}")

print()
print("RUNTIME")
print("JIT warm-up:", f"{jit_time:.6f} s", "(excluded)")
print("Scanner compilation:", f"{scanner_compile_time:.6f} s", "(excluded)")
print("Context preprocessing:", f"{context_time:.6f} s")
print("Occurrence index:", f"{occ_time:.6f} s")
print("KT arrays:", f"{kt_time:.6f} s")
print("NUMBA scan:", f"{numba_scan_time:.6f} s")
print("Indexed total (excluding JIT):", f"{total_reduced_runtime:.6f} s")
print("Previous rolling scan (historical):", f"{previous_fast_scan:.6f} s")
if baseline_available:
    print("Matched original baseline:", f"{baseline_runtime:.6f} s")
    print("Speed-up vs matched original:", f"{speedup_vs_original:.3f}x")
else:
    print("Original baseline comparison: skipped (no matching DataTest)")

order = np.argsort(scores)

print()
print("TOP 20")
print(f"{'#':>3} {'start':>7} {'end':>7} {'len':>6} {'D':>3} {'penalized -A':>14}")

number_to_show = min(20, len(order))

for rank in range(number_to_show):
    idx = int(order[rank])

    print(
        f"{rank + 1:3d} "
        f"{starts[idx] + 1:7d} "
        f"{ends[idx]:7d} "
        f"{lengths[idx]:6d} "
        f"{depths[idx]:3d} "
        f"{scores[idx]:14.6f}"
    )

print()
print("LATEX VALUES")
print(f"indexed_scan_seconds = {numba_scan_time}")
print(f"indexed_total_seconds = {total_reduced_runtime}")
print(f"candidate_count = {candidate_count_numba}")
print(f"detected_interval = [{best_start + 1},{best_end}]")
print(f"best_penalized_score = {-best_score}")
print(f"global_detection = {global_detection}")
print(f"IoU = {iou_numba}")

np.savez_compressed(
    OUTPUT_FILE,
    DataTest=x,
    text_bits=DataP,
    typical_prefix=C0,
    results=np.column_stack((scores, starts + 1, ends, lengths, depths, L0s, LUs)),
    depth_penalty=depth_penalty,
    epsilon=EPSILON,
    alpha=alpha,
    Dmax=DMAX,
    Lmax=LMAX,
    candidate_count=candidate_count_numba,
    exhaustive_count=exhaustive_count,
    total_weight=total_weight,
    kraft_sum=kraft_sum,
    max_validation_error=max_validation_error,
    index_runtime=index_runtime,
    kt_time=kt_time,
    scan_runtime=numba_scan_time,
    total_runtime=total_reduced_runtime,
    baseline_available=baseline_available,
    baseline_runtime=baseline_runtime,
    global_detection=global_detection,
    tie_seed=TEXT_TIE_SEED
)
print("Saved:", OUTPUT_FILE)