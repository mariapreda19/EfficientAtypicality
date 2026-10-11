from project_paths import DATA_ROOT, KNOWN_TYPICAL, RESULTS_ROOT

import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
from bisect import bisect_left
import os

MAT_FILE = str(KNOWN_TYPICAL / "Data40kTosses.mat")
BASELINE_FILE = str(RESULTS_ROOT / 'experiment3_literal_matlab_port.npz')
OUTPUT_FILE = str(RESULTS_ROOT / 'experiment3_indexed_multiscale_penalized.npz')
alpha = 0.05
baseline_runtime = float("nan")
baseline_available = False


class RANDU:
    def __init__(self, seed):
        self.current = int(seed)

    def new(self, n):
        data = np.zeros(n, dtype=np.uint32)
        modv = (1 << 31) - 1
        for i in range(n):
            self.current = (65539 * self.current) & modv
            data[i] = self.current
        return data

    def bits(self, n, s, e=None):
        if e is None:
            e = s
        d = self.new(n)
        nbits = e - s + 1
        b = np.zeros(n * nbits, dtype=np.uint8)
        for i in range(1, nbits + 1):
            bit_position = e - i + 1
            bit_values = ((d.astype(np.uint64) >> (bit_position - 1)) & 1).astype(np.uint8)
            b[i - 1::nbits] = bit_values
        return b


mat = sio.loadmat(MAT_FILE)
original_data = mat["data"].squeeze().astype(np.uint8)
p = float(np.sum(original_data) / len(original_data))

print("ORIGINAL COIN DATA")
print("total tosses:", len(original_data))
print("ones:", int(np.sum(original_data)))
print("zeros:", int(len(original_data) - np.sum(original_data)))
print("p:", p)

N_RANDU = 1000
seed = 7

r = RANDU(seed)
rdata = r.new(N_RANDU)

r = RANDU(seed)
raw_bits = r.bits(N_RANDU, 1, 31)

bdata = np.zeros(N_RANDU, dtype=np.uint8)
bit_sums = np.zeros(N_RANDU, dtype=np.int32)

for i in range(N_RANDU):
    start_bit = i * 31
    end_bit = (i + 1) * 31
    bit_sums[i] = int(np.sum(raw_bits[start_bit:end_bit]))

    if bit_sums[i] >= 16:
        bdata[i] = 1
    else:
        bdata[i] = 0

print()
print("RANDU — PAPER DEFINITION")
print("seed:", seed)
print("values:", N_RANDU)
print("threshold: 15.5")
print("random tie handling: NONE")
print("RANDU ones:", int(np.sum(bdata)))
print("RANDU zeros:", int(len(bdata) - np.sum(bdata)))
print("RANDU empirical p:", float(np.mean(bdata)))
print("sum == 16 cases:", int(np.sum(bit_sums == 16)))
print("First 10 states:", rdata[:10])
print("First 30 output bits:", bdata[:30])

data = original_data[15000:20000].copy()

start = 3000
endpoint = 4000
data[start:endpoint] = bdata[:endpoint - start]

print()
print("EXPERIMENT")
print("N:", len(data))
print("coin source positions: 15001--20000")
print("RANDU insertion: 3001--4000")
print("inserted length:", endpoint - start)
print("coin segment p:", float(np.mean(original_data[15000:20000])))
print("experimental p:", float(np.mean(data)))

x = np.asarray(data, dtype=np.uint8).copy()
N = len(x)
epsilon = 0.1
Dmax = 15
Lmax = 1000

assert N == 5000
assert abs(p - 0.505425) < 1e-12

if os.path.isfile(BASELINE_FILE):
    with np.load(BASELINE_FILE, allow_pickle=False) as z:
        if ('DataTest' in z.files and 'total_runtime' in z.files
                and np.array_equal(z['DataTest'].astype(np.uint8), x)
                and ('maxD' not in z.files or int(z['maxD']) == Dmax)
                and ('maxL' not in z.files or int(z['maxL']) == Lmax)):
            baseline_runtime = float(z['total_runtime'])
            baseline_available = True
        else:
            print("Baseline exists but data or parameters differ; runtime comparison skipped")

print()
print("REDUCED METHOD")
print("N:", N)
print("p:", p)
print("Dmax:", Dmax)
print("Lmax:", Lmax)
print("epsilon:", epsilon)
print("true RANDU interval: [3001,4000]")
print("baseline available for direct comparison:", baseline_available)

typ_prefix = np.zeros(N + 1, dtype=np.float64)

c0 = math.log2(1.0 / (1.0 - p))
c1 = math.log2(1.0 / p)

for i in range(N):
    if x[i]:
        typ_prefix[i + 1] = typ_prefix[i] + c1
    else:
        typ_prefix[i + 1] = typ_prefix[i] + c0


def typical_codelength(a, b):
    return float(typ_prefix[b] - typ_prefix[a])


def logs_int(n):
    if n <= 1:
        return 0.0
    m = 0.0
    k = math.log2(float(n))

    while k > 0:
        m += k
        k = math.log2(k)

    return m


print("BUILDING CONTEXT IDS")
index_start = time.perf_counter()

ctx = []
ctx.append(np.zeros(N, dtype=np.int32))

for d in range(1, Dmax + 1):
    ids = np.full(N, -1, dtype=np.int32)
    if N > d:
        state = 0
        for k in range(d):
            state = (state << 1) | int(x[k])
        ids[d] = state
        mask = (1 << d) - 1
        for t in range(d + 1, N):
            state = ((state << 1) & mask) | int(x[t - 1])
            ids[t] = state
    ctx.append(ids)

occ = []
for d in range(Dmax + 1):
    level = {}
    ids = ctx[d]
    for t in range(N):
        c = int(ids[t])
        if c < 0:
            continue
        bit = int(x[t])
        pair = level.get(c)
        if pair is None:
            pair = [[], []]
            level[c] = pair
        pair[bit].append(t)
    occ.append(level)

del ctx

index_runtime = time.perf_counter() - index_start

print(f"index construction    : {index_runtime:.6f} s")
stored_occurrences = sum(len(pair[0]) + len(pair[1]) for level in occ for pair in level.values())
print("stored occurrences:", stored_occurrences)


def range_count(lst, a, b):
    return bisect_left(lst, b) - bisect_left(lst, a)


def node_counts(d, c, a, b):
    pair = occ[d].get(c)
    if pair is None:
        return 0, 0
    return range_count(pair[0], a, b), range_count(pair[1], a, b)

def kt_logprob(n0, n1):
    n = n0 + n1
    if n == 0:
        return 0.0

    return (math.lgamma(n0 + 0.5) + math.lgamma(n1 + 0.5) - math.lgamma(n + 1.0) - 2.0 * math.lgamma(0.5)) / math.log(2.0)


def log2add(x, y):
    if x == -math.inf:
        return y
    if y == -math.inf:
        return x
    m = max(x, y)
    return m + math.log2(2.0 ** (x - m) + 2.0 ** (y - m))


def indexed_ctw_all_depths(a, b):
    pe = []
    for _ in range(Dmax + 1):
        pe.append({})
    active = [0]
    for d in range(Dmax + 1):
        next_active = []
        for c in active:
            n0, n1 = node_counts(d, c, a, b)
            if n0 == 0 and n1 == 0:
                continue
            pe[d][c] = kt_logprob(n0, n1)
            if d < Dmax:
                next_active.append(c)
                next_active.append((1 << d) + c)
        active = next_active
        if not active and d < Dmax:
            break

    root_logprob = np.zeros(Dmax + 1, dtype=np.float64)
    root_logprob[0] = pe[0].get(0, 0.0)

    for md in range(1, Dmax + 1):
        pw = dict(pe[md])
        for d in range(md - 1, -1, -1):
            parent = {}
            for c, pe_here in pe[d].items():
                child0 = c
                child1 = (1 << d) + c
                split = pw.get(child0, 0.0) + pw.get(child1, 0.0)
                parent[c] = log2add(pe_here, split) - 1.0
            pw = parent
        root_logprob[md] = pw.get(0, 0.0)

    return root_logprob


def direct_counts(a, b, d):
    counts = {}
    for t in range(a, b):
        if t < d:
            continue
        c = 0
        for q in range(t - d, t):
            c = (c << 1) | int(x[q])
        bit = int(x[t])
        if c not in counts:
            counts[c] = [0, 0]
        counts[c][bit] += 1
    return counts


def direct_ctw(a, b, target_depth):
    pe = []
    for d in range(target_depth + 1):
        counts = direct_counts(a, b, d)
        level = {}
        for c, pair in counts.items():
            level[c] = kt_logprob(pair[0], pair[1])
        pe.append(level)
    if target_depth == 0:
        return pe[0].get(0, 0.0)
    pw = dict(pe[target_depth])
    for d in range(target_depth - 1, -1, -1):
        parent = {}
        for c, pe_here in pe[d].items():
            child0 = c
            child1 = (1 << d) + c
            parent[c] = log2add(pe_here, pw.get(child0, 0.0) + pw.get(child1, 0.0)) - 1.0
        pw = parent
    return pw.get(0, 0.0)


depth_values = []
for d in range(Dmax + 1):
    depth_values.append(logs_int(d))
depth_normalizer = math.log2(sum(2.0 ** (-v) for v in depth_values))
depth_penalty = np.asarray(depth_values, dtype=np.float64) + depth_normalizer
assert sum(2.0 ** (-v) for v in depth_penalty) <= 1.0 + 1e-12

print("VALIDATION — DIRECT VS INDEXED")

validation_intervals = [
    (15, 40),
    (100, 175),
    (500, 600),
    (3000, 3100),
    (3388, 3884),
    (3500, 4000)
]

max_count_error = 0
max_logprob_error = 0.0
validation_depths = list(range(Dmax + 1))

for a, b in validation_intervals:
    indexed = indexed_ctw_all_depths(a, b)
    local_error = 0.0

    for d in validation_depths:
        counts = direct_counts(a, b, d)
        for c, pair in counts.items():
            indexed_pair = node_counts(d, c, a, b)
            max_count_error = max(max_count_error, abs(pair[0] - indexed_pair[0]), abs(pair[1] - indexed_pair[1]))
        direct = direct_ctw(a, b, d)
        local_error = max(local_error, abs(float(indexed[d]) - direct))

    max_logprob_error = max(max_logprob_error, local_error)
    print(f"interval [{a + 1},{b}] length={b - a} indexed/direct error={local_error:.3e}")

print("maximum independent count error:", max_count_error)
print("maximum independent logP error:", max_logprob_error)
assert max_count_error == 0
assert max_logprob_error < 1e-9

print("INDEPENDENT EXACTNESS CHECK: PASS")

lengths = []
m = 1

while m <= Lmax:
    lengths.append(m)
    new_m = int(math.ceil((1.0 + epsilon) * m))
    if new_m <= m:
        new_m = m + 1
    m = new_m

lengths = sorted(set(lengths))

print("MULTISCALE FAMILY")
print("number of scales:", len(lengths))
print("smallest scale:", lengths[0])
print("largest scale:", lengths[-1])
print("scales:", lengths)

candidates = []
K = len(lengths)
scale_info = []

for m in lengths:
    h = max(1, int(math.floor(epsilon * m)))

    j_first = (Dmax + h - 1) // h
    j_last = (N - m) // h
    if j_first > j_last:
        continue
    n_k = j_last - j_first + 1
    w = 1.0 / (K * n_k)
    interval_penalty = math.log2(1.0 / w)
    scale_info.append((m, h, j_first, j_last, n_k, w, interval_penalty))

    for j in range(j_first, j_last + 1):
        a = j * h
        candidates.append((a, a + m, m, interval_penalty))

exhaustive_candidates = 0

for i in range(1, N + 1):
    exhaustive_candidates += min(i, Lmax)

total_weight = sum(item[4] * item[5] for item in scale_info)

print("exhaustive candidates :", exhaustive_candidates)
print("multiscale candidates :", len(candidates))
print("candidate reduction   :", f"{exhaustive_candidates / len(candidates):.3f}x")
print("sum of weights         :", f"{total_weight:.12f}")
assert total_weight <= 1.0 + 1e-12
print("RUNNING REDUCED MULTISCALE SCAN")

scan_start = time.perf_counter()

best_score = math.inf
best_interval = None
best_depth = None
best_LU = None
best_L0 = None
threshold = math.log2(1.0 / alpha)

reduced_endpoint_score = np.full(N, np.nan, dtype=np.float64)

for candidate_number, (a, b, m, interval_penalty) in enumerate(candidates, start=1):
    root_logprob = indexed_ctw_all_depths(a, b)
    depth_costs = -root_logprob + depth_penalty

    dstar = int(np.argmin(depth_costs))
    LU = float(depth_costs[dstar])
    L0 = typical_codelength(a, b)
    score = LU - L0 + interval_penalty
    endpoint = b - 1

    if np.isnan(reduced_endpoint_score[endpoint]):
        reduced_endpoint_score[endpoint] = score
    else:
        if score < reduced_endpoint_score[endpoint]:
            reduced_endpoint_score[endpoint] = score

    if score < best_score:
        best_score = score
        best_interval = (a, b)
        best_depth = dstar
        best_LU = LU
        best_L0 = L0

    show_progress = False

    if candidate_number % 5000 == 0:
        show_progress = True

    if candidate_number == len(candidates):
        show_progress = True

    if show_progress:
        elapsed = time.perf_counter() - scan_start
        print(
            f"{candidate_number:6d}/{len(candidates):6d} | "
            f"elapsed {elapsed:8.2f} s | best {best_score: .6f}"
        )

scan_runtime = time.perf_counter() - scan_start
total_reduced_runtime = index_runtime + scan_runtime
global_detection = bool(-best_score >= threshold)

a, b = best_interval

detected_start = a + 1
detected_end = b
detected_length = b - a

true_start = 3001
true_end = 4000

overlap = max(0, min(detected_end, true_end) - max(detected_start, true_start) + 1)
union = detected_length + (true_end - true_start + 1) - overlap
iou = overlap / union
detected_precision = overlap / detected_length
true_coverage = overlap / (true_end - true_start + 1)

speedup_scan = baseline_runtime / scan_runtime if baseline_available else float("nan")
speedup_total = baseline_runtime / total_reduced_runtime if baseline_available else float("nan")

print("REDUCED METHOD RESULT")
print("true interval: [3001,4000]")
print(f"detected interval: [{detected_start},{detected_end}]")
print("detected length:", detected_length)
print("selected depth:", best_depth)
print("L_U:", best_LU)
print("L_0:", best_L0)
print("L_U - L_0 + interval penalty:", best_score)
print("global alpha threshold:", threshold)
print("global detection:", global_detection)
print("declared interval:", (detected_start, detected_end) if global_detection else None)
print("overlap:", overlap)
print("detected precision:", detected_precision)
print("true interval covered :", true_coverage)
print("IoU:", iou)
print("index runtime:", f"{index_runtime:.6f} s")
print("scan runtime:", f"{scan_runtime:.6f} s")
print("total reduced runtime :", f"{total_reduced_runtime:.6f} s")
if baseline_available:
    print("baseline runtime:", f"{baseline_runtime:.6f} s")
    print("scan speedup:", f"{speedup_scan:.3f}x")
    print("total speedup:", f"{speedup_total:.3f}x")
else:
    print("baseline comparison: skipped (no matching original DataTest)")

rho = 1.0 / (1.0 + epsilon) ** 2
guaranteed_subinterval = Lmax * rho - 1.0

print("MULTISCALE GUARANTEE")
print("rho:", rho)
print("for length 500:", guaranteed_subinterval)
print("i.e. at least about   :", int(math.floor(guaranteed_subinterval)), "samples")

plt.figure(figsize=(15, 5))

valid = ~np.isnan(reduced_endpoint_score)

plt.plot(
    np.arange(1, N + 1)[valid],
    reduced_endpoint_score[valid],
    label="Penalized multiscale score (-A)"
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axvspan(3001, 4000, alpha=0.15, label="Inserted RANDU")
plt.axvspan(
    detected_start,
    detected_end,
    alpha=0.15,
    label="Best reduced interval"
)

plt.xlabel("Endpoint")
plt.ylabel("L_U - L_0 + interval penalty")
plt.legend()
plt.show()

print("HISTORICAL ORIGINAL EXHAUSTIVE (not a matched baseline):")
print("best interval: [3389,3884]")
print("length: 496")
print("depth : 1")
print("score: -21.967594438781")
print("runtime: 523.273595 s")

print("OUR REDUCED METHOD:")
print(f"best interval: [{detected_start},{detected_end}]")
print(f"length: {detected_length}")
print(f"depth:{best_depth}")
print(f"penalized score: {best_score:.12f}")
print(f"runtime scan:{scan_runtime:.6f} s")
print(f"runtime total: {total_reduced_runtime:.6f} s")
print(f"candidate reduction: {exhaustive_candidates / len(candidates):.3f}x")
if baseline_available:
    print(f"total speedup: {speedup_total:.3f}x")
else:
    print("total speedup: N/A (unmatched historical baseline)")

np.savez_compressed(
    OUTPUT_FILE,
    DataTest=x,
    typical_prefix=typ_prefix,
    results_best=np.asarray([detected_start, detected_end, detected_length, best_depth, best_score, best_L0, best_LU], dtype=np.float64),
    endpoint_scores=reduced_endpoint_score,
    depth_penalty=depth_penalty,
    epsilon=epsilon,
    alpha=alpha,
    Dmax=Dmax,
    Lmax=Lmax,
    candidate_count=len(candidates),
    exhaustive_count=exhaustive_candidates,
    total_weight=total_weight,
    max_validation_error=max_logprob_error,
    index_runtime=index_runtime,
    scan_runtime=scan_runtime,
    total_runtime=total_reduced_runtime,
    baseline_available=baseline_available,
    baseline_runtime=baseline_runtime,
    global_detection=global_detection,
    seed=seed
)
print("Saved:", OUTPUT_FILE)