from project_paths import DATA_ROOT, RESULTS_ROOT
from pathlib import Path
import numpy as np
import pandas as pd
import math
import random
import time
import matplotlib.pyplot as plt
from bisect import bisect_left
import os

BASE = Path(DATA_ROOT)
epsilon = 0.1
Dmax = 15
Lmax = 1000
alpha = 0.05
HRV_TIE_SEED = 7
BASELINE_FILE = str(RESULTS_ROOT / 'experiment4_literal_matlab_port.npz')
OUTPUT_FILE = str(RESULTS_ROOT / 'experiment4_indexed_multiscale_penalized.npz')
baseline_runtime = float("nan")
baseline_available = False
true_start = 2501
true_end = 4000


def load_coins(path):
    sheet = pd.read_excel(path, sheet_name='Sheet2', header=None, dtype=str)
    seen = {}

    for ncol, scol, hcol, tcol in ((3, 4, 5, 6), (11, 12, 13, 14)):
        for row in sheet.itertuples(index=False, name=None):
            n = str(row[ncol]).strip()
            initial = str(row[scol]).strip()

            if not n.isdecimal() or initial not in ('H', 'T'):
                continue

            i = int(n)
            if i < 1 or i > 40000:
                continue

            head = str(row[hcol]).strip() == '1'
            tail = str(row[tcol]).strip() == '1'

            if head == tail or i in seen:
                raise ValueError(f'Invalid/duplicate toss {i}')

            seen[i] = int(head)

    if len(seen) != 40000 or set(seen) != set(range(1, 40001)):
        raise ValueError('Missing coin toss indices')

    ordered = []
    for i in range(1, 40001):
        ordered.append(seen[i])
    coins = np.array(ordered, dtype=np.uint8)

    if int(coins.sum()) != 20217:
        raise ValueError(f'Heads do not match paper: {int(coins.sum())}')
    return coins


def read_hrv(path):
    tie_rng = random.Random(HRV_TIE_SEED)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for line in f:
            if line.strip():
                lines.append(line.rstrip('\n\r'))

    hrv = []
    labels = []

    for line in lines:
        parts = line.split()
        numeric_candidates = []

        for token in parts:
            try:
                value = float(token)
                numeric_candidates.append(value)
            except ValueError:
                pass

        if len(numeric_candidates) == 0:
            continue

        rr = numeric_candidates[0]
        hrv.append(rr)

        label = None
        for token in parts:
            if len(token) == 1 and token.isalpha():
                label = token

        labels.append(label)

    hrv = np.asarray(hrv, dtype=np.float64)
    hrvb = np.zeros(len(hrv) - 1, dtype=np.uint8)
    equal_count = 0

    for i in range(1, len(hrv)):
        if hrv[i] > hrv[i - 1]:
            hrvb[i - 1] = 1
        elif hrv[i] < hrv[i - 1]:
            hrvb[i - 1] = 0
        else:
            equal_count += 1
            if tie_rng.random() > 0.5:
                hrvb[i - 1] = 1
            else:
                hrvb[i - 1] = 0

    return hrv, hrvb, labels, equal_count


excel_hits = sorted(BASE.rglob('40000tosses.xlsx'))
if len(excel_hits) != 1:
    raise FileNotFoundError(
        f'Expected one 40000tosses.xlsx under {BASE}; found {len(excel_hits)}: {excel_hits[:5]}'
    )

hrv_hits = sorted(BASE.rglob('16786.txt'))
if len(hrv_hits) != 1:
    raise FileNotFoundError(
        f'Expected one 16786.txt under {BASE}; found {len(hrv_hits)}: {hrv_hits[:5]}'
    )

original_data = load_coins(excel_hits[0])
p = float(np.sum(original_data) / len(original_data))

hrv1, hrvb1, label1, hrv_equal_count = read_hrv(hrv_hits[0])

hrv_start = 30000
hrv_endpoint = 31500
hrv_segment = hrvb1[hrv_start:hrv_endpoint].copy()

if len(hrv_segment) != 1500:
    raise ValueError(f'Expected 1500 HRV bits, got {len(hrv_segment)}')

data_hrv = original_data[15000:20000].copy()
insert_start = 2500
insert_endpoint = 4000
data_hrv[insert_start:insert_endpoint] = hrv_segment
data = data_hrv.copy()

x = np.asarray(data, dtype=np.uint8).copy()
N = len(x)

assert N == 5000
assert abs(p - 0.505425) < 1e-12

if os.path.isfile(BASELINE_FILE):
    with np.load(BASELINE_FILE, allow_pickle=False) as z:
        if 'DataTest' in z.files and np.array_equal(z['DataTest'].astype(np.uint8), x):
            baseline_runtime = float(z['total_runtime'])
            baseline_available = True
        else:
            print('Baseline exists but DataTest differs or is missing; runtime comparison skipped')

print('EXPERIMENT 4 — REDUCED METHOD')
print('coin file:', excel_hits[0])
print('HRV file:', hrv_hits[0])
print('coin tosses:', len(original_data))
print('coin heads:', int(np.sum(original_data)))
print('p:', p)
print('HRV source positions: [30001,31500]')
print('HRV segment length:', len(hrv_segment))
print('HRV ones:', int(np.sum(hrv_segment)))
print('HRV zeros:', int(len(hrv_segment) - np.sum(hrv_segment)))
print('N:', N)
print('Dmax:', Dmax)
print('Lmax:', Lmax)
print('epsilon:', epsilon)
print('true HRV interval: [2501,4000]')
print('historical baseline interval: [2859,3293]')
print('historical baseline score   : -114.350081446013')
print('historical baseline runtime : 500.600820 s')
print('baseline available for direct comparison:', baseline_available)
print('HRV tie seed:', HRV_TIE_SEED)


def reduced_logs(n):
    if n <= 1:
        return 0.0
    m = 0.0
    k = math.log2(float(n))
    while k > 0:
        m += k
        k = math.log2(k)
    return m


typ_prefix = np.zeros(N + 1, dtype=np.float64)
cost0 = math.log2(1.0 / (1.0 - p))
cost1 = math.log2(1.0 / p)

for t in range(N):
    if x[t] == 1:
        typ_prefix[t + 1] = typ_prefix[t] + cost1
    else:
        typ_prefix[t + 1] = typ_prefix[t] + cost0


def reduced_typical_codelength(a, b):
    return float(typ_prefix[b] - typ_prefix[a])


print('BUILDING INDEX')

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
print(f'index construction    : {index_runtime:.6f} s')

stored_occurrences = sum(len(pair[0]) + len(pair[1]) for level in occ for pair in level.values())
print('stored occurrences   :', stored_occurrences)


def range_count(lst, a, b):
    return bisect_left(lst, b) - bisect_left(lst, a)


def node_counts(d, c, a, b):
    pair = occ[d].get(c)
    if pair is None:
        return 0, 0
    return range_count(pair[0], a, b), range_count(pair[1], a, b)


def reduced_kt_logprob(n0, n1):
    n = n0 + n1
    if n == 0:
        return 0.0

    return (
        math.lgamma(n0 + 0.5) + math.lgamma(n1 + 0.5)
        - math.lgamma(n + 1.0) - 2.0 * math.lgamma(0.5)
    ) / math.log(2.0)


def reduced_log2add(x1, x2):
    if x1 == -math.inf:
        return x2
    if x2 == -math.inf:
        return x1

    m = max(x1, x2)
    return m + math.log2(2.0 ** (x1 - m) + 2.0 ** (x2 - m))


def indexed_ctw_all_depths_hrv(a, b):
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
            pe[d][c] = reduced_kt_logprob(n0, n1)
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
                parent[c] = reduced_log2add(pe_here, split) - 1.0
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
            level[c] = reduced_kt_logprob(pair[0], pair[1])
        pe.append(level)
    if target_depth == 0:
        return pe[0].get(0, 0.0)
    pw = dict(pe[target_depth])
    for d in range(target_depth - 1, -1, -1):
        parent = {}
        for c, pe_here in pe[d].items():
            child0 = c
            child1 = (1 << d) + c
            parent[c] = reduced_log2add(pe_here, pw.get(child0, 0.0) + pw.get(child1, 0.0)) - 1.0
        pw = parent
    return pw.get(0, 0.0)


depth_values = []
for d in range(Dmax + 1):
    depth_values.append(reduced_logs(d))
depth_normalizer = math.log2(sum(2.0 ** (-v) for v in depth_values))
depth_penalty = np.asarray(depth_values, dtype=np.float64) + depth_normalizer
assert sum(2.0 ** (-v) for v in depth_penalty) <= 1.0 + 1e-12


print('PRE-SCAN SANITY CHECK')

sanity_intervals = [
    (15, 40),
    (100, 175),
    (2500, 2600),
    (2858, 3293),
    (3500, 4000)
]

sanity_max_error = 0.0
validation_depths = [0, 1, 2, 4, 6, 8, 12, 15]

for a, b in sanity_intervals:
    indexed = indexed_ctw_all_depths_hrv(a, b)
    err = max(abs(float(indexed[d]) - direct_ctw(a, b, d)) for d in validation_depths)
    sanity_max_error = max(sanity_max_error, err)
    print(f'interval [{a + 1},{b}] length={b - a} indexed/direct error={err:.3e}')

print('maximum independent error:', sanity_max_error)
assert sanity_max_error < 1e-9
print('INDEPENDENT EXACTNESS CHECK: PASS')

lengths = []
m = 1

while m <= Lmax:
    lengths.append(m)
    new_m = int(math.ceil((1.0 + epsilon) * m))

    if new_m <= m:
        new_m = m + 1

    m = new_m

lengths = sorted(set(lengths))

print('MULTISCALE FAMILY')
print('number of scales:', len(lengths))
print('smallest scale:', lengths[0])
print('largest scale:', lengths[-1])
print('scales:', lengths)

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

candidate_reduction = exhaustive_candidates / len(candidates)
total_weight = sum(item[4] * item[5] for item in scale_info)

print('exhaustive candidates :', exhaustive_candidates)
print('multiscale candidates :', len(candidates))
print('candidate reduction   :', f'{candidate_reduction:.3f}x')
print('sum of weights         :', f'{total_weight:.12f}')
assert total_weight <= 1.0 + 1e-12

print('RUNNING REDUCED MULTISCALE SCAN')

scan_start = time.perf_counter()
best_score = math.inf
best_interval = None
best_depth = None
best_LU = None
best_L0 = None
threshold = math.log2(1.0 / alpha)
reduced_endpoint_score = np.full(N, np.nan, dtype=np.float64)

for candidate_number, (a, b, m, interval_penalty) in enumerate(candidates, start=1):
    root_logprob = indexed_ctw_all_depths_hrv(a, b)
    depth_costs = -root_logprob + depth_penalty

    dstar = int(np.argmin(depth_costs))
    LU = float(depth_costs[dstar])
    L0 = reduced_typical_codelength(a, b)
    score = LU - L0 + interval_penalty
    endpoint = b - 1

    if np.isnan(reduced_endpoint_score[endpoint]):
        reduced_endpoint_score[endpoint] = score
    elif score < reduced_endpoint_score[endpoint]:
        reduced_endpoint_score[endpoint] = score

    if score < best_score:
        best_score = score
        best_interval = (a, b)
        best_depth = dstar
        best_LU = LU
        best_L0 = L0

    if candidate_number % 5000 == 0 or candidate_number == len(candidates):
        elapsed = time.perf_counter() - scan_start
        print(
            f'{candidate_number:6d}/{len(candidates):6d} | '
            f'elapsed {elapsed:8.2f} s | best {best_score: .6f}'
        )

scan_runtime = time.perf_counter() - scan_start
total_reduced_runtime = index_runtime + scan_runtime
global_detection = bool(-best_score >= threshold)

a, b = best_interval
detected_start = a + 1
detected_end = b
detected_length = b - a

overlap = max(0, min(detected_end, true_end) - max(detected_start, true_start) + 1)

true_length = true_end - true_start + 1
detected_precision = overlap / detected_length
true_coverage = overlap / true_length
union = detected_length + true_length - overlap


detected_inside = (detected_start >= true_start and detected_end <= true_end)

scan_speedup = baseline_runtime / scan_runtime if baseline_available else float("nan")
total_speedup = baseline_runtime / total_reduced_runtime if baseline_available else float("nan")

print('REDUCED METHOD RESULT')
print('true HRV interval: [2501,4000]')
print('best candidate interval:', f'[{detected_start},{detected_end}]')
print('best candidate length:', detected_length)
print('selected depth:', best_depth)
print('L_U:', best_LU)
print('L_0:', best_L0)
print('L_U - L_0 + interval penalty:', best_score)
print('global alpha threshold:', threshold)
print('global detection:', global_detection)
print('declared interval:', (detected_start, detected_end) if global_detection else None)
print('overlap:', overlap)
print('best candidate inside HRV:', detected_inside)
print('detected precision:', detected_precision)
print('true interval covered :', true_coverage)
print('index runtime:', f'{index_runtime:.6f} s')
print('scan runtime:', f'{scan_runtime:.6f} s')
print('total reduced runtime :', f'{total_reduced_runtime:.6f} s')
if baseline_available:
    print('baseline runtime:', f'{baseline_runtime:.6f} s')
    print('scan speedup:', f'{scan_speedup:.3f}x')
    print('total speedup:', f'{total_speedup:.3f}x')
else:
    print('baseline comparison: skipped (no matching original DataTest)')

rho = 1.0 / (1.0 + epsilon) ** 2
guaranteed_length_500 = 500 * rho - 1.0
baseline_best_length = 435
guaranteed_from_baseline = baseline_best_length * rho - 1.0

print('MULTISCALE GEOMETRIC GUARANTEE')
print('rho:', rho)
print('full HRV length: 1500')
print('Lmax: 500')
print('guarantee applies to : subintervals with length <= 500')
print('for length 500:', guaranteed_length_500)
print('floor:', int(math.floor(guaranteed_length_500)))
print('baseline best length:', baseline_best_length)
print('guaranteed from 435:', guaranteed_from_baseline)
print('floor:', int(math.floor(guaranteed_from_baseline)))

positions = np.arange(1, N + 1)
valid = ~np.isnan(reduced_endpoint_score)

plt.figure(figsize=(15, 5))
plt.plot(
    positions[valid],
    reduced_endpoint_score[valid],
    label='Penalized multiscale score (-A)'
)
plt.axhline(0, linestyle='--', linewidth=1)
plt.axvspan(
    true_start,
    true_end,
    alpha=0.12,
    label='Inserted normal HRV'
)
plt.axvspan(
    detected_start,
    detected_end,
    alpha=0.18,
    label='Best reduced candidate'
)
plt.xlabel('Endpoint')
plt.ylabel('L_U - L_0 + interval penalty')
plt.title('Experiment 4 — Normal HRV — reduced multiscale method')
plt.legend()
plt.show()

datab_reduced = x.astype(float).copy()
datab_reduced[x == 0] = -1.0
cumdata_reduced = np.cumsum(datab_reduced)

plt.figure(figsize=(15, 6))
plt.plot(
    positions,
    cumdata_reduced,
    linewidth=0.5,
    label='Random Walk of Mixed Data'
)
plt.plot(
    positions[valid],
    3.0 * reduced_endpoint_score[valid],
    label='3 × penalized ΔL'
)
plt.axvspan(
    true_start,
    true_end,
    alpha=0.1,
    label='Inserted normal HRV'
)
plt.xlabel('Samples')
plt.ylabel('Cumulative Bits / atypicality')
plt.title('Mixture of Coin Tosses and Normal HRV — reduced method')
plt.legend()
plt.show()

print('FINAL COMPARISON — EXPERIMENT 4')

print('HISTORICAL ORIGINAL EXHAUSTIVE (reference only):')
print('true interval: [2501,4000]')
print('best interval: [2859,3293]')
print('length: 435')
print('depth: 4')
print('score: -114.350081446013')
print('candidates:', exhaustive_candidates)
print('runtime: 500.600820 s')

print('OUR REDUCED METHOD:')
print(f'best candidate interval: [{detected_start},{detected_end}]')
print(f'length: {detected_length}')
print(f'depth: {best_depth}')
print(f'penalized score (negative A): {best_score:.12f}')
print(f'candidates: {len(candidates)}')
print(f'index runtime: {index_runtime:.6f} s')
print(f'scan runtime: {scan_runtime:.6f} s')
print(f'total runtime: {total_reduced_runtime:.6f} s')
print(f'candidate reduction : {candidate_reduction:.3f}x')
if baseline_available:
    print(f'total speedup: {total_speedup:.3f}x')
else:
    print('baseline comparison: skipped')
print(f'global detection: {global_detection}')
print(f'inside HRV: {detected_inside}')
print(f'precision: {detected_precision:.6f}')

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.savez_compressed(
    OUTPUT_FILE,
    DataTest=x,
    C0=typ_prefix,
    best_interval=np.asarray(best_interval, dtype=np.int32),
    declared_interval=np.asarray(best_interval if global_detection else [], dtype=np.int32),
    best_score=best_score,
    best_depth=best_depth,
    depth_penalty=depth_penalty,
    depth_normalizer=depth_normalizer,
    alpha=alpha,
    global_detection=global_detection,
    threshold=threshold,
    candidate_count=len(candidates),
    total_weight=total_weight,
    max_validation_error=sanity_max_error,
    scan_runtime=scan_runtime,
    index_runtime=index_runtime,
    total_runtime=total_reduced_runtime,
    HRV_TIE_SEED=HRV_TIE_SEED,
    baseline_available=baseline_available,
    baseline_runtime=baseline_runtime,
    endpoint_scores=reduced_endpoint_score
)

print('Saved:', OUTPUT_FILE)