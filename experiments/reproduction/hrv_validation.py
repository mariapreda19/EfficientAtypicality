from project_paths import DATA_ROOT
from pathlib import Path
import numpy as np
import pandas as pd
import math
import random
import time

BASE = Path(DATA_ROOT)
Dmax = 15
Lmax = 500
epsilon = 0.1


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
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for line in f:
            if line.strip():
                lines.append(line.rstrip('\n\r'))

    hrv = []

    for line in lines:
        parts = line.split()
        numeric_candidates = []

        for token in parts:
            try:
                value = float(token)
                numeric_candidates.append(value)
            except ValueError:
                pass

        if len(numeric_candidates) > 0:
            hrv.append(numeric_candidates[0])

    hrv = np.asarray(hrv, dtype=np.float64)
    hrvb = np.zeros(len(hrv) - 1, dtype=np.uint8)

    for i in range(1, len(hrv)):
        if hrv[i] > hrv[i - 1]:
            hrvb[i - 1] = 1
        elif hrv[i] < hrv[i - 1]:
            hrvb[i - 1] = 0
        else:
            if random.random() > 0.5:
                hrvb[i - 1] = 1
            else:
                hrvb[i - 1] = 0

    return hrvb


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

hrvb = read_hrv(hrv_hits[0])
hrv_segment = hrvb[30000:31500].copy()

if len(hrv_segment) != 1500:
    raise ValueError(f'Expected 1500 HRV bits, got {len(hrv_segment)}')

data = original_data[15000:20000].copy()
data[2500:4000] = hrv_segment
x = np.asarray(data, dtype=np.uint8).copy()

assert len(x) == 5000
assert Dmax == 15
assert Lmax == 500
assert abs(p - 0.505425) < 1e-12


def reduced_logs(n):
    m = 0.0
    k = math.log2(float(n))
    while k > 0:
        m += k
        k = math.log2(k)
    return m


typ_prefix = np.zeros(len(x) + 1, dtype=np.float64)
cost0 = math.log2(1.0 / (1.0 - p))
cost1 = math.log2(1.0 / p)

for t in range(len(x)):
    if x[t] == 1:
        typ_prefix[t + 1] = typ_prefix[t] + cost1
    else:
        typ_prefix[t + 1] = typ_prefix[t] + cost0


def reduced_typical_codelength(a, b):
    return float(typ_prefix[b] - typ_prefix[a])


ctx = []
for _ in range(Dmax + 1):
    ctx.append(np.zeros(len(x), dtype=np.int32))

for d in range(1, Dmax + 1):
    mask = (1 << d) - 1
    state = 0

    for t in range(len(x)):
        ctx[d][t] = state
        state = (state << 1 | int(x[t])) & mask


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
    node_counts = []
    for _ in range(Dmax + 1):
        node_counts.append({})

    for t in range(a, b):
        bit = int(x[t])

        for d in range(Dmax + 1):
            context = int(ctx[d][t])

            if context not in node_counts[d]:
                node_counts[d][context] = [0, 0]

            node_counts[d][context][bit] += 1

    pe = []
    for _ in range(Dmax + 1):
        pe.append({})

    for d in range(Dmax + 1):
        for context, counts in node_counts[d].items():
            pe[d][context] = reduced_kt_logprob(counts[0], counts[1])

    root_logprob = np.zeros(Dmax + 1, dtype=np.float64)
    root_logprob[0] = pe[0].get(0, 0.0)

    for max_depth in range(1, Dmax + 1):
        pw = []
        for _ in range(max_depth + 1):
            pw.append({})

        for context in node_counts[max_depth]:
            pw[max_depth][context] = pe[max_depth][context]

        for d in range(max_depth - 1, -1, -1):
            for context in node_counts[d]:
                child0 = 2 * context
                child1 = child0 + 1
                left = pw[d + 1].get(child0, 0.0)
                right = pw[d + 1].get(child1, 0.0)
                split = left + right
                pw[d][context] = reduced_log2add(pe[d][context], split) - 1.0

        root_logprob[max_depth] = pw[0].get(0, 0.0)

    return root_logprob


lengths = []
m = 1

while m <= Lmax:
    lengths.append(m)
    new_m = int(math.ceil((1.0 + epsilon) * m))

    if new_m <= m:
        new_m = m + 1

    m = new_m

lengths = sorted(set(lengths))

candidates = []

for m in lengths:
    h = max(1, int(math.floor(epsilon * m)))
    a = 0

    while a + m <= len(x):
        candidates.append((a, a + m, m))
        a += h

candidates = list(dict.fromkeys(candidates))

best_score = math.inf
best_interval = None
best_depth = None
best_LU = None
best_L0 = None

print('RECOMPUTING REDUCED WINNER')

for candidate_number, (a, b, m) in enumerate(candidates, start=1):
    root_logprob = indexed_ctw_all_depths_hrv(a, b)
    depth_costs = np.empty(Dmax, dtype=np.float64)

    for d in range(1, Dmax + 1):
        depth_costs[d - 1] = -root_logprob[d] + reduced_logs(d)

    dstar = int(np.argmin(depth_costs)) + 1
    LU = float(depth_costs[dstar - 1]) + reduced_logs(m)
    L0 = reduced_typical_codelength(a, b)
    score = LU - L0

    if score < best_score:
        best_score = score
        best_interval = (a, b)
        best_depth = dstar
        best_LU = LU
        best_L0 = L0

    if candidate_number % 10000 == 0 or candidate_number == len(candidates):
        print(f'{candidate_number}/{len(candidates)} | best={best_score:.12f}')

print('recomputed winner:', f'[{best_interval[0] + 1},{best_interval[1]}]')
print('recomputed depth:', best_depth)
print('recomputed score:', f'{best_score:.12f}')


def direct_kt_logprob(n0, n1):
    n0 = int(n0)
    n1 = int(n1)

    return (
        math.lgamma(n0 + 0.5) + math.lgamma(n1 + 0.5)
        - math.lgamma(n0 + n1 + 1.0) - 2.0 * math.lgamma(0.5)
    ) / math.log(2.0)


def direct_log2add(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a

    m = max(a, b)
    return m + math.log2(2.0 ** (a - m) + 2.0 ** (b - m))


def direct_context_from_data(x, t, d):
    if d == 0:
        return 0

    context = 0
    start = t - d

    for k in range(start, t):
        context <<= 1
        if k >= 0:
            context |= int(x[k])

    return context


def direct_counts_for_interval(x, a, b, Dmax):
    counts = []

    for _ in range(Dmax + 1):
        counts.append({})

    for t in range(a, b):
        bit = int(x[t])

        for d in range(Dmax + 1):
            context = direct_context_from_data(x, t, d)

            if context not in counts[d]:
                counts[d][context] = [0, 0]

            counts[d][context][bit] += 1

    return counts


def direct_ctw_all_depths(x, a, b, Dmax):
    node_counts = direct_counts_for_interval(x, a, b, Dmax)
    pe = []

    for _ in range(Dmax + 1):
        pe.append({})

    for d in range(Dmax + 1):
        for context, counts in node_counts[d].items():
            pe[d][context] = direct_kt_logprob(counts[0], counts[1])

    result = np.zeros(Dmax + 1, dtype=np.float64)
    result[0] = pe[0].get(0, 0.0)

    for max_depth in range(1, Dmax + 1):
        pw = []

        for _ in range(max_depth + 1):
            pw.append({})

        for context in node_counts[max_depth]:
            pw[max_depth][context] = pe[max_depth][context]

        for d in range(max_depth - 1, -1, -1):
            for context in node_counts[d]:
                child0 = 2 * context
                child1 = child0 + 1
                left = pw[d + 1].get(child0, 0.0)
                right = pw[d + 1].get(child1, 0.0)
                split = left + right
                pw[d][context] = direct_log2add(pe[d][context], split) - 1.0

        result[max_depth] = pw[0].get(0, 0.0)

    return result


validation_intervals = [
    (0, 25),
    (100, 175),
    (2500, 2600),
    (2858, 3293),
    (2744, 3238),
    (3500, 4000)
]

print('TRUE VALIDATION — INDEPENDENT DIRECT CTW VS INDEXED CTW')

global_max_error = 0.0
validation_start = time.perf_counter()

for a, b in validation_intervals:
    direct_start = time.perf_counter()
    direct_logP = direct_ctw_all_depths(x, a, b, Dmax)
    direct_runtime = time.perf_counter() - direct_start

    indexed_start = time.perf_counter()
    indexed_logP = indexed_ctw_all_depths_hrv(a, b)
    indexed_runtime = time.perf_counter() - indexed_start

    interval_max_error = 0.0

    for d in range(Dmax + 1):
        error = abs(direct_logP[d] - indexed_logP[d])

        if error > interval_max_error:
            interval_max_error = error

        if error > global_max_error:
            global_max_error = error

    if interval_max_error < 1e-10:
        status = 'PASS'
    else:
        status = 'FAIL'

    print(
        f'[{a + 1},{b}] length={b - a} | error={interval_max_error:.3e} | '
        f'direct={direct_runtime:.6f}s | indexed={indexed_runtime:.6f}s | {status}'
    )

validation_runtime = time.perf_counter() - validation_start

if global_max_error < 1e-10:
    all_passed = True
else:
    all_passed = False

print('maximum abs logP error:', f'{global_max_error:.15e}')
print('validation runtime    :', f'{validation_runtime:.6f} s')
print('all intervals passed  :', all_passed)

winner_a = 2744
winner_b = 3238
winner_length = winner_b - winner_a

winner_direct_logP = direct_ctw_all_depths(x, winner_a, winner_b, Dmax)
direct_depth_costs = np.zeros(Dmax, dtype=np.float64)

for d in range(1, Dmax + 1):
    direct_depth_costs[d - 1] = -winner_direct_logP[d] + reduced_logs(d)

direct_best_depth = int(np.argmin(direct_depth_costs)) + 1
direct_LU = float(direct_depth_costs[direct_best_depth - 1]) + reduced_logs(winner_length)

direct_L0 = 0.0

for t in range(winner_a, winner_b):
    if int(x[t]) == 1:
        direct_L0 += math.log2(1.0 / p)
    else:
        direct_L0 += math.log2(1.0 / (1.0 - p))

direct_score = direct_LU - direct_L0

depth_ok = direct_best_depth == best_depth == 8
LU_error = abs(direct_LU - best_LU)
L0_error = abs(direct_L0 - best_L0)
score_error = abs(direct_score - best_score)
expected_score_error = abs(direct_score - (-198.247541355728))

print('INDEPENDENT RECOMPUTATION OF OUR WINNING INTERVAL')
print('interval:', f'[{winner_a + 1},{winner_b}]')
print('length:', winner_length)
print('best depth DIRECT:', direct_best_depth)
print('L_U DIRECT:', direct_LU)
print('L_0 DIRECT:', direct_L0)
print('score DIRECT:', direct_score)
print('depth agreement:', depth_ok)
print('L_U error:', f'{LU_error:.15e}')
print('L_0 error:', f'{L0_error:.15e}')
print('score error:', f'{score_error:.15e}')
print('expected score error:', f'{expected_score_error:.15e}')

assert all_passed, 'Direct and indexed CTW do not agree.'
assert best_interval == (2744, 3238), f'Expected winner [2745,3238], got [{best_interval[0] + 1},{best_interval[1]}].'
assert direct_best_depth == 8, f'Expected depth 8, got {direct_best_depth}.'
assert best_depth == 8, f'Scan depth should be 8, got {best_depth}.'
assert LU_error < 1e-9, f'L_U mismatch: {LU_error}'
assert L0_error < 1e-9, f'L_0 mismatch: {L0_error}'
assert score_error < 1e-9, f'Score mismatch: {score_error}'
assert expected_score_error < 1e-9, 'Independent score does not reproduce -198.247541355728.'

print('VALIDATION PASSED COMPLETELY')