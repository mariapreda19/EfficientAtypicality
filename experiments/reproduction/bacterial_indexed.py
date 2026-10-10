from project_paths import DATA_ROOT, RESULTS_ROOT
import os
import math
import time
from bisect import bisect_left, bisect_right
import numpy as np
import matplotlib.pyplot as plt

BASELINE_FILE = str(RESULTS_ROOT / 'experiment5_literal_matlab_port.npz')
OUTPUT_FILE = str(RESULTS_ROOT / 'experiment5_indexed_multiscale_penalized.npz')
epsilon = 0.1
alpha = 0.05
TRUE_1 = (1617, 2000)
TRUE_2 = (2001, 4061)


def build_bacterial_without_baseline():
    def DNAReadFunc(filename):
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        mapping = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
        bits = []
        bases = []
        for ch in text.upper():
            pair = mapping.get(ch)
            if pair is not None:
                bits.extend(pair)
                bases.append((pair[0] << 1) | pair[1])
        return np.asarray(bits, dtype=np.uint8), np.asarray(bases, dtype=np.uint8)

    def logsp(a, b):
        if a < b:
            a, b = b, a
        return a + math.log1p(2.0 ** (b - a)) / math.log(2.0)

    class ContextTree:
        def __init__(self, D):
            self.D = int(D)
            self.Context = 0
            self.ContextLen = 0

        def ContextAddress(self, length):
            shift = int(length) - self.ContextLen
            addr = self.Context << shift if shift >= 0 else self.Context >> (-shift)
            addr |= 1 << length
            return int(addr)

        def UpdateContext(self, bit):
            if self.ContextLen < self.D - 1:
                self.ContextLen += 1
            else:
                self.Context >>= 1
            pos = self.ContextLen - 1
            if bit:
                self.Context |= 1 << pos
            else:
                self.Context &= ~(1 << pos)

        def SetContext(self, bits):
            bits = [int(b) for b in bits]
            if len(bits) > self.D - 1:
                bits = bits[-(self.D - 1):]
            self.ContextLen = len(bits)
            self.Context = 0
            for i, bit in enumerate(reversed(bits)):
                if bit:
                    self.Context |= 1 << i
            return self

    class CTWbn(ContextTree):
        def __init__(self, D, N=1000):
            super().__init__(D)
            self.N = int(N)
            self.Pe = {}
            self.Pw = [{} for _ in range(self.D)]
            self.DynamicCount = {}
            self.Tail = {}
            self.Rep = 0
            self.Frozen = False
            self.Changed = True

        def _get_pe(self, addr):
            return self.Pe.get(addr, 0.0)

        def _set_pe(self, addr, value):
            if value == 0.0:
                self.Pe.pop(addr, None)
            else:
                self.Pe[addr] = float(value)

        def _get_pw(self, D, addr):
            return self.Pw[D - 1].get(addr, 0.0)

        def _set_pw(self, D, addr, value):
            if value == 0.0:
                self.Pw[D - 1].pop(addr, None)
            else:
                self.Pw[D - 1][addr] = float(value)

        def _counts(self, addr):
            return self.DynamicCount.get(addr, [0, 0])

        def Update(self, bit):
            bit = int(bit)
            for rep in range(self.Rep + 1):
                if self.ContextLen < self.D:
                    self.Tail[self.ContextAddress(self.ContextLen)] = 1.0
                for i in range(self.ContextLen, -1, -1):
                    addr = self.ContextAddress(i)
                    counts = self._counts(addr)
                    old_pe = self._get_pe(addr)
                    prob = (counts[bit] + 0.5) / (counts[0] + counts[1] + 1.0)
                    new_pe = old_pe + math.log2(prob)
                    self._set_pe(addr, new_pe)
                    if not self.Frozen:
                        new_counts = [counts[0], counts[1]]
                        new_counts[bit] += 1
                        self.DynamicCount[addr] = new_counts
                    child = addr * 2
                    self._set_pw(i + 1, addr, new_pe)
                    for DD in range(i + 2, self.D + 1):
                        children_value = self._get_pw(DD, child) + self._get_pw(DD, child + 1) - self.Tail.get(addr,
                                                                                                               0.0)
                        self._set_pw(DD, addr, logsp(new_pe, children_value) - 1.0)
                self.UpdateContext(bit)
                self.Changed = True
            return self

        def CalcCodeLength(self, D=None):
            if D is None:
                D = self.D
            if D == 0:
                counts = self._counts(1)
                return float(counts[0] + counts[1])
            return float(-self._get_pw(int(D), 1))

        def ResetContext(self):
            self.Context = 0
            self.ContextLen = 0
            return self

    class CTWbFrozen(ContextTree):
        def __init__(self, CTWi, D=None, PreCalc=1):
            if D is None:
                cl = [CTWi.CalcCodeLength(d) for d in range(CTWi.D + 1)]
                D = int(np.argmin(np.asarray(cl)))
            D = max(int(D), 2)
            super().__init__(D)
            self.Context = int(CTWi.Context)
            self.ContextLen = int(CTWi.ContextLen)
            self.Size = (1 << self.D) - 1
            self.PreCalc = int(PreCalc)
            self.Piid = {}
            self.P1iid = {}
            self.Pet = 0.0
            if self.PreCalc:
                self.InitProb(CTWi)

        def CalcProb(self, addr, CTWi):
            P1iid = 0.0
            Piid = 0.0
            maxin = (self.Size + 1) // 2 - 1
            pw_here = CTWi._get_pw(self.D, addr)
            if pw_here != 0.0:
                counts = CTWi._counts(addr)
                P1iid = (counts[1] + 0.5) / (counts[0] + counts[1] + 1.0)
                if addr <= maxin:
                    Piid = CTWi._get_pe(addr) - logsp(CTWi._get_pe(addr),
                                                      CTWi._get_pw(self.D, addr * 2) + CTWi._get_pw(self.D,
                                                                                                    addr * 2 + 1))
            return Piid, P1iid

        def InitProb(self, CTWi):
            for addr in range(1, self.Size + 1):
                if CTWi._get_pw(self.D, addr) != 0.0:
                    Piid, P1iid = self.CalcProb(addr, CTWi)
                    if Piid != 0.0:
                        self.Piid[addr] = float(Piid)
                    if P1iid != 0.0:
                        self.P1iid[addr] = float(P1iid)

        def Update(self, bit):
            bit = int(bit)
            P1Niid = 0.5
            for i in range(self.ContextLen, -1, -1):
                shift = i - self.ContextLen
                addr = self.Context << shift if shift >= 0 else self.Context >> (-shift)
                addr |= 1 << i
                P1iid = self.P1iid.get(addr, 0.0)
                if P1iid != 0.0:
                    Piid = 2.0 ** self.Piid.get(addr, 0.0)
                    P1Niid = P1iid * Piid + (1.0 - Piid) * P1Niid
            P1Niid = min(max(P1Niid, 1e-300), 1.0 - 1e-16)
            if bit:
                self.Pet += math.log2(P1Niid)
            else:
                self.Pet += math.log2(1.0 - P1Niid)
            self.UpdateContext(bit)
            return self

        def CalcCodeLength(self):
            return float(-self.Pet)

    d1 = os.path.join(str(DATA_ROOT), "chr6.txt")
    d2 = os.path.join(str(DATA_ROOT), "chr6c.txt")
    if not os.path.isfile(d1) or not os.path.isfile(d2):
        raise FileNotFoundError("Missing chr6.txt or chr6c.txt under " + str(DATA_ROOT))
    DataBin1, DataNum1 = DNAReadFunc(d1)
    DataBin2, DataNum2 = DNAReadFunc(d2)
    DataTrain = DataBin2[10000:20000].copy()
    DataTest = np.concatenate((DataBin2[:2000], DataBin2[4999:10000]))
    Dmax, Lmax = 16, 800
    assert len(DataTrain) == 10000 and len(DataTest) == 7001

    Typical = CTWbn(Dmax, len(DataTrain))
    Typical.SetContext(DataTrain[:Dmax])
    for bit in DataTrain[Dmax:]:
        Typical.Update(int(bit))
    Typical.ResetContext()
    Frozen = CTWbFrozen(Typical)
    TCL = np.zeros(len(DataTest), dtype=np.float64)
    for t, bit in enumerate(DataTest):
        Frozen.Update(int(bit))
        TCL[t] = Frozen.CalcCodeLength()

    print("Typical model trained and frozen; depth:", Frozen.D)
    return DataTest, TCL, Dmax, Lmax


baseline_available = os.path.exists(BASELINE_FILE)
if baseline_available:
    z = np.load(BASELINE_FILE, allow_pickle=False)
    x = z['DataTest'].astype(np.uint8)
    TCL = z['TCL'].astype(np.float64)
    Dmax = int(z['maxD'])
    Lmax = int(z['maxL'])
    baseline_runtime = float(z['total_runtime'])
    typical_setup_time = 0.0
else:
    print('Baseline file not found; preparing Bacterial data and frozen typical model')
    setup0 = time.perf_counter()
    x, TCL, Dmax, Lmax = build_bacterial_without_baseline()
    typical_setup_time = time.perf_counter() - setup0
    baseline_runtime = float('nan')
N = len(x)
assert N == 7001 and Dmax == 16 and Lmax == 800
assert len(TCL) == N

print('EXPERIMENT 5 — INDEXED MULTISCALE ATYPICALITY')

print('N:', N)
print('Dmax:', Dmax)
print('Lmax:', Lmax)
print('epsilon:', epsilon)
print('fragment 1:', TRUE_1)
print('fragment 2:', TRUE_2)
if baseline_available:
    print('Baseline total runtime:', baseline_runtime, 's')
else:
    print('Baseline comparison: unavailable (no original run)')
    print('Frozen typical setup:', f'{typical_setup_time:.6f} s')

C0 = np.empty(N + 1, dtype=np.float64)
C0[0] = 0.0
C0[1:] = TCL


def typical_length(s, e):
    return float(C0[e] - C0[s - 1])


def logs(n):
    if n <= 1:
        return 0.0
    ans = 0.0
    y = math.log2(n)
    while y > 0:
        ans += y
        y = math.log2(y)
    return ans


depth_values = []
for d in range(Dmax + 1):
    depth_values.append(logs(d))
depth_normalizer = math.log2(sum(2.0 ** (-v) for v in depth_values))
depth_penalty = np.asarray(depth_values, dtype=np.float64) + depth_normalizer
assert sum(2.0 ** (-v) for v in depth_penalty) <= 1.0 + 1e-12

print('BUILDING CONTEXT IDS')

t0 = time.perf_counter()
context_ids = []
context_ids.append(np.zeros(N, dtype=np.int32))
for d in range(1, Dmax + 1):
    ids = np.full(N, -1, dtype=np.int32)
    if N > d:
        c = 0
        for k in range(d):
            c = c << 1 | int(x[k])
        ids[d] = c
        mask = (1 << d) - 1
        for t in range(d + 1, N):
            c = c << 1 & mask | int(x[t - 1])
            ids[t] = c
    context_ids.append(ids)
context_time = time.perf_counter() - t0
print('Context preprocessing :', f'{context_time:.6f} s')

print('BUILDING OCCURRENCE INDEX')

t0 = time.perf_counter()
occ = []
for d in range(Dmax + 1):
    level = {}
    ids = context_ids[d]
    for py_t in range(N):
        c = int(ids[py_t])
        if c < 0:
            continue
        b = int(x[py_t])
        pair = level.get(c)
        if pair is None:
            pair = [[], []]
            level[c] = pair
        pair[b].append(py_t + 1)
    occ.append(level)
occ_time = time.perf_counter() - t0
stored_occurrences = 0
for level in occ:
    for pair in level.values():
        stored_occurrences += len(pair[0]) + len(pair[1])
print('Occurrence preprocessing:', f'{occ_time:.6f} s')
print('Stored occurrences      :', stored_occurrences)

del context_ids


def count_sorted(arr, s, e):
    return bisect_right(arr, e) - bisect_left(arr, s)


def node_counts(depth, context, s, e):
    pair = occ[depth].get(context)
    if pair is None:
        return (0, 0)
    return (count_sorted(pair[0], s, e), count_sorted(pair[1], s, e))


LOG2 = math.log(2.0)
LOG_PI = math.log(math.pi)
kt_cache = {}


def KT(a, b):
    key = (int(a), int(b))
    if key in kt_cache:
        return kt_cache[key]
    value = (math.lgamma(a + 0.5) + math.lgamma(b + 0.5) - LOG_PI - math.lgamma(a + b + 1.0)) / LOG2
    kt_cache[key] = value
    return value


def ctw_mix(a, b):
    m = max(a, b)
    return m + math.log2(2.0 ** (a - m) + 2.0 ** (b - m)) - 1.0


def indexed_ctw_all_depths(s, e):
    Pe = []
    for d in range(Dmax + 1):
        Pe.append({})
    active = [0]
    for d in range(Dmax + 1):
        next_active = []
        for c in active:
            n0, n1 = node_counts(d, c, s, e)
            if n0 == 0 and n1 == 0:
                continue
            Pe[d][c] = KT(n0, n1)
            if d < Dmax:
                next_active.append(c)
                next_active.append((1 << d) + c)
        active = next_active
        if not active and d < Dmax:
            break
    logQ = np.empty(Dmax + 1, dtype=np.float64)
    logQ[0] = Pe[0].get(0, 0.0)
    for target_depth in range(1, Dmax + 1):
        Pw = dict(Pe[target_depth])
        for j in range(target_depth - 1, -1, -1):
            parent = {}
            for c, pe in Pe[j].items():
                child0 = c
                child1 = (1 << j) + c
                child_product = Pw.get(child0, 0.0) + Pw.get(child1, 0.0)
                parent[c] = ctw_mix(pe, child_product)
            Pw = parent
        logQ[target_depth] = Pw.get(0, 0.0)
    return logQ


def alternative_length(s, e):
    logQ = indexed_ctw_all_depths(s, e)
    LU_all = -logQ + depth_penalty
    D = int(np.argmin(LU_all))
    return (float(LU_all[D]), D, float(logQ[D]))


def direct_counts(s, e, d):
    counts = {}
    for t1 in range(s, e + 1):
        t = t1 - 1
        if t < d:
            continue
        c = 0
        for q in range(t - d, t):
            c = c << 1 | int(x[q])
        b = int(x[t])
        pair = counts.get(c)
        if pair is None:
            pair = [0, 0]
            counts[c] = pair
        pair[b] += 1
    return counts


def direct_ctw(s, e, target_depth):
    Pe = []
    for d in range(target_depth + 1):
        counts = direct_counts(s, e, d)
        level = {}
        for c, pair in counts.items():
            level[c] = KT(pair[0], pair[1])
        Pe.append(level)
    if target_depth == 0:
        return Pe[0].get(0, 0.0)
    Pw = dict(Pe[target_depth])
    for j in range(target_depth - 1, -1, -1):
        parent = {}
        for c, pe in Pe[j].items():
            child0 = c
            child1 = (1 << j) + c
            parent[c] = ctw_mix(pe, Pw.get(child0, 0.0) + Pw.get(child1, 0.0))
        Pw = parent
    return Pw.get(0, 0.0)


print('INDEPENDENT EXACTNESS TEST')

validation_intervals = [(101, 120), (501, 600), (1501, 1700), (1617, 1800),
                        (1801, 2000), (2001, 2200), (3001, 3300), (3861, 4061),
                        (4062, 4261), (6001, 6300)]
validation_depths = [0, 1, 2, 4, 6, 8, 12, 16]
max_error = 0.0
worst = None
for s, e in validation_intervals:
    indexed = indexed_ctw_all_depths(s, e)
    local_max = 0.0
    for d in validation_depths:
        direct = direct_ctw(s, e, d)
        error = abs(direct - float(indexed[d]))
        local_max = max(local_max, error)
        if error > max_error:
            max_error = error
            worst = (s, e, d, direct, float(indexed[d]))
    print(f'[{s:4d},{e:4d}] max error = {local_max:.3e}')

print('Max error:', f'{max_error:.3e}')
print('Worst case:', worst)
if max_error > 1e-10:
    raise RuntimeError('Exactness test failed')

print('EXACTNESS CHECK: PASS')

lengths = []
m = 1
while m <= Lmax:
    lengths.append(m)
    new_m = int(math.ceil((1.0 + epsilon) * m))
    if new_m <= m:
        new_m = m + 1
    m = new_m
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

candidate_count = 0
for item in scale_info:
    candidate_count += item[4]

exhaustive_count = 0
for m in range(1, Lmax + 1):
    exhaustive_count += N - m + 1
total_weight = 0.0
for item in scale_info:
    n_k = item[4]
    weight = item[5]
    total_weight += n_k * weight

print('MULTISCALE FAMILY')

print('Number of scales      :', K)
print('Exhaustive candidates :', exhaustive_count)
print('Multiscale candidates :', candidate_count)
print('Candidate reduction   :', f'{exhaustive_count / candidate_count:.3f}x')
print('Sum of weights        :', f'{total_weight:.12f}')
assert total_weight <= 1.0 + 1e-12

print('SANITY CHECK — RAW SCORE')

sanity = [(1501, 1600), (1617, 1800), (1801, 2000), (2001, 2200), (2501, 2800), (3301, 3600), (3861, 4061),
          (4062, 4261)]
for s, e in sanity:
    L0 = typical_length(s, e)
    LU, D, _ = alternative_length(s, e)
    raw = L0 - LU
    print(f'[{s:4d},{e:4d}] len={e - s + 1:4d} D={D:2d} L0={L0:10.4f} LU={LU:10.4f} RAW={raw:10.4f}')

print('RUNNING INDEXED MULTISCALE SCAN')

scan0 = time.perf_counter()
results = []
processed = 0

for scale_no, (m, h, j_first, j_last, n_k, weight, interval_penalty) in enumerate(scale_info, start=1):
    scale0 = time.perf_counter()
    for j in range(j_first, j_last + 1):
        s = 1 + j * h
        e = s + m - 1
        L0 = C0[e] - C0[s - 1]
        LU, D, logQ = alternative_length(s, e)
        raw_score = L0 - LU
        penalized_score = raw_score - interval_penalty
        results.append((float(penalized_score), float(raw_score), int(s), int(e), int(m), int(D), float(L0), float(LU),
                        float(interval_penalty)))
        processed += 1

    scale_time = time.perf_counter() - scale0
    elapsed = time.perf_counter() - scan0

    print(
        f'scale {scale_no:2d}/{K} '
        f'| m={m:4d} '
        f'| h={h:3d} '
        f'| n={n_k:4d} '
        f'| penalty={interval_penalty:7.3f} '
        f'| processed={processed:7d} '
        f'| scale={scale_time:7.3f}s '
        f'| total={elapsed:8.3f}s'
    )

scan_time = time.perf_counter() - scan0
assert processed == candidate_count


def penalized_score(item):
    return item[0]


results.sort(key=penalized_score, reverse=True)
threshold = math.log2(1 / alpha)


def intersection(A, B):
    left = max(A[0], B[0])
    right = min(A[1], B[1])
    return max(0, right - left + 1)


def int_len(I):
    return I[1] - I[0] + 1


def overlap_metrics(detected, truth):
    inter = intersection(detected, truth)
    ld = int_len(detected)
    lt = int_len(truth)
    union = ld + lt - inter
    precision = inter / ld
    coverage = inter / lt
    iou = inter / union
    return (inter, precision, coverage, iou)


print('GLOBAL TOP 20 — PENALIZED SCORE')

for rank, r in enumerate(results[:20], start=1):
    A, raw, s, e, m, D, L0, LU, pen = r
    ov1 = intersection((s, e), TRUE_1)
    ov2 = intersection((s, e), TRUE_2)
    print(f'{rank:2d}. [{s:4d},{e:4d}] len={m:4d} D={D:2d} RAW={raw:10.4f} pen={pen:7.3f}'
          f' A={A:10.4f} ovI1={ov1:4d} ovI2={ov2:4d}')


def raw_score(item):
    return item[1]


raw_results = sorted(results, key=raw_score, reverse=True)

print('GLOBAL TOP 10 — RAW SCORE')

for rank, r in enumerate(raw_results[:10], start=1):
    A, raw, s, e, m, D, L0, LU, pen = r
    print(f'{rank:2d}. [{s:4d},{e:4d}] len={m:4d} D={D:2d} RAW={raw:10.4f} A={A:10.4f}')


def best_overlapping(truth):
    for r in results:
        if intersection((r[2], r[3]), truth) > 0:
            return r
    return None


def best_inside(truth):
    for r in results:
        if r[2] >= truth[0]:
            if r[3] <= truth[1]:
                return r
    return None


def report(title, r, truth):
    print(title)
    if r is None:
        print('NONE')
        return
    A, raw, s, e, m, D, L0, LU, pen = r
    inter, precision, coverage, iou = overlap_metrics((s, e), truth)
    print('interval:', (s, e))
    print('length:', m)
    print('depth:', D)
    print('L0:', f'{L0:.9f}')
    print('LU:', f'{LU:.9f}')
    print('RAW:', f'{raw:.9f}')
    print('interval penalty:', f'{pen:.9f}')
    print('A penalized:', f'{A:.9f}')
    print('overlap:', inter)
    print('precision:', f'{precision:.6f}')
    print('coverage:', f'{coverage:.6f}')
    print('IoU:', f'{iou:.6f}')


print('Post-hoc fragment evaluation - penalized')

report('BEST OVERLAPPING I1', best_overlapping(TRUE_1), TRUE_1)
report('BEST INSIDE I1', best_inside(TRUE_1), TRUE_1)
report('BEST OVERLAPPING I2', best_overlapping(TRUE_2), TRUE_2)
report('BEST INSIDE I2', best_inside(TRUE_2), TRUE_2)


def overlap_fraction(A, B):
    inter = intersection(A, B)
    if inter == 0:
        return 0.0
    return inter / min(int_len(A), int_len(B))


selected = []
for r in results:
    if r[0] < threshold:
        break
    I = (r[2], r[3])
    okay = True
    for q in selected:
        J = (q[2], q[3])
        if overlap_fraction(I, J) >= 0.25:
            okay = False
            break
    if okay:
        selected.append(r)
    if len(selected) >= 30:
        break

print('GLOBAL ALPHA THRESHOLD:', threshold)
print('GLOBAL DETECTION:', bool(results and results[0][0] >= threshold))
print('TOP 30 DISTINCT SIGNIFICANT DETECTIONS — PENALIZED, NO GROUND TRUTH')

for rank, r in enumerate(selected, start=1):
    A, raw, s, e, m, D, L0, LU, pen = r
    ov1 = intersection((s, e), TRUE_1)
    ov2 = intersection((s, e), TRUE_2)
    print(f'{rank:2d}. [{s:4d},{e:4d}] len={m:4d} D={D:2d} RAW={raw:9.3f} A={A:9.3f} ovI1={ov1:4d} ovI2={ov2:4d}')
rank1 = None
rank2 = None
for rank, r in enumerate(results, start=1):
    I = (r[2], r[3])
    if rank1 is None:
        if intersection(I, TRUE_1) > 0:
            rank1 = rank
    if rank2 is None:
        if intersection(I, TRUE_2) > 0:
            rank2 = rank
    if rank1 is not None:
        if rank2 is not None:
            break

print('FIRST PENALIZED GLOBAL RANK')

print('I1 first overlap rank :', rank1)
print('I2 first overlap rank :', rank2)
our_total = context_time + occ_time + scan_time

print('RUNTIME')

if baseline_available:
    print('Literal baseline:', f'{baseline_runtime:.6f} s')
else:
    print('Literal baseline: unavailable (no original run)')
print('Context preprocessing:', f'{context_time:.6f} s')
print('Occurrence index:', f'{occ_time:.6f} s')
print('Indexed scan:', f'{scan_time:.6f} s')
print('Our total:', f'{our_total:.6f} s')
if baseline_available:
    print('Runtime speedup:', f'{baseline_runtime / our_total:.3f}x')
print('Candidate reduction:', f'{exhaustive_count / candidate_count:.3f}x')
plt.figure(figsize=(16, 7))
plt.axvspan(TRUE_1[0], TRUE_1[1], alpha=0.15, label='Bacterial fragment 1')
plt.axvspan(TRUE_2[0], TRUE_2[1], alpha=0.15, label='Bacterial fragment 2')
for rank, r in enumerate(selected[:15], start=1):
    A = r[0]
    s = r[2]
    e = r[3]
    plt.plot([s, e], [A, A], linewidth=3)
    plt.text((s + e) / 2, A, str(rank), ha='center', va='bottom', fontsize=8)
plt.axhline(0.0, linewidth=1)
plt.xlabel('Position in DataTest')
plt.ylabel('Penalized atypicality score A(I)')
plt.title('Experiment 5 — indexed multiscale atypicality')
plt.legend()
plt.show()
np.savez_compressed(OUTPUT_FILE, results=np.asarray(results, dtype=np.float64),
                    raw_results=np.asarray(raw_results, dtype=np.float64),
                    selected=np.asarray(selected, dtype=np.float64).reshape(-1, 9),
                    true1=np.asarray(TRUE_1, dtype=np.int32), true2=np.asarray(TRUE_2, dtype=np.int32), epsilon=epsilon,
                    alpha=alpha, Dmax=Dmax, depth_penalty=depth_penalty, depth_normalizer=depth_normalizer, Lmax=Lmax,
                    number_scales=K, exhaustive_count=exhaustive_count, candidate_count=candidate_count,
                    context_time=context_time, occurrence_time=occ_time, scan_time=scan_time, total_time=our_total,
                    baseline_available=baseline_available, baseline_runtime=baseline_runtime,
                    typical_setup_time=typical_setup_time, DataTest=x, TCL=TCL)

print('Saved:', OUTPUT_FILE)