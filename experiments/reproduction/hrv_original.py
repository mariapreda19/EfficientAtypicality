from project_paths import DATA_ROOT
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
import time
import re
BASE = str(DATA_ROOT)
MAT_FILE = f'{BASE}/Data40kTosses.mat'
HRV_FILE = f'{BASE}/16786.txt'

def logs(n):
    m = 0.0
    k = math.log2(float(n))
    while k > 0:
        m += k
        k = math.log2(k)
    return m

class ContextTree:
    def __init__(self):
        self.D = None
        self.Context = 0
        self.ContextLen = 0
        self.ln2i = 1.0 / math.log(2.0)

    def Children(self, addr):
        c0 = addr * 2
        c1 = c0 + 1
        return (c0, c1)

    def logsp(self, x, y):
        a = min(x, y)
        b = max(x, y)
        return b + math.log1p(2.0 ** (a - b)) * self.ln2i

    def UpdateContext(self, bit):
        if self.ContextLen < self.D - 1:
            self.ContextLen += 1
        else:
            self.Context >>= 1
        mask = 1 << self.ContextLen - 1
        if bit:
            self.Context = self.Context | mask
        else:
            self.Context = self.Context & ~mask

    def ContextAddress(self, length):
        shift = length - self.ContextLen
        if shift >= 0:
            addr = self.Context << shift
        else:
            addr = self.Context >> -shift
        mask = 1 << length
        addr = addr | mask
        return int(addr)

class CTWbn(ContextTree):

    def __init__(self, D, N=1000):
        super().__init__()
        self.D = int(D)
        self.DynamicCount = {}
        self.Pe = {}
        self.Pw = []
        for _ in range(self.D + 1):
            self.Pw.append({})
        self.Tail = {}
        self.InitCodeLength = 0
        self.Rep = 0
        self.OldCodeLength = 0
        self.Changed = True
        self.Frozen = False

    def _get_count(self, addr):
        if addr in self.DynamicCount:
            return self.DynamicCount[addr]
        return [0, 0]

    def _get_pe(self, addr):
        return self.Pe.get(addr, 0.0)

    def _get_pw(self, D, addr):
        return self.Pw[D].get(addr, 0.0)

    def _get_tail(self, addr):
        return self.Tail.get(addr, 0.0)

    @staticmethod
    def KTe(P, ab, bit):
        if bit == 0:
            return P + math.log2((ab[0] + 0.5) / (ab[0] + ab[1] + 1.0))
        else:
            return P + math.log2((ab[1] + 0.5) / (ab[0] + ab[1] + 1.0))

    def Update(self, bits):
        if np.isscalar(bits):
            bits = [int(bits)]
        for bit in bits:
            bit = int(bit)
            bitx = bit
            for rep in range(self.Rep + 1):
                if self.ContextLen < self.D:
                    addr_tail = self.ContextAddress(self.ContextLen)
                    self.Tail[addr_tail] = 1.0
                for i in range(self.ContextLen, -1, -1):
                    addr = self.ContextAddress(i)
                    ab = self._get_count(addr)
                    old_pe = self._get_pe(addr)
                    self.Pe[addr] = self.KTe(old_pe, ab, bitx)
                    if not self.Frozen:
                        if addr not in self.DynamicCount:
                            self.DynamicCount[addr] = [0, 0]
                        self.DynamicCount[addr][bitx] += 1
                    oldPw = self._get_pw(self.D, 1)
                    child = addr * 2
                    self.Pw[i + 1][addr] = self.Pe[addr]
                    for D in range(i + 2, self.D + 1):
                        split = self._get_pw(D, child) + self._get_pw(D, child + 1) - self._get_tail(addr)
                        self.Pw[D][addr] = self.logsp(self.Pe[addr], split) - 1.0
                ps = 2.0 ** (self._get_pw(self.D, 1) - oldPw)
                bitx = int(random.random() > ps) ^ bitx
            self.UpdateContext(bit)
            self.Changed = True
        return self

    def CalcCodeLength(self, D=None):
        if D is None:
            D = self.D
        if D == 0:
            counts = self._get_count(1)
            return float(counts[0] + counts[1])
        return float(-self._get_pw(D, 1))
mat = sio.loadmat(MAT_FILE)
original_data = mat['data'].squeeze().astype(np.uint8)
p = float(np.sum(original_data) / len(original_data))

print('ORIGINAL COIN DATA')
print('total tosses:', len(original_data))
print('ones:', int(np.sum(original_data)))
print('zeros:', int(len(original_data) - np.sum(original_data)))
print('p:', p)

def HRVRead_python(addr):
    with open(addr, 'r', encoding='utf-8', errors='ignore') as f:
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
            if random.random() > 0.5:
                hrvb[i - 1] = 1
            else:
                hrvb[i - 1] = 0
    return (hrv, hrvb, labels, equal_count)
hrv1, hrvb1, label1, hrv_equal_count = HRVRead_python(HRV_FILE)

print('HRV DATA')
print('HRV file: 16786.txt')
print('RR intervals:', len(hrv1))
print('binary HRV symbols:', len(hrvb1))
print('equal RR cases total:', hrv_equal_count)
print('first 10 RR values:', hrv1[:10])
print('first 30 HRV bits:', hrvb1[:30])
hrv_start = 30000
hrv_endpoint = 31500
hrv_segment = hrvb1[hrv_start:hrv_endpoint].copy()
assert len(hrv_segment) == 1500


print('HRV SEGMENT')
print('HRV source positions: 30001--31500')
print('segment length:', len(hrv_segment))
print('ones:', int(np.sum(hrv_segment)))
print('zeros:', int(len(hrv_segment) - np.sum(hrv_segment)))
print('empirical p:', float(np.mean(hrv_segment)))
data_hrv = original_data[15000:20000].copy()
insert_start = 2500
insert_endpoint = 4000
data_hrv[insert_start:insert_endpoint] = hrv_segment
data = data_hrv.copy()

print('Experiment 4')

print('N:', len(data))
print('coin source positions: 15001--20000')
print('HRV insertion: 2501--4000')
print('inserted length:', len(hrv_segment))
print('HRV source positions: 30001--31500')
print('maxL: 500')
print('maxD: 15')
print('typical-model p:', p)
print('coin segment p:', float(np.mean(original_data[15000:20000])))
print('experimental p:', float(np.mean(data)))
datab = data.astype(float).copy()
datab[data == 0] = -1.0
cumdataM = np.cumsum(datab)

def CTWforKnownTyp_corrected(data, p, maxD, maxL):
    N = len(data)
    CTWarray = []
    for _ in range(maxL):
        CTWarray.append(CTWbn(maxD, N))
    cl = np.zeros((N, maxL), dtype=float)
    typcl = np.zeros_like(cl)
    cldiff = np.zeros_like(cl)
    cldiffm = np.zeros(N, dtype=float)
    best_depth_matrix = np.zeros((N, maxL), dtype=np.int16)
    tcl0 = np.zeros(N + 1, dtype=float)
    s = 0
    for i_py in range(N):
        i = i_py + 1
        s += int(data[i_py])
        tcl0[i] = s * math.log2(1.0 / p) + (i - s) * math.log2(1.0 / (1.0 - p))
    for i_py in range(N):
        i = i_py + 1
        print(i, N)
        if i > maxL:
            CTWarray[0:-1] = CTWarray[1:]
            CTWarray[-1] = CTWbn(maxD, N)
        cdata = int(data[i_py])
        number_active = min(i, maxL)
        for j_py in range(number_active):
            j = j_py + 1
            CTWarray[j_py] = CTWarray[j_py].Update(cdata)
            clt = np.zeros(maxD, dtype=float)
            for D in range(1, maxD + 1):
                clt[D - 1] = CTWarray[j_py].CalcCodeLength(D) + logs(D)
            if i <= maxL:
                segment_length = i - j + 1
            else:
                segment_length = maxL - j + 1
            cl[i_py, j_py] = np.min(clt) + logs(segment_length)
            best_depth_matrix[i_py, j_py] = int(np.argmin(clt)) + 1
            interval_end = i
            interval_start = i - segment_length + 1
            typcl[i_py, j_py] = tcl0[interval_end] - tcl0[interval_start - 1]
            cldiff[i_py, j_py] = cl[i_py, j_py] - typcl[i_py, j_py]
        cldiffm[i_py] = np.min(cldiff[i_py, :number_active])
    return (cldiffm, cl, typcl, cldiff, best_depth_matrix)
maxL = 1000
maxD = 15

print("RUNNING AUTHORS' EXHAUSTIVE CTW METHOD")

print('N:', len(data))
print('maxL:', maxL)
print('maxD:', maxD)
print('Numba: NO')
print('indexed CTW: NO')
print('multiscale: NO')
print('candidate reduction: NO')
print('vectorized CTW: NO')
print('precomputed KT: NO')
print('circular buffer: NO')
print('candidate search: EXHAUSTIVE')
t0 = time.perf_counter()
cldiffm, cl, typcl, cldiff, best_depth_matrix = CTWforKnownTyp_corrected(data, p, maxD, maxL)
runtime = time.perf_counter() - t0
min_i_py = int(np.argmin(cldiffm))
min_position = min_i_py + 1
active = min(min_position, maxL)
best_j_py = int(np.argmin(cldiff[min_i_py, :active]))
best_j = best_j_py + 1
if min_position <= maxL:
    best_length = min_position - best_j + 1
else:
    best_length = maxL - best_j + 1
best_end = min_position
best_start = best_end - best_length + 1
best_depth = int(best_depth_matrix[min_i_py, best_j_py])
minimum_value = float(cldiffm[min_i_py])
true_start = 2501
true_end = 4000
hrv_scores = cldiffm[true_start - 1:true_end]
hrv_local_idx = int(np.argmin(hrv_scores))
hrv_min_position = true_start + hrv_local_idx
hrv_min_value = float(hrv_scores[hrv_local_idx])
overlap = max(0, min(best_end, true_end) - max(best_start, true_start) + 1)

if best_length > 0:
    detected_precision = overlap / best_length
else:
    detected_precision = 0.0
true_coverage = overlap / (true_end - true_start + 1)
union = best_length + (true_end - true_start + 1) - overlap

if union > 0:
    iou = overlap / union
else:
    iou = 0.0
print()

print('RESULT')

print('True HRV interval: 2501--4000')
print('Minimum cldiffm:', minimum_value)
print('Position of minimum:', min_position)
print('Best interval:', f'{best_start}--{best_end}')
print('Best interval length:', best_length)
print('Selected depth:', best_depth)
print('Minimum inside HRV:', hrv_min_value)
print('HRV min position  :', hrv_min_position)
print('overlap:', overlap)
print('detected precision    :', detected_precision)
print('true interval covered :', true_coverage)
print('IoU:', iou)
print('CTW runtime:', f'{runtime:.6f} s')
print('CTW runtime:', f'{runtime / 60:.6f} min')
positions = np.arange(1, len(data) + 1)
plt.figure(figsize=(15, 6))
plt.plot(positions, cumdataM, linewidth=0.5, label='Random Walk of Mixed Data')
plt.plot(positions, 3.0 * cldiffm, label='3 × ΔL(n)')
plt.axvspan(2501, 4000, alpha=0.1, label='Inserted normal HRV')
plt.xlabel('Samples')
plt.ylabel('Cumulative Bits / atypicality')
plt.title('Mixture of Coin Tosses and Normal HRV Data')
plt.legend()
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(positions, cldiffm)
plt.axhline(0, linestyle='--', linewidth=1)
plt.axvspan(2501, 4000, alpha=0.15, label='Inserted normal HRV')
plt.scatter([min_position], [minimum_value], s=40, label='Global minimum')
plt.xlabel('Position')
plt.ylabel('cldiffm')
plt.title('Experiment 4 — normal HRV atypicality score')
plt.legend()
plt.show()
print()

print('SUMMARY FOR REPRODUCTION — EXPERIMENT 4')

print(f'N = {len(data)}')
print(f'p = {p}')
print('HRV source = 16786.txt')
print('HRV source positions = [30001,31500]')
print(f'HRV ones = {int(np.sum(hrv_segment))}')
print(f'HRV zeros = {int(len(hrv_segment) - np.sum(hrv_segment))}')
print(f'HRV empirical p = {float(np.mean(hrv_segment))}')
print('true interval = [2501,4000]')
print(f'minimum cldiffm = {minimum_value:.12f}')
print(f'minimum position = {min_position}')
print(f'detected interval = [{best_start},{best_end}]')
print(f'detected length = {best_length}')
print(f'selected depth = {best_depth}')
print(f'minimum inside HRV = {hrv_min_value:.12f}')
print(f'position inside HRV = {hrv_min_position}')
print(f'detected precision = {detected_precision:.12f}')
print(f'true HRV coverage = {true_coverage:.12f}')
print(f'IoU = {iou:.12f}')
print(f'runtime seconds = {runtime:.6f}')
print(f'runtime minutes = {runtime / 60:.6f}')
