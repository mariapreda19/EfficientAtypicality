from project_paths import DATA_ROOT

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
import time

BASE = str(DATA_ROOT)

MAT_FILE = f"{BASE}/Data40kTosses.mat"

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
        return c0, c1

    def logsp(self, x, y):
        a = min(x, y)
        b = max(x, y)
        return b + math.log1p(2.0 ** (a - b)) * self.ln2i

    def UpdateContext(self, bit):
        if self.ContextLen < self.D - 1:
            self.ContextLen += 1
        else:
            self.Context >>= 1
        mask = 1 << (self.ContextLen - 1)
        if bit:
            self.Context = self.Context | mask
        else:
            self.Context = self.Context & ~mask

    def ContextAddress(self, length):
        shift = length - self.ContextLen
        if shift >= 0:
            addr = self.Context << shift
        else:
            addr = self.Context >> (-shift)
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
print("sum == 16 cases:",int(np.sum(bit_sums == 16)))
print("First 10 states:", rdata[:10])
print("First 30 output bits:", bdata[:30])

data = original_data[15000:20000].copy()

start = 3000
endpoint = 4000
data[start:endpoint] = bdata[:endpoint - start]

print("EXPERIMENT")

print("N:", len(data))
print("coin source positions: 15001--20000")
print("RANDU insertion: 3001--4000")
print("inserted length:", endpoint - start)
print("maxL: 500")
print("maxD: 15")
print("coin segment p:",float(np.mean(original_data[15000:20000])))
print("experimental p:", float(np.mean(data)))

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
    return cldiffm, cl, typcl, cldiff, best_depth_matrix

maxL = 1000
maxD = 15

print("RUNNING AUTHORS' EXHAUSTIVE CTW METHOD")
print("N:", len(data))
print("maxL :", maxL)
print("maxD :", maxD)
print("Numba : NO")
print("indexed CTW: NO")
print("multiscale: NO")
print("candidate reduction: NO")
print("vectorized CTW: NO")
print("precomputed KT: NO")
print("circular buffer: NO")
print("candidate search: EXHAUSTIVE")

t0 = time.perf_counter()
cldiffm, cl, typcl, cldiff, best_depth_matrix = CTWforKnownTyp_corrected(data, p, maxD, maxL)
runtime = time.perf_counter() - t0
min_i_py = int(np.argmin(cldiffm))
min_position = min_i_py + 1
if min_position <= maxL:
    active = min_position
else:
    active = maxL

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
randu_start = 3001
randu_end = 4000
randu_scores = cldiffm[randu_start - 1:randu_end]
randu_local_idx = int(np.argmin(randu_scores))
randu_min_position = randu_start + randu_local_idx
randu_min_value = float(randu_scores[randu_local_idx])
print("RESULT")
print("True RANDU interval: 3001--4000")
print("Minimum cldiffm:", minimum_value)
print("Position of minimum:", min_position)
print("Best interval:", f"{best_start}--{best_end}")
print("Best interval length:", best_length)
print("Selected depth:", best_depth)
print("Minimum inside RANDU:", randu_min_value)
print("RANDU min position:", randu_min_position)
print("CTW runtime:", f"{runtime:.6f} s")
print("CTW runtime:", f"{runtime/60:.3f} min")

positions = np.arange(1, len(data) + 1)
plt.figure(figsize=(15, 6))
plt.plot(positions, cumdataM, linewidth=0.5, label="Random Walk of Mixed Data")
plt.plot(positions, 3.0 * cldiffm,label="3 × ΔL(n)")
plt.axvspan(3001, 4000, alpha=0.10,label="RANDU")
plt.xlabel("Samples")
plt.ylabel("Cumulative Bits / atypicality")
plt.title("Mixture of Coin Tosses and RANDU Data")
plt.legend()
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(positions, cldiffm)
plt.axhline(0,linestyle="--",linewidth=1)
plt.axvspan(3001, 4000, alpha=0.15, label="Inserted RANDU")
plt.scatter([min_position],[minimum_value],s=40,label="Global minimum")
plt.xlabel("Position")
plt.ylabel("cldiffm")
plt.title("RANDU atypicality score")
plt.legend()
plt.show()
print("SUMMARY FOR REPRODUCTION")
print(f"N = {len(data)}")
print(f"p = {p}")
print(f"RANDU ones = {int(np.sum(bdata))}")
print(f"RANDU zeros = {int(len(bdata)-np.sum(bdata))}")
print("RANDU threshold = 15.5")
print("true interval = [3001,4000]")
print(f"minimum cldiffm = {minimum_value:.12f}")
print(f"minimum position = {min_position}")
print(f"detected interval = [{best_start},{best_end}]")
print(f"detected length = {best_length}")
print(f"selected depth = {best_depth}")
print(f"minimum inside RANDU = {randu_min_value:.12f}")
print(f"position inside RANDU = {randu_min_position}")
print(f"runtime seconds = {runtime:.6f}")
print(f"runtime minutes = {runtime/60:.6f}")
