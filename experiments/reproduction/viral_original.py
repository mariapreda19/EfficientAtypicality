from project_paths import DATA_ROOT, RESULTS_ROOT

import os
import math
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt

def find_training_based():
    search_roots = ["/kaggle/input", "/kaggle/working"]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if ("chr1.txt" in filenames and "chr1c.txt" in filenames):
                return dirpath

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".zip"):
                    zip_path = os.path.join(dirpath, fn)
                    out = "/kaggle/working/it2016_extracted"
                    os.makedirs(out, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zz:
                            zz.extractall(out)
                    except Exception:
                        continue

                    for dp, dns, fns in os.walk(out):
                        if ("chr1.txt" in fns and "chr1c.txt" in fns):
                            return dp
    return None

ROOT = str(DATA_ROOT)

if ROOT is None:
    raise FileNotFoundError("chr1.txt and  chr1c.txt not found")

print("DATA DIRECTORY")

print("ROOT =", ROOT)
print("chr1.txt exists  =", os.path.exists(os.path.join(ROOT, "chr1.txt")))
print("chr1c.txt exists =", os.path.exists(os.path.join(ROOT, "chr1c.txt")))

def logs(n):
    if n <= 1:
        return 0.0
    m = 0.0
    k = math.log2(n)
    while k > 0:
        m += k
        k = math.log2(k)
    return m

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
        length = int(length)
        shift = length - self.ContextLen
        if shift >= 0:
            addr = self.Context << shift
        else:
            addr = self.Context >> (-shift)
        addr |= 1 << length
        return int(addr)

    def UpdateContext(self, bit):
        bit = int(bit)
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
        converted_bits = []
        for b in bits:
            converted_bits.append(int(b))
        bits = converted_bits

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
        self.Pw = []
        for _ in range(self.D):
            self.Pw.append({})
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

    @staticmethod
    def KTe(old_pe, counts, bit):
        a = counts[0]
        b = counts[1]
        if bit == 0:
            p = (a + 0.5) / (a + b + 1.0)
        else:
            p = (b + 0.5) / (a + b + 1.0)

        return old_pe + math.log2(p)

    def Update(self, bit):
        bit = int(bit)
        bitx = bit
        for rep in range(self.Rep + 1):
            if self.ContextLen < self.D:
                addr_tail = self.ContextAddress(self.ContextLen)
                self.Tail[addr_tail] = 1.0
            for i in range(self.ContextLen, -1, -1):
                addr = self.ContextAddress(i)
                counts = self._counts(addr)
                old_pe = self._get_pe(addr)
                new_pe = self.KTe(old_pe, counts, bitx)
                self._set_pe(addr, new_pe)
                if not self.Frozen:
                    new_counts = [counts[0], counts[1]]
                    new_counts[bitx] += 1
                    self.DynamicCount[addr] = new_counts

                oldPw = self._get_pw(self.D, 1)
                child = addr * 2
                self._set_pw(i + 1, addr, new_pe)
                for DD in range(i + 2, self.D + 1):
                    children_value = (
                        self._get_pw(DD, child) + self._get_pw(DD, child + 1) - self.Tail.get(addr, 0.0)
                    )

                    new_pw = logsp(new_pe, children_value) - 1.0
                    self._set_pw(DD, addr, new_pw)

            self.UpdateContext(bit)
            self.Changed = True
        return self

    def CalcCodeLength(self, D=None):
        if D is None:
            D = self.D
        D = int(D)
        if D == 0:
            counts = self._counts(1)
            return float(counts[0] + counts[1])
        return float(-self._get_pw(D, 1))

    @property
    def CodeLength(self):
        return self.CalcCodeLength()

    def ResetContext(self):
        self.Context = 0
        self.ContextLen = 0
        return self

class CTWbFrozen(ContextTree):
    def __init__(self, CTWi, D=None, PreCalc=1):
        if D is None:
            cl = []
            for d in range(CTWi.D + 1):
                cl.append(CTWi.CalcCodeLength(d))

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
                c0 = addr * 2
                c1 = c0 + 1
                Piid = (CTWi._get_pe(addr) - logsp(CTWi._get_pe(addr),
                        CTWi._get_pw(self.D, c0) +
                        CTWi._get_pw(self.D, c1)))
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
            if shift >= 0:
                addr = self.Context << shift
            else:
                addr = self.Context >> (-shift)

            addr |= 1 << i
            P1iid = self._update_prob_at_addr(addr, P1Niid)
            if P1iid is not None:
                P1Niid = P1iid
        P1Niid = min(max(P1Niid, 1e-300), 1.0 - 1e-16)
        if bit:
            self.Pet += math.log2(P1Niid)
        else:
            self.Pet += math.log2(1.0 - P1Niid)
        self.UpdateContext(bit)
        return self

    def _update_prob_at_addr(self, addr, current):
        P1iid = self.P1iid.get(addr, 0.0)
        if P1iid == 0.0:
            return None
        Piid_log = self.Piid.get(addr, 0.0)
        Piid = 2.0 ** Piid_log
        return P1iid * Piid + (1.0 - Piid) * current

    def CalcCodeLength(self):
        return float(-self.Pet)

def DNAReadFunc(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    DataBin = []
    DataNum = []
    for ch in text:
        c = ch.upper()
        if c == "A":
            DataBin.extend([0, 0])
            DataNum.append(0)
        elif c == "C":
            DataBin.extend([0, 1])
            DataNum.append(1)
        elif c == "G":
            DataBin.extend([1, 0])
            DataNum.append(2)
        elif c == "T":
            DataBin.extend([1, 1])
            DataNum.append(3)

    return np.asarray(DataBin, dtype=np.uint8), np.asarray(DataNum, dtype=np.uint8)

def DNAPlotTogether_sp_ep(DataBin1, DataNum1, DataBin2, DataNum2):
    minlen = min(len(DataNum1), len(DataNum2))
    D1 = DataNum1[:minlen]
    D2 = DataNum2[:minlen]
    diff = np.flatnonzero(D1 != D2)
    if len(diff) == 0:
        raise RuntimeError("DNAPlotTogether: no differing base found.")
    first_diff_base = int(diff[0]) + 1
    LenDiff = abs(len(DataNum1) - len(DataNum2))
    sp = 2 * first_diff_base
    ep = sp + 2 * LenDiff
    return int(sp), int(ep)

d1 = os.path.join(ROOT, "chr1.txt")
d2 = os.path.join(ROOT, "chr1c.txt")
DataBin1, DataNum1 = DNAReadFunc(d1)
DataBin2, DataNum2 = DNAReadFunc(d2)
sp, ep = DNAPlotTogether_sp_ep(DataBin1, DataNum1, DataBin2, DataNum2)
print("FIGURE 10 — ORIGINAL DNA DATA")
print("chr1 bases:", len(DataNum1))
print("chr1c bases:", len(DataNum2))
print("chr1 bits:", len(DataBin1))
print("chr1c bits:", len(DataBin2))
print("DNAPlotTogether sp:", sp)
print("DNAPlotTogether ep:", ep)

DataTrain = np.asarray(DataBin2[0:10000], dtype=np.uint8)
DataTest = np.concatenate([np.asarray(DataBin2[10000:14180], dtype=np.uint8),
        np.asarray(DataBin2[ep - 1:25000], dtype=np.uint8)])

maxD = 16
maxL = 1000
print("EXACT DNAmain.m SETUP")
print("DataTrain length:", len(DataTrain))
print("DataTest length:", len(DataTest))
print("first test piece:", 14180 - 10000)
print("second test piece:", 25000 - (ep - 1))
print("maxD:", maxD)
print("maxL:", maxL)
assert len(DataNum1) == 18995
assert len(DataNum2) == 21305
assert len(DataBin1) == 37990
assert len(DataBin2) == 42610
assert sp == 13568
assert ep == 18188
assert len(DataTrain) == 10000
assert len(DataTest) == 10993
assert maxD == 16
assert maxL == 400

print("FIGURE 10 SETUP CHECK: PASS")
print("TRAINING TYPICAL CTW")

total_start = time.perf_counter()
training_start = time.perf_counter()
T = len(DataTrain)
Typical = CTWbn(maxD, T)
Typical.SetContext(DataTrain[:maxD])
TrainDataRest = DataTrain[maxD:]
for i in range(len(TrainDataRest)):
    Typical.Update(int(TrainDataRest[i]))
typical_code_length = Typical.CodeLength
print("Typical.CodeLength:", typical_code_length)
Typical.ResetContext()
Frozen = CTWbFrozen(Typical)
print("Frozen depth:", Frozen.D)
training_runtime = time.perf_counter() - training_start
print("training runtime:", f"{training_runtime:.6f} s")
CTWarray = []
for _ in range(maxL):
    CTWarray.append(CTWbn(maxD, len(DataTest)))
n = len(DataTest)
cl = np.zeros((n, maxL), dtype=np.float64)
typcl = np.zeros((n, maxL), dtype=np.float64)
cldiff = np.zeros((n, maxL), dtype=np.float64)
cldiffm = np.zeros(n, dtype=np.float64)
TCL = np.zeros(n, dtype=np.float64)
winning_depth_matrix = np.zeros((n, maxL), dtype=np.int16)
print("RUNNING LITERAL CTWfunc.m SCAN")

scan_start = time.perf_counter()
for py_i in range(n):
    i = py_i + 1
    cdata = int(DataTest[py_i])
    Frozen.Update(cdata)
    TCL[py_i] = Frozen.CalcCodeLength()
    upto = min(i, maxL)
    for py_j in range(upto):
        j = py_j + 1
        tree = CTWarray[py_j]
        tree.Update(cdata)
        best_value = float("inf")
        best_D = None
        for D in range(1, maxD + 1):
            value = tree.CalcCodeLength(D) + logs(D)
            if value < best_value:
                best_value = value
                best_D = D

        cl[py_i, py_j] = best_value + logs(maxL - j + 1)
        winning_depth_matrix[py_i, py_j] = best_D
    if i > maxL:
        CTWarray[:-1] = CTWarray[1:]
        CTWarray[-1] = CTWbn(maxD, len(DataTest))

    if i % 250 == 0 or i == n:
        elapsed = time.perf_counter() - scan_start
        print(f"scan {i:5d}/{n} | elapsed {elapsed:9.2f} s")

print("CALCULATING cldiff — LITERAL MATLAB INDEXING")
for py_i in range(n):
    i = py_i + 1
    if i < maxL:
        for py_j in range(i):
            j = (py_j + 1)
            matlab_previous_index = j
            typ_value = TCL[py_i] - TCL[matlab_previous_index - 1]
            typcl[py_i, py_j] = typ_value
            cldiff[py_i, py_j] = cl[py_i, py_j] - typ_value
        cldiffm[py_i] = np.min(cldiff[py_i, :i])
    else:
        for py_j in range(maxL):
            j = (py_j + 1)
            matlab_index = i - (maxL - j)
            typ_value = TCL[py_i] - TCL[matlab_index - 1]
            typcl[py_i, py_j] = typ_value
            cldiff[py_i, py_j] = cl[py_i, py_j] - typ_value
        cldiffm[py_i] = np.min(cldiff[py_i, :])

scan_runtime = time.perf_counter() - scan_start
total_runtime = time.perf_counter() - total_start
global_py_i = int(np.argmin(cldiffm))
global_i = global_py_i + 1
if global_i < maxL:
    valid_cols = global_i
else:
    valid_cols = maxL
global_py_j = int(np.argmin(cldiff[global_py_i, :valid_cols]))
global_j = global_py_j + 1
global_D = int(winning_depth_matrix[global_py_i, global_py_j])
if global_i > maxL:
    nominal_length = maxL - global_j + 1
else:
    nominal_length = global_i - global_j + 1

nominal_start = global_i - nominal_length + 1
nominal_end = global_i
print("DataTrain length:", len(DataTrain))
print("DataTest length:", len(DataTest))
print("maxD:", maxD)
print("maxL:", maxL)
print("Frozen depth:", Frozen.D)
print("minimum cldiffm:",float(cldiffm[global_py_i]))
print("minimum position:", global_i)
print("winning MATLAB j:", global_j)
print("winning depth D:", global_D)
print("nominal length:", nominal_length)
print("nominal interval:", ( nominal_start, nominal_end))
print("training runtime:", f"{training_runtime:.6f} s")
print("scan runtime:", f"{scan_runtime:.6f} s")
print("total runtime:", f"{total_runtime:.6f} s")
print("total runtime minutes :", f"{total_runtime / 60.0:.6f}")
plt.figure(figsize=(16, 6))
plt.plot(np.arange(1, n + 1),
    3.0 * cldiffm,
    linewidth=1.0,
    label="3 * cldiffm"
)

plt.axhline(0.0, linewidth=1.0)
plt.axvline(global_i, linestyle="--", linewidth=1.0, label="global minimum")
plt.xlabel("Position in DataTest")
plt.ylabel("3 * cldiffm")
plt.title("Experiment 6 — Figure 10 viral infection — literal CTWfunc")
plt.legend()
plt.show()
OUTPUT = str(RESULTS_ROOT / "experiment6_literal_matlab_port.npz")
np.savez_compressed(OUTPUT,
    DataTrain=DataTrain,
    DataTest=DataTest,
    DataBin1=DataBin1,
    DataBin2=DataBin2,
    DataNum1=DataNum1,
    DataNum2=DataNum2,
    TCL=TCL,
    cl=cl,
    typcl=typcl,
    cldiff=cldiff,
    cldiffm=cldiffm,
    winning_depth_matrix=winning_depth_matrix,
    maxD=maxD,
    maxL=maxL,
    sp=sp,
    ep=ep,
    typical_code_length=typical_code_length,
    frozen_depth=Frozen.D,
    minimum_cldiffm=float(cldiffm[global_py_i]),
    minimum_position=global_i,
    winning_j=global_j,
    winning_D=global_D,
    nominal_start=nominal_start,
    nominal_end=nominal_end,
    nominal_length=nominal_length,
    training_runtime=training_runtime,
    scan_runtime=scan_runtime,
    total_runtime=total_runtime
)
