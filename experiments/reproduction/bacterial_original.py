from project_paths import DATA_ROOT, RESULTS_ROOT
import os
import glob
import math
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt
ROOT = str(DATA_ROOT)
print('\nROOT =', ROOT)
for filename in ['chr6.txt', 'chr6c.txt']:
    p = os.path.join(ROOT, filename)
    print(filename, 'exists =', os.path.exists(p))
print()

def logs(n):
    m = 0.0
    k = math.log2(n)
    while k > 0:
        m += k
        k = math.log2(k)
    return m

class ContextTree:
    def __init__(self):
        self.D = 1
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
        return self

    def ContextAddress(self, length):
        shift = length - self.ContextLen
        if shift >= 0:
            addr = self.Context << shift
        else:
            addr = self.Context >> -shift
        addr |= 1 << length
        return addr

    def SetContext(self, bits):
        converted = []
        for x in bits:
            converted.append(int(x))
        bits = converted
        if len(bits) > self.D - 1:
            bits = bits[-(self.D - 1):]
        self.ContextLen = len(bits)
        context = 0
        for k, bit in enumerate(reversed(bits)):
            if bit:
                context += 1 << k
        self.Context = context
        return self

class CTWbn(ContextTree):

    def __init__(self, D, N=1000):
        super().__init__()
        self.D = int(D)
        self.DynamicCount = {}
        self.Pe = {}
        self.Pw = [None]
        for _ in range(self.D):
            self.Pw.append({})
        self.Tail = {}
        self.InitCodeLength = 0.0
        self.Rep = 0
        self.OldCodeLength = 0.0
        self.Changed = True
        self.Frozen = False

    def counts(self, addr):
        return self.DynamicCount.get(addr, [0, 0])

    def pe(self, addr):
        return self.Pe.get(addr, 0.0)

    def pw(self, D, addr):
        return self.Pw[D].get(addr, 0.0)

    def tail(self, addr):
        return self.Tail.get(addr, 0.0)

    def ResetContext(self):
        self.ContextLen = 0
        self.Context = 0
        return self

    @staticmethod
    def KTe(P, ab, bit):
        if bit == 0:
            P = P + math.log2((ab[0] + 0.5) / (ab[0] + ab[1] + 1.0))
        else:
            P = P + math.log2((ab[1] + 0.5) / (ab[0] + ab[1] + 1.0))
        return P

    def Update(self, bits):
        if np.isscalar(bits):
            bits = [int(bits)]
        else:
            converted = []
            for x in bits:
                converted.append(int(x))
            bits = converted
        for bit in bits:
            bitx = bit
            for _ in range(self.Rep + 1):
                if self.ContextLen < self.D:
                    addr_tail = self.ContextAddress(self.ContextLen)
                    self.Tail[addr_tail] = 1.0
                oldPw = 0.0
                for i in range(self.ContextLen, -1, -1):
                    addr = self.ContextAddress(i)
                    ab = self.counts(addr)
                    new_pe = self.KTe(self.pe(addr), ab, bitx)
                    self.Pe[addr] = new_pe
                    if not self.Frozen:
                        new_ab = [ab[0], ab[1]]
                        new_ab[bitx] += 1
                        self.DynamicCount[addr] = new_ab
                    oldPw = self.pw(self.D, 1)
                    child = addr * 2
                    self.Pw[i + 1][addr] = new_pe
                    for depth in range(i + 2, self.D + 1):
                        child_sum = self.pw(depth, child) + self.pw(depth, child + 1) - self.tail(addr)
                        self.Pw[depth][addr] = self.logsp(new_pe, child_sum) - 1.0
                ps = 2.0 ** (self.pw(self.D, 1) - oldPw)
            self.UpdateContext(bit)
            self.Changed = True
        return self

    def CalcCodeLength(self, D=None):
        if D is None:
            D = self.D
        if D == 0:
            ab = self.counts(1)
            return float(ab[0] + ab[1])
        return float(-self.pw(D, 1))

    @property
    def CodeLength(self):
        if self.Changed:
            self.Changed = False
            self.OldCodeLength = self.CalcCodeLength()
        return self.OldCodeLength

class CTWbFrozen(ContextTree):

    def __init__(self, CTWi, PreCalc=1, Context=None, ContextLen=None, D=None):
        super().__init__()
        self.PreCalc = int(PreCalc)
        if Context is None:
            self.Context = CTWi.Context
            self.ContextLen = CTWi.ContextLen
        else:
            self.Context = int(Context)
            self.ContextLen = int(ContextLen)
        if D is None:
            cl = []
            for d in range(CTWi.D + 1):
                cl.append(CTWi.CalcCodeLength(d))
            D = int(np.argmin(cl))
        self.D = max(int(D), 2)
        self.Size = 2 ** self.D - 1
        self.Piid = {}
        self.P1iid = {}
        self.Pet = 0.0
        self.CTW = None
        if self.PreCalc:
            self.InitProb(CTWi)
        else:
            self.CTW = CTWi

    def CalcProb(self, addr, CTWi):
        P1iid = 0.0
        Piid = 0.0
        maxin = (self.Size + 1) / 2 - 1
        if CTWi.pw(self.D, addr) != 0.0:
            ab = CTWi.counts(addr)
            P1iid = (ab[1] + 0.5) / (ab[1] + ab[0] + 1.0)
            if addr <= maxin:
                c0, c1 = CTWi.Children(addr)
                Piid = CTWi.pe(addr) - self.logsp(CTWi.pe(addr), CTWi.pw(self.D, c0) + CTWi.pw(self.D, c1))
        return (P1iid, Piid)

    def InitProb(self, CTWi):
        for addr in range(1, self.Size + 1):
            if CTWi.pw(self.D, addr) != 0.0:
                p1, pi = self.CalcProb(addr, CTWi)
                if p1 != 0.0:
                    self.P1iid[addr] = p1
                if pi != 0.0:
                    self.Piid[addr] = pi
        return self

    def CalcCodeLength(self):
        return float(-self.Pet)

    def Restart(self):
        self.Pet = 0.0
        self.Context = 0
        self.ContextLen = 0
        return self

    def Update(self, bits):
        if np.isscalar(bits):
            bits = [int(bits)]
        else:
            converted = []
            for x in bits:
                converted.append(int(x))
            bits = converted
        if self.PreCalc:
            for bit in bits:
                P1Niid = 0.5
                for i in range(self.ContextLen, -1, -1):
                    shift = i - self.ContextLen
                    if shift >= 0:
                        addr = self.Context << shift
                    else:
                        addr = self.Context >> -shift
                    addr |= 1 << i
                    p1 = self.P1iid.get(addr, 0.0)
                    if p1 != 0.0:
                        pi_log = self.Piid.get(addr, 0.0)
                        pi = 2.0 ** pi_log
                        P1Niid = p1 * pi + (1.0 - pi) * P1Niid
                if bit:
                    self.Pet += math.log2(P1Niid)
                else:
                    self.Pet += math.log2(1.0 - P1Niid)
                self.UpdateContext(bit)
        else:
            for bit in bits:
                P1Niid = 0.5
                for i in range(self.ContextLen, -1, -1):
                    shift = i - self.ContextLen
                    if shift >= 0:
                        addr = self.Context << shift
                    else:
                        addr = self.Context >> -shift
                    addr |= 1 << i
                    if self.CTW.pw(self.D, addr) != 0.0:
                        p1, pi_log = self.CalcProb(addr, self.CTW)
                        pi = 2.0 ** pi_log
                        P1Niid = p1 * pi + (1.0 - pi) * P1Niid
                if bit:
                    self.Pet += math.log2(P1Niid)
                else:
                    self.Pet += math.log2(1.0 - P1Niid)
                self.UpdateContext(bit)
        return self

def DNAReadFunc(filename):
    DataBin = []
    DataNum = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        for c in line:
            letter = c.lower()
            if letter == 'a':
                DataBin.extend([0, 0])
                DataNum.append(0)
            elif letter == 'c':
                DataBin.extend([0, 1])
                DataNum.append(1)
            elif letter == 'g':
                DataBin.extend([1, 0])
                DataNum.append(2)
            elif letter == 't':
                DataBin.extend([1, 1])
                DataNum.append(3)
    return (np.asarray(DataBin, dtype=np.uint8), np.asarray(DataNum, dtype=np.uint8))

def DNAPlotTogether_sp_ep(DataNum1, DataNum2):
    D1 = DataNum1.copy()
    D2 = DataNum2.copy()
    if len(D1) > len(D2):
        LenDiff = len(D1) - len(D2)
        D1 = D1[:len(D2)]
    else:
        LenDiff = len(D2) - len(D1)
        D2 = D2[:len(D1)]
    differences = np.flatnonzero(D1 - D2 != 0)
    first_difference = int(differences[0]) + 1
    sp = 2 * first_difference
    ep = sp + 2 * LenDiff
    return (sp, ep)
d1 = os.path.join(ROOT, 'chr6.txt')
d2 = os.path.join(ROOT, 'chr6c.txt')
DataBin1, DataNum1 = DNAReadFunc(d1)
DataBin2, DataNum2 = DNAReadFunc(d2)
sp, ep = DNAPlotTogether_sp_ep(DataNum1, DataNum2)
print('FIGURE 9 — DATA')
print('chr6 bases:', len(DataNum1))
print('chr6c bases:', len(DataNum2))
print('chr6 bits:', len(DataBin1))
print('chr6c bits:', len(DataBin2))
print('DNAPlotTogether sp:', sp)
print('DNAPlotTogether ep:', ep)
DataTrain = DataBin2[10000:20000].copy()
DataTest = np.concatenate((DataBin2[0:2000], DataBin2[4999:10000]))
maxD = 16
maxL = 2500
print('EXACT DNAmain.m SETUP')
print('DataTrain length:', len(DataTrain))
print('DataTest length:', len(DataTest))
print('maxD:', maxD)
print('maxL:', maxL)
assert len(DataTrain) == 10000
assert len(DataTest) == 7001
total_start = time.perf_counter()
T = len(DataTrain)
Typical = CTWbn(maxD, T)
Typical.SetContext(DataTrain[:maxD])
TrainDataCTW = DataTrain[maxD:]
training_start = time.perf_counter()
for idx, bit in enumerate(TrainDataCTW, start=1):
    Typical.Update(int(bit))
    if idx % 1000 == 0:
        print(f'training {idx}/{len(TrainDataCTW)}')
training_runtime = time.perf_counter() - training_start
print()
print('Typical.CodeLength:', Typical.CodeLength)
Typical.ResetContext()
Frozen = CTWbFrozen(Typical)
print('Frozen depth D:', Frozen.D)
N = len(DataTest)
CTWarray = []
for _ in range(maxL):
    CTWarray.append(CTWbn(maxD, N))
cl = np.zeros((N, maxL), dtype=np.float64)
typcl = np.zeros((N, maxL), dtype=np.float64)
cldiff = np.zeros((N, maxL), dtype=np.float64)
cldiffm = np.zeros(N, dtype=np.float64)
TCL = np.zeros(N, dtype=np.float64)
winning_depth = np.zeros((N, maxL), dtype=np.int16)
print('STARTING LITERAL CTWfunc.m SCAN')
scan_start = time.perf_counter()
for py_i in range(N):
    i = py_i + 1
    cdata = int(DataTest[py_i])
    Frozen.Update(cdata)
    TCL[py_i] = Frozen.CalcCodeLength()
    limit = min(i, maxL)
    for py_j in range(limit):
        j = py_j + 1
        CTWarray[py_j].Update(cdata)
        clt = np.zeros(maxD, dtype=np.float64)
        for D in range(1, maxD + 1):
            clt[D - 1] = CTWarray[py_j].CalcCodeLength(D) + logs(D)
        best_index = int(np.argmin(clt))
        cl[py_i, py_j] = clt[best_index] + logs(maxL - j + 1)
        winning_depth[py_i, py_j] = best_index + 1
    if i > maxL:
        CTWarray[:-1] = CTWarray[1:]
        CTWarray[-1] = CTWbn(maxD, N)
    if i == 1 or i % 100 == 0 or i == N:
        elapsed = time.perf_counter() - scan_start
        print(f'{i:5d}/{N} | elapsed {elapsed:9.2f} s')
scan_runtime = time.perf_counter() - scan_start
print('Calculating typcl / cldiff exactly as CTWfunc.m')
for py_i in range(N):
    i = py_i + 1
    if i < maxL:
        for py_j in range(i):
            j = py_j + 1
            typcl[py_i, py_j] = TCL[py_i] - TCL[j - 1]
            cldiff[py_i, py_j] = cl[py_i, py_j] - typcl[py_i, py_j]
        cldiffm[py_i] = np.min(cldiff[py_i, :i])
    else:
        for py_j in range(maxL):
            j = py_j + 1
            k = i - (maxL - j)
            typcl[py_i, py_j] = TCL[py_i] - TCL[k - 1]
            cldiff[py_i, py_j] = cl[py_i, py_j] - typcl[py_i, py_j]
        cldiffm[py_i] = np.min(cldiff[py_i, :])
total_runtime = time.perf_counter() - total_start
min_py_i = int(np.argmin(cldiffm))
minimum = float(cldiffm[min_py_i])
minimum_position = min_py_i + 1
winning_py_j = int(np.argmin(cldiff[min_py_i, :]))
winning_j = winning_py_j + 1
winning_D = int(winning_depth[min_py_i, winning_py_j])
nominal_length = maxL - winning_j + 1
nominal_start = minimum_position - nominal_length + 1
print()

print('EXPERIMENT 5 — LITERAL MATLAB -> PYTHON')

print('DataTrain length:', len(DataTrain))
print('DataTest length:', len(DataTest))
print('maxD:', maxD)
print('maxL:', maxL)
print('Frozen depth:', Frozen.D)
print('minimum cldiffm :', minimum)
print('minimum position:', minimum_position)
print('winning MATLAB j:', winning_j)
print('winning depth D :', winning_D)
print('nominal length  :', nominal_length)
print('nominal interval:', (nominal_start, minimum_position))
print('training runtime:', f'{training_runtime:.6f} s')
print('scan runtime:', f'{scan_runtime:.6f} s')
print('total runtime:', f'{total_runtime:.6f} s')
print('total runtime minutes :', f'{total_runtime / 60:.6f}')
regions = {'before': (1, 1616), 'bacterial fragment 1': (1617, 2000), 'bacterial fragment 2': (2001, 4061),
           'after': (4062, N)}


print('cldiffm BY REGION — REPORTING ONLY')

for name, (a, b) in regions.items():
    values = cldiffm[a - 1:b]
    local_idx = int(np.argmin(values))
    pos = a + local_idx
    score = float(values[local_idx])
    print(f'{name:24s} [{a:4d},{b:4d}] min={score: .12f} at {pos}')
datab = DataTest.astype(np.float64).copy()
datab[DataTest == 0] = -1.0
DataCum = np.cumsum(datab)
plt.figure(figsize=(15, 6))
plt.plot(np.arange(1, N + 1), DataCum, linewidth=0.001, label='Random Walk of DNA Data')
plt.plot(np.arange(1, N + 1), 3.0 * cldiffm, label='3 × cldiffm')
plt.axvline(1617, linestyle='--')
plt.axvline(2000, linestyle='--')
plt.axvline(4061, linestyle='--')
plt.title('Figure 9 — Bacterial Infection')
plt.xlabel('Sample')
plt.legend()
plt.show()
plt.figure(figsize=(15, 6))
plt.plot(np.arange(1, N + 1), cldiff[:, 0], label='cldiff(:,1)')
plt.plot(np.arange(1, N + 1), cldiff[:, 299], label='cldiff(:,300)')
plt.plot(np.arange(1, N + 1), cldiff[:, 699], label='cldiff(:,700)')
plt.plot(np.arange(1, N + 1), cldiffm, linewidth=3, label='cldiffm')
plt.legend()
plt.show()
save_path = str(RESULTS_ROOT / 'experiment5_literal_matlab_port.npz')
np.savez_compressed(save_path, DataTrain=DataTrain, DataTest=DataTest, sp=sp, ep=ep, maxD=maxD,
                    maxL=maxL, TCL=TCL, cl=cl, typcl=typcl, cldiff=cldiff, cldiffm=cldiffm,
                    winning_depth=winning_depth, minimum=minimum, minimum_position=minimum_position,
                    winning_j=winning_j, winning_D=winning_D, nominal_length=nominal_length,
                    nominal_start=nominal_start, training_runtime=training_runtime, scan_runtime=scan_runtime,
                    total_runtime=total_runtime)
print('Saved results to:', save_path)

