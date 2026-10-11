from project_paths import DATA_ROOT

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

BASE_PATH = str(DATA_ROOT)
COIN_FILE = f"{BASE_PATH}/Data40kTosses.mat"
TEXT_FILE = f"{BASE_PATH}/the home of abraham.txt"
total_start_time = time.perf_counter()

def logs(n):
    m = 0.0
    k = np.log2(n)
    while k > 0:
        m += k
        k = np.log2(k)
    return m

def text_to_word_len(filename):
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
    bin_data = np.zeros(len(wordlen) - 1, dtype=int)
    for i in range(1, len(wordlen)):
        d = wordlen[i] - wordlen[i - 1]
        if d > 0:
            bin_data[i - 1] = 1
        elif d < 0:
            bin_data[i - 1] = 0
        else:
            bin_data[i - 1] = int(np.random.rand() > 0.5)

    return wordlen, bin_data

class CTWbn:
    def __init__(self, D, N=1000):
        self.D = D
        self.Context = 0
        self.ContextLen = 0
        size = 2 ** D
        self.DynamicCount = np.zeros((size, 2), dtype=np.int64)
        self.Pe = np.zeros(size, dtype=float)
        self.Pw = []
        for d in range(1, D + 1):
            self.Pw.append(np.zeros(2 ** d, dtype=float))
        self.Tail = np.zeros(size, dtype=float)
        self.Frozen = False
        self.Rep = 0

    def context_address(self, length):
        if self.ContextLen == 0:
            lower = 0
        else:
            mask = (1 << self.ContextLen) - 1
            lower = self.Context & mask
        if length < self.ContextLen:
            lower >>= (self.ContextLen - length)
        elif length > self.ContextLen:
            lower <<= (length - self.ContextLen)
        addr = lower | (1 << length)
        return addr

    def update_context(self, bit):
        if self.ContextLen < self.D - 1:
            self.ContextLen += 1
        else:
            self.Context >>= 1
        position = self.ContextLen - 1
        if bit:
            self.Context |= (1 << position)
        else:
            self.Context &= ~(1 << position)

    @staticmethod
    def logsp(x, y):
        a = min(x, y)
        b = max(x, y)
        return b + np.log1p(2 ** (a - b)) / np.log(2)

    @staticmethod
    def KTe(P, ab, bit):
        if bit == 0:
            return P + np.log2((ab[0] + 0.5) / (np.sum(ab) + 1))
        else:
            return P + np.log2((ab[1] + 0.5) / (np.sum(ab) + 1))

    def update(self, bits):
        if np.isscalar(bits):
            bits = [int(bits)]
        for bit in bits:
            bitx = int(bit)
            for _ in range(self.Rep + 1):
                if self.ContextLen < self.D:
                    addr = self.context_address(self.ContextLen)
                    self.Tail[addr] = 1
                for i in range(self.ContextLen, -1, -1):
                    addr = self.context_address(i)
                    self.Pe[addr] = self.KTe(self.Pe[addr], self.DynamicCount[addr], bitx)
                    if not self.Frozen:
                        self.DynamicCount[addr, bitx] += 1

                    self.Pw[i][addr] = self.Pe[addr]
                    child = addr * 2
                    for depth in range(i + 2, self.D + 1):
                        self.Pw[depth - 1][addr] = (self.logsp(self.Pe[addr], self.Pw[depth - 1][child] +
                                self.Pw[depth - 1][child + 1] - self.Tail[addr]) - 1)

            self.update_context(bit)
        return self

    def calc_code_length(self, D=None):
        if D is None:
            D = self.D
        if D == 0:
            return float(np.sum(self.DynamicCount[1]))
        return float(-self.Pw[D - 1][1])

def CTWforKnownTyp(data, p, maxD, maxL):
    data = np.asarray(data, dtype=int)
    n = len(data)
    print("Creating CTW trees...")
    CTWarray = []
    for _ in range(maxL):
        CTWarray.append(CTWbn(maxD, n))
    cl = np.zeros((n, maxL), dtype=float)
    cldiff = np.zeros((n, maxL), dtype=float)
    cldiffm = np.zeros(n, dtype=float)
    tcl = np.zeros(n, dtype=float)
    s = 0
    for i in range(n):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"CTW: {i + 1}/{n}")
        s += data[i]
        tcl[i] = (s * np.log2(1 / p) + ((i + 1) - s) * np.log2(1 / (1 - p)))
        cdata = data[i]
        number_of_trees = min(i + 1, maxL)
        for j in range(number_of_trees):
            CTWarray[j].update(cdata)
            clt = np.zeros(maxD, dtype=float)
            for D in range(1, maxD + 1):
                clt[D - 1] = (CTWarray[j].calc_code_length(D) + logs(D))
            segment_length = maxL - j
            cl[i, j] = (np.min(clt) + logs(segment_length))

        if i + 1 > maxL:
            CTWarray[:-1] = CTWarray[1:]
            CTWarray[-1] = CTWbn(maxD, n)

    for i in range(n):
        matlab_i = i + 1
        if matlab_i < maxL:
            for j in range(matlab_i):
                matlab_j = j + 1
                previous_index = (matlab_i - (matlab_i - matlab_j))
                if previous_index == 0:
                    previous_tcl = 0.0
                else:
                    previous_tcl = (tcl[previous_index - 1])

                typcl = tcl[i] - previous_tcl
                cldiff[i, j] = cl[i, j] - typcl
            cldiffm[i] = np.min(cldiff[i, :matlab_i])

        else:
            for j in range(maxL):
                matlab_j = j + 1
                previous_index = (matlab_i - (maxL - matlab_j))
                if previous_index == 0:
                    previous_tcl = 0.0
                else:
                    previous_tcl = (tcl[previous_index - 1])
                typcl = tcl[i] - previous_tcl
                cldiff[i, j] = cl[i, j] - typcl
            cldiffm[i] = np.min(cldiff[i])

    return cldiffm

print("LOADING ORIGINAL COIN-TOSS DATA")
mat = sio.loadmat(COIN_FILE)
data = mat["data"].squeeze().astype(int)
p = np.sum(data) / len(data)

print("Total number of tosses :", len(data))
print("Number of ones:", int(np.sum(data)))
print("Number of zeros:", int(len(data) - np.sum(data)))
print("p:", p)

print("CONVERTING ORIGINAL TEXT")
wordlen, DataP = text_to_word_len(TEXT_FILE)
print("Number of words:", len(wordlen))
print("Generated binary values:", len(DataP))

print("First 20 word lengths:")
print(wordlen[:20])

print("First 20 text bits:")
print(DataP[:20])
start = 2000
number_of_inserted_bits = len(DataP)
endpoint = start + number_of_inserted_bits
data_exp = data[:4000].copy()
data_exp[start:endpoint] = DataP
print("EXPERIMENTAL SEQUENCE")
print("Total sequence length:", len(data_exp))
print("Inserted text bits:", number_of_inserted_bits)
print("MATLAB-style positions:", start + 1, "to", endpoint)
print("Python indices:", start, "to", endpoint - 1)
datab = data_exp.copy()
datab[datab == 0] = -1
cumdataM = np.cumsum(datab)
maxL = 1000
maxD = 15
print("RUNNING CTW ATYPICALITY PROCEDURE")
print("maxL =", maxL)
print("maxD =", maxD)
ctw_start_time = time.perf_counter()
cldiffm = CTWforKnownTyp(data_exp, p, maxD, maxL)
ctw_end_time = time.perf_counter()
ctw_runtime_seconds = ctw_end_time - ctw_start_time
minimum_position = np.argmin(cldiffm) + 1
minimum_value = np.min(cldiffm)
total_end_time = time.perf_counter()
total_runtime_seconds = total_end_time - total_start_time

def format_runtime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

print("FINAL RESULTS")
print("Minimum cldiffm:", minimum_value)
print("Position of minimum:", minimum_position)
print("Inserted text interval :", f"{start + 1} - {endpoint}")
print("Number inserted:", number_of_inserted_bits)
print("CTW runtime (seconds)  :", f"{ctw_runtime_seconds:.6f}")
print("CTW runtime (hh:mm:ss) :", format_runtime(ctw_runtime_seconds))
print("Total runtime (seconds):", f"{total_runtime_seconds:.6f}")
print("Total runtime:", format_runtime(total_runtime_seconds))
plt.figure(figsize=(15, 6))
plt.plot(cumdataM, linewidth=0.5, label="Random walk")
plt.plot(3 * cldiffm, linewidth=1.0, label="3 × cldiffm")
plt.xlabel("Position")
plt.ylabel("Value")
plt.title("Text inserted into coin tosses")
plt.legend()
plt.show()
plt.figure(figsize=(15, 6))
plt.plot(cumdataM, linewidth=0.5, label="Random walk")
plt.plot(3 * cldiffm, linewidth=1.0, label="3 × cldiffm")

plt.axvline(start, linestyle="--", linewidth=1, label="Text starts")
plt.axvline(endpoint, linestyle="--", linewidth=1, label="Text ends")
plt.xlabel("Position")
plt.ylabel("Value")
plt.title("Reproduction check — text insertion boundaries")
plt.legend()
plt.show()