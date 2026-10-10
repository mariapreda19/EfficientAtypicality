from __future__ import annotations
from project_paths import DATA_FILES, RESULTS_ROOT
from array import array
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import heapq
import itertools
import json
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
INPUT_ROOT = DATA_FILES
OUTPUT_ROOT = RESULTS_ROOT
RANDU_SEED = 5
RANDU_LENGTH = 2000
INSERT_AT = 20000
EPSILON = 0.1
MAX_LENGTH = 5000
MAX_DEPTH = 4
ALPHA = 0.05
P0_ONE = 0.5
RUN_COIN_CONTROL = True
TOP_K = 10

def load_coins(path):
    sheet = pd.read_excel(path, sheet_name='Sheet2', header=None, dtype=str)
    seen = {}
    for ncol, scol, hcol, tcol in ((3, 4, 5, 6), (11, 12, 13, 14)):
        for row in sheet.itertuples(index=False, name=None):
            n, initial = (str(row[ncol]).strip(), str(row[scol]).strip())
            if not n.isdecimal() or initial not in ('H', 'T'):
                continue
            i = int(n)
            if not 1 <= i <= 40000:
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
    x = np.array(ordered, dtype=np.uint8)
    if int(x.sum()) != 20217:
        raise ValueError('Heads do not match paper (20,217)')
    return x

def randu(n, seed):
    if not 0 < seed < 2 ** 31:
        raise ValueError('RANDU seed out of range')
    state, bits = (seed, np.empty(n, dtype=np.uint8))
    for i in range(n):
        state = 65539 * state % 2 ** 31
        bits[i] = int(state.bit_count() >= 16)
    return bits

def scales(N, L, eps, dmax):
    if not 0 < eps <= 0.5 or not 1 <= L <= N:
        raise ValueError('Invalid epsilon/L')
    m, result = (1, [])
    while m <= L:
        h = max(1, math.floor(eps * m + 1e-12))
        j_first = (dmax + h - 1) // h
        j_last = (N - m) // h
        if j_first <= j_last:
            result.append((m, h, j_first, j_last, j_last - j_first + 1))
        m = math.ceil((1 + eps) * m - 1e-12)
    return result

def occurrence_index(x, dmax):
    positions = defaultdict(lambda: array('I'))
    N = len(x)
    for t in range(N):
        bit = int(x[t])
        positions[0, 0, bit].append(t)
        context = 0
        for j in range(1, min(dmax, t) + 1):
            context |= int(x[t - j]) << j - 1
            positions[j, context, bit].append(t)
    return positions
LOG2_PI = math.log2(math.pi)
LOG2 = math.log(2.0)

def ctw_depth_logs_from_counts(counts, dmax):
    pe = []
    for j in range(dmax + 1):
        level = []
        for ctx in range(1 << j):
            a = counts[j][ctx][0]
            b = counts[j][ctx][1]
            value = (math.lgamma(a + 0.5) + math.lgamma(b + 0.5) - math.lgamma(a + b + 1)) / LOG2 - LOG2_PI
            level.append(value)
        pe.append(level)

    roots = []
    for d in range(dmax + 1):
        weighted = pe[d]
        for j in range(d - 1, -1, -1):
            next_level = []
            for ctx in range(1 << j):
                first = pe[j][ctx] - 1
                second = weighted[ctx] + weighted[ctx | 1 << j] - 1
                largest = max(first, second)
                value = largest + math.log2(1.0 + 2.0 ** (-abs(first - second)))
                next_level.append(value)
            weighted = next_level
        roots.append(weighted[0])
    return roots

def indexed_ctw_depth_logs(index, start, end, dmax):
    counts = []
    for j in range(dmax + 1):
        nodes = []
        for ctx in range(1 << j):
            pair = []
            for bit in (0, 1):
                positions = index.get((j, ctx, bit), ())
                pair.append(bisect_left(positions, end) - bisect_left(positions, start))
            nodes.append(pair)
        counts.append(nodes)
    return ctw_depth_logs_from_counts(counts, dmax)

def direct_ctw_depth_logs(x, start, end, dmax):
    counts = []
    for j in range(dmax + 1):
        level = []
        for ctx in range(1 << j):
            level.append([0, 0])
        counts.append(level)
    for t in range(start, end):
        bit = int(x[t])
        counts[0][0][bit] += 1
        ctx = 0
        for j in range(1, dmax + 1):
            ctx |= int(x[t - j]) << j - 1
            counts[j][ctx][bit] += 1
    return ctw_depth_logs_from_counts(counts, dmax)

def check_oracle(x, index, dmax, checks=12):
    rng = np.random.default_rng(20260923)
    starts = [dmax, dmax + 1, min(INSERT_AT, len(x) - 10)]
    random_starts = rng.integers(dmax, len(x) - 20, size=checks)
    for t in random_starts:
        starts.append(int(t))
    worst = 0.0
    for start in starts:
        end = min(len(x), start + int(rng.integers(1, 30)))
        a = indexed_ctw_depth_logs(index, start, end, dmax)
        b = direct_ctw_depth_logs(x, start, end, dmax)
        for u, v in zip(a, b):
            error = abs(u - v)
            if error > worst:
                worst = error
    if worst > 1e-09:
        raise AssertionError(f'Indexed CTW differs from direct: {worst} bits')
    return worst

def check_normalization():
    history = [0, 1, 0, 1]
    for d in range(3):
        for length in (1, 2, 3):
            total = 0.0
            for suffix in itertools.product((0, 1), repeat=length):
                x = np.array(history + list(suffix), dtype=np.uint8)
                total += 2.0 ** direct_ctw_depth_logs(x, len(history), len(x), d)[d]
            if not math.isclose(total, 1.0, abs_tol=1e-12):
                raise AssertionError(f'CTW is not normalized: d={d}, m={length}, total={total}')

def overlap_iou(start, end, truth):
    if truth is None:
        return None
    a, b = truth
    intersection = max(0, min(end, b) - max(start, a))
    return intersection / (end - start + (b - a) - intersection)

def run_uad(x, name, truth=None):
    N = len(x)
    grid = scales(N, MAX_LENGTH, EPSILON, MAX_DEPTH)
    M = 0
    for item in grid:
        M += item[-1]
    if not M:
        raise ValueError('No candidates')
    total_t0 = time.perf_counter()
    index = occurrence_index(x, MAX_DEPTH)
    index_seconds = time.perf_counter() - total_t0
    oracle_error = check_oracle(x, index, MAX_DEPTH)
    prior = -np.log2([1 - P0_ONE, P0_ONE])
    prefix = np.r_[0.0, np.cumsum(prior[x], dtype=np.float64)]
    log_depth = math.log2(MAX_DEPTH + 1)
    log_M = math.log2(M)
    threshold = math.log2(1 / ALPHA)
    heap = []
    best_uniform = None
    best_scale = None
    best_contained = None
    best_start_score = np.full(N, np.nan)
    log_e_total = -math.inf
    scanned = 0
    scan_t0 = time.perf_counter()
    for k, (m, h, j_first, j_last, Nk) in enumerate(grid):
        scale_penalty = math.log2(Nk) + 2 * math.log2(k + 1) + math.log2(math.pi ** 2 / 6)
        for j in range(j_first, j_last + 1):
            start = j * h
            end = start + m
            log_q = indexed_ctw_depth_logs(index, start, end, MAX_DEPTH)
            d = max(range(MAX_DEPTH + 1), key=lambda z: log_q[z] - log_depth)
            raw = float(prefix[end] - prefix[start] + log_q[d])
            adjusted = raw - log_depth
            au = adjusted - log_M
            asc = adjusted - scale_penalty
            rec = {'start_1based': start + 1, 'end_1based': end, 'length': m, 'depth': d, 'raw_bits': raw, 'A_uniform_bits': au, 'A_scale_bits': asc, 'iou': overlap_iou(start, end, truth)}
            if best_uniform is None or au > best_uniform['A_uniform_bits']:
                best_uniform = rec
            if best_scale is None or asc > best_scale['A_scale_bits']:
                best_scale = rec
            if truth is not None and truth[0] <= start and (end <= truth[1]):
                if best_contained is None or au > best_contained['A_uniform_bits']:
                    best_contained = rec
            if math.isnan(best_start_score[start]) or au > best_start_score[start]:
                best_start_score[start] = au
            key = (au, scanned, rec)
            if len(heap) < TOP_K:
                heapq.heappush(heap, key)
            elif au > heap[0][0]:
                heapq.heapreplace(heap, key)
            log_e_total = np.logaddexp2(log_e_total, au)
            scanned += 1
        if k % 12 == 0 or k == len(grid) - 1:
            print(f'{name}: {k + 1}/{len(grid)} scales, {scanned:,}/{M:,} candidates', flush=True)
    scan_seconds = time.perf_counter() - scan_t0
    sum_scale_weights = 0.0
    for k in range(len(grid)):
        sum_scale_weights += 6 / (math.pi ** 2 * (k + 1) ** 2)

    top_uniform = []
    sorted_heap = sorted(heap, reverse=True)
    for item in sorted_heap:
        top_uniform.append(item[2])

    result = {'name': name, 'n': N, 'epsilon': EPSILON, 'rho': (1 + EPSILON) ** (-2), 'max_length': MAX_LENGTH,
              'max_depth': MAX_DEPTH, 'scales': len(grid), 'candidates': M, 'depth_candidates': M * (MAX_DEPTH + 1),
              'candidate_weight_mode': 'uniform over intervals × uniform over depths', 'sum_uniform_weights': 1.0,
              'sum_scale_weights': sum_scale_weights, 'alpha': ALPHA, 'threshold_bits': threshold, 'index_seconds': index_seconds, 'scan_seconds': scan_seconds, 'total_uad_seconds': time.perf_counter() - total_t0, 'max_index_direct_error_bits': oracle_error, 'best_uniform': best_uniform, 'best_scale_weighted': best_scale, 'best_contained_in_insert': best_contained, 'detected_uniform': best_uniform['A_uniform_bits'] >= threshold, 'detected_scale': best_scale['A_scale_bits'] >= threshold, 'log2_total_e_value_uniform': float(log_e_total), 'top_uniform': top_uniform, 'best_start_scores_uniform': best_start_score}
    if truth is not None:
        a, b = truth
        guarantee = (b - a) / (1 + EPSILON) ** 2 - 1
        contained_lengths = []
        for m, h, j0, j1, n in grid:
            left = max(j0, (a + h - 1) // h)
            right = min(j1, (b - m) // h)
            if left <= right:
                contained_lengths.append(m)
        result['guarantee_check'] = {'minimum_bound': guarantee, 'largest_contained_candidate': max(contained_lengths,
                                default=0), 'holds': bool(contained_lengths and max(contained_lengths) >= guarantee)}
    return result

start_all = time.perf_counter()
check_normalization()
excel_hits = sorted(INPUT_ROOT.rglob('40000tosses.xlsx'))
if len(excel_hits) != 1:
    raise FileNotFoundError(f'Expected one 40000tosses.xlsx; found {len(excel_hits)}: {excel_hits[:5]}')
coins = load_coins(excel_hits[0])
insert = randu(RANDU_LENGTH, RANDU_SEED)
mixed = np.concatenate((coins[:INSERT_AT], insert, coins[INSERT_AT:]))
truth = (INSERT_AT, INSERT_AT + RANDU_LENGTH)
assert len(coins) == 40000 and len(mixed) == 40000 + RANDU_LENGTH
print(f'Coins: {len(coins):,}, heads: {coins.sum():,}; RANDU: {RANDU_LENGTH:,}, ones fraction: {insert.mean():.4f};'
      f' mixed: {len(mixed):,}', flush=True)
print(f'Inserted RANDU at 1-based positions {truth[0] + 1:,}–{truth[1]:,}', flush=True)
mixed_result = run_uad(mixed, 'mixed', truth)
control_result = run_uad(coins, 'coin_control') if RUN_COIN_CONTROL else None
mixed_json = {}
for key, value in mixed_result.items():
    if key != 'best_start_scores_uniform':
        mixed_json[key] = value

control_json = None
if control_result is not None:
    control_json = {}
    for key, value in control_result.items():
        if key != 'best_start_scores_uniform':
            control_json[key] = value

summary = {'setup': {'coins': len(coins), 'coin_heads': int(coins.sum()), 'mixed_bits': len(mixed),
                     'randu_seed': RANDU_SEED, 'randu_ones_fraction': float(insert.mean()), 'insertion_1based': [truth[0] + 1, truth[1]],
                     'frozen_p0_one': P0_ONE, 'depth_penalty_bits': math.log2(MAX_DEPTH + 1),
                     'note': ''}, 'mixed': mixed_json, 'coin_control': control_json, 'exhaustive_reference':
    {'mixed_intervals': 97921200, 'mixed_scan_seconds_on_prior_host': 315.234, 'control_intervals': 93121200,
     'control_scan_seconds_on_prior_host': 291.783}, 'total_cell_seconds': None}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(mixed_result['top_uniform']).to_csv(OUTPUT_ROOT / 'uad_top10.csv', index=False)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
walk = np.cumsum(2 * mixed.astype(np.int16) - 1)
ix = np.arange(1, len(mixed) + 1)
ax1.plot(ix, walk, lw=0.8)
ax1.set_ylabel('Cumulative ±1 walk')
ax2.plot(ix, mixed_result['best_start_scores_uniform'], lw=0.7, color='tab:red')
ax2.axhline(math.log2(1 / ALPHA), color='black', ls='--', lw=0.8, label='global alpha threshold')
ax2.set(xlabel='Bit position (1-based)', ylabel='Best penalized A per start')
for ax in (ax1, ax2):
    ax.axvspan(truth[0] + 1, truth[1], color='orange', alpha=0.16)
ax2.legend()
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / 'uad_coin_randu_full.png', dpi=160)
plt.show()
summary['total_cell_seconds'] = time.perf_counter() - start_all
(OUTPUT_ROOT / 'uad_coin_randu_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'mixed_candidates': mixed_result['candidates'],
                  'control_candidates': control_result['candidates']
                  if control_result else None, 'mixed_index_seconds': round(mixed_result['index_seconds'], 3),
                  'mixed_scan_seconds': round(mixed_result['scan_seconds'], 3), 'control_scan_seconds': round(control_result['scan_seconds'], 3)
                  if control_result else None, 'mixed_best': mixed_result['best_uniform'], 'control_best': control_result['best_uniform']
                  if control_result else None, 'alpha': ALPHA, 'threshold_bits': mixed_result['threshold_bits'],
                  'detected_mixed': mixed_result['detected_uniform'], 'detected_control': control_result['detected_uniform']
                  if control_result else None, 'largest_contained_vs_bound': mixed_result['guarantee_check'],
                  'max_index_direct_error_bits': mixed_result['max_index_direct_error_bits'], 'total_cell_seconds':
                      round(summary['total_cell_seconds'], 3)}, indent=2))