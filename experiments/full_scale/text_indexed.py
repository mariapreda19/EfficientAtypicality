from __future__ import annotations
from project_paths import DATA_FILES, RESULTS_ROOT
from array import array
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import heapq
import hashlib
import itertools
import json
import math
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
INPUT_ROOT = DATA_FILES
OUTPUT_ROOT = RESULTS_ROOT
BOOK_FILE = None
BOOK_LENGTH = None
TIE_SEED = 1
INSERT_AT = 20500
EPSILON = 0.05
MAX_LENGTH = 7000
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

def find_book_file():
    if BOOK_FILE is not None:
        path = Path(BOOK_FILE)
        if not path.is_file():
            raise FileNotFoundError(f'BOOK_FILE is not a file: {path}')
        return path

    key = 'on_the_shores_of_the_great_sea'
    files = []

    for path in INPUT_ROOT.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ('', '.txt', '.text', '.md'):
            continue

        normalized = str(path).lower().replace('-', '_').replace(' ', '_')
        if key in normalized:
            files.append(path)

    exact = []
    for path in files:
        normalized = path.stem.lower().replace('-', '_').replace(' ', '_')
        if key in normalized:
            exact.append(path)

    if len(exact) == 1:
        return exact[0]

    if len(files) == 1:
        return files[0]

    local = Path('upload/Text lipit (2)(6).txt')
    if not files and local.is_file() and not Path('/kaggle/input').exists():
        return local

    raise FileNotFoundError(f'Expected one book text under {INPUT_ROOT}; found {files[:10]}')


def make_book_insert(source, n, tie_seed):
    content = source.read_text(encoding='utf-8-sig')
    content = re.sub('(?im)^Original Copyright[^\\n]*Heritage History[^\\n]*$', '', content)
    words = re.findall("[A-Za-z]+(?:['’][A-Za-z]+)*", content)
    if n is None:
        n = len(words) - 1
    if n < 1 or len(words) < n + 1:
        raise ValueError(f'Need {n + 1} words for {n} comparisons; got {len(words)}')
    length_values = []
    for word in words[:n + 1]:
        length = 0
        for character in word:
            if character.isalpha():
                length += 1
        length_values.append(length)
    lengths = np.array(length_values)
    changes = np.sign(np.diff(lengths))
    bits = (changes > 0).astype(np.uint8)
    ties = np.flatnonzero(changes == 0)
    bits[ties] = np.random.default_rng(tie_seed).integers(0, 2, size=len(ties))
    return (bits, len(words), len(ties))

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
                value = max(first, second) + math.log2(1.0 + 2.0 ** (-abs(first - second)))
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

def run_uad(x, name):
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
            d = 0
            best_depth_value = log_q[0] - log_depth
            for depth in range(1, MAX_DEPTH + 1):
                depth_value = log_q[depth] - log_depth
                if depth_value > best_depth_value:
                    best_depth_value = depth_value
                    d = depth
            raw = float(prefix[end] - prefix[start] + log_q[d])
            adjusted = raw - log_depth
            au = adjusted - log_M
            asc = adjusted - scale_penalty
            rec = {'start_1based': start + 1, 'end_1based': end, 'length': m, 'depth': d, 'raw_bits': raw, 'A_uniform_bits': au, 'A_scale_bits': asc}
            if best_uniform is None or au > best_uniform['A_uniform_bits']:
                best_uniform = rec
            if best_scale is None or asc > best_scale['A_scale_bits']:
                best_scale = rec
            if math.isnan(best_start_score[start]) or au > best_start_score[start]:
                best_start_score[start] = au
            key = (au, scanned, rec)
            if len(heap) < TOP_K:
                heapq.heappush(heap, key)
            else:
                if au > heap[0][0]:
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

    result = {'name': name, 'n': N, 'epsilon': EPSILON, 'rho': (1 + EPSILON) ** (-2), 'max_length': MAX_LENGTH, 'max_depth': MAX_DEPTH, 'scales': len(grid), 'candidates': M, 'depth_candidates': M * (MAX_DEPTH + 1), 'candidate_weight_mode': 'uniform over intervals × uniform over depths', 'sum_uniform_weights': 1.0, 'sum_scale_weights': sum_scale_weights, 'alpha': ALPHA, 'threshold_bits': threshold, 'index_seconds': index_seconds, 'scan_seconds': scan_seconds, 'total_uad_seconds': time.perf_counter() - total_t0, 'max_index_direct_error_bits': oracle_error, 'best_uniform': best_uniform, 'best_scale_weighted': best_scale, 'detected_uniform': best_uniform['A_uniform_bits'] >= threshold, 'detected_scale': best_scale['A_scale_bits'] >= threshold, 'log2_total_e_value_uniform': float(log_e_total), 'top_uniform': top_uniform, 'best_start_scores_uniform': best_start_score}
    return result

def evaluate_after_scan(result, truth=None):
    selected = [result['best_uniform'], result['best_scale_weighted']]
    for item in result['top_uniform']:
        selected.append(item)

    for rec in selected:
        if truth is not None:
            rec['iou'] = overlap_iou(rec['start_1based'] - 1, rec['end_1based'], truth)
        else:
            rec['iou'] = None

    if truth is not None:
        a, b = truth
        grid = scales(result['n'], MAX_LENGTH, EPSILON, MAX_DEPTH)
        guarantee = (b - a) / (1 + EPSILON) ** 2 - 1
        contained_lengths = []

        for m, h, j0, j1, n in grid:
            left = max(j0, (a + h - 1) // h)
            right = min(j1, (b - m) // h)
            if left <= right:
                contained_lengths.append(m)

        if contained_lengths:
            largest_contained = max(contained_lengths)
        else:
            largest_contained = 0

        holds = bool(contained_lengths and largest_contained >= guarantee)
        result['guarantee_check'] = {
            'minimum_bound': guarantee,
            'largest_contained_candidate': largest_contained,
            'holds': holds
        }

    result['evaluation_note'] = 'IoU and geometric guarantee use the known insertion only after the blind scan.'

start_all = time.perf_counter()
check_normalization()
excel_hits = sorted(INPUT_ROOT.rglob('40000tosses.xlsx'))
if len(excel_hits) != 1:
    raise FileNotFoundError(f'Expected one 40000tosses.xlsx; found {len(excel_hits)}: {excel_hits[:5]}')
coins = load_coins(excel_hits[0])
book_path = find_book_file()
book_source_sha256 = hashlib.sha256(book_path.read_bytes()).hexdigest()
insert, n_book_words, n_ties = make_book_insert(book_path, BOOK_LENGTH, TIE_SEED)
book_bit_sha256 = hashlib.sha256(insert.tobytes()).hexdigest()
mixed = np.concatenate((coins[:INSERT_AT], insert, coins[INSERT_AT:]))
truth = (INSERT_AT, INSERT_AT + len(insert))
assert len(coins) == 40000 and len(mixed) == 40000 + len(insert)
print(f'Coins: {len(coins):,}, heads: {coins.sum():,}; book bits: {len(insert):,}, ones fraction: {insert.mean():.4f}; mixed: {len(mixed):,}', flush=True)
print(f'Book: {book_path}; words: {n_book_words}; equal-length pairs: {n_ties}; source SHA256: {book_source_sha256}; bit SHA256: {book_bit_sha256}', flush=True)
print(f'Inserted book bits at 1-based positions {truth[0] + 1:,}–{truth[1]:,}', flush=True)
mixed_result = run_uad(mixed, 'mixed')
if RUN_COIN_CONTROL:
    control_result = run_uad(coins, 'coin_control')
else:
    control_result = None
evaluate_after_scan(mixed_result, truth)
if control_result is not None:
    evaluate_after_scan(control_result)
mixed_json = {}
for key, value in mixed_result.items():
    if key != 'best_start_scores_uniform':
        mixed_json[key] = value

if control_result is not None:
    control_json = {}
    for key, value in control_result.items():
        if key != 'best_start_scores_uniform':
            control_json[key] = value
else:
    control_json = None

summary = {'setup': {'coins': len(coins), 'coin_heads': int(coins.sum()), 'mixed_bits': len(mixed), 'book_bits': len(insert), 'book_file': str(book_path), 'book_source_sha256': book_source_sha256, 'book_words_available': n_book_words, 'book_tie_seed': TIE_SEED, 'book_equal_length_pairs': n_ties, 'book_bit_sha256': book_bit_sha256, 'book_ones_fraction': float(insert.mean()), 'insertion_1based': [truth[0] + 1, truth[1]], 'frozen_p0_one': P0_ONE, 'depth_penalty_bits': math.log2(MAX_DEPTH + 1), 'note': 'All available words after removing matching Heritage History page footers: next longer→1, shorter→0, equal→reproducible fair bit. Apostrophes are excluded from word length. Exact excerpt, tokenizer, insertion and tie seed are not published in the paper. Text is not assumed i.i.d. for the power theorem.'}, 'mixed': mixed_json, 'coin_control': control_json, 'total_cell_seconds': None}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(mixed_result['top_uniform']).to_csv(OUTPUT_ROOT / 'uad_book_top10.csv', index=False)
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
fig.savefig(OUTPUT_ROOT / 'uad_coin_book_full.png', dpi=160)
plt.show()
summary['total_cell_seconds'] = time.perf_counter() - start_all
(OUTPUT_ROOT / 'uad_coin_book_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
if control_result is not None:
    control_candidates = control_result['candidates']
    control_scan_seconds = round(control_result['scan_seconds'], 3)
    control_best = control_result['best_uniform']
    detected_control = control_result['detected_uniform']
else:
    control_candidates = None
    control_scan_seconds = None
    control_best = None
    detected_control = None

print(json.dumps({'mixed_candidates': mixed_result['candidates'], 'control_candidates': control_candidates, 'mixed_index_seconds': round(mixed_result['index_seconds'], 3), 'mixed_scan_seconds': round(mixed_result['scan_seconds'], 3), 'control_scan_seconds': control_scan_seconds, 'mixed_best': mixed_result['best_uniform'], 'control_best': control_best, 'alpha': ALPHA, 'threshold_bits': mixed_result['threshold_bits'], 'detected_mixed': mixed_result['detected_uniform'], 'detected_control': detected_control, 'largest_contained_vs_bound': mixed_result['guarantee_check'], 'max_index_direct_error_bits': mixed_result['max_index_direct_error_bits'], 'total_cell_seconds': round(summary['total_cell_seconds'], 3)}, indent=2))