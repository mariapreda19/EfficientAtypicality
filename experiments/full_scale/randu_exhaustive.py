from project_paths import DATA_FILES, RESULTS_ROOT
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
ROOT = DATA_FILES
OUT = RESULTS_ROOT
MODE = 'randu'
SOURCE_FILE = None
START_1_BASED = 20001
INSERT_LENGTH = 2000
RANDU_SEED = 5
TIE_SEED = 1
PLOT_FIRST = 1
PLOT_LAST = 40000 + INSERT_LENGTH
MAX_LENGTH = 5000
MAX_DEPTH = 4
HEADER_BITS = 0.0
RUN_COIN_CONTROL = True

def coin_bits(path):
    a = pd.read_excel(path, sheet_name='Sheet2', header=None, dtype=str)
    entries = {}
    for number_col, start_col, head_col, tail_col in [(3, 4, 5, 6), (11, 12, 13, 14)]:
        for row in a.itertuples(index=False, name=None):
            number, start = (str(row[number_col]).strip(), str(row[start_col]).strip())
            if not number.isdecimal() or start not in {'H', 'T'}:
                continue
            i = int(number)
            if i < 1 or i > 40000:
                continue
            h = str(row[head_col]).strip() == '1'
            t = str(row[tail_col]).strip() == '1'
            if h == t or i in entries:
                raise ValueError(f'Bad/duplicate toss {i}')
            entries[i] = int(h)
    if len(entries) != 40000 or set(entries) != set(range(1, 40001)):
        raise ValueError('Missing toss numbers')
    ordered = []
    for i in range(1, 40001):
        ordered.append(entries[i])
    x = np.array(ordered, dtype=np.int8)
    assert int(x.sum()) == 20217, f'Not the paper dataset: {x.sum()} heads'
    return x

def make_insert(mode, source, n, randu_seed, tie_seed):
    rng = np.random.default_rng(tie_seed)
    if mode == 'randu':
        if not 0 < randu_seed < 2 ** 31:
            raise ValueError('RANDU_SEED out of range')
        state, bits = (randu_seed, [])
        for _ in range(n):
            state = 65539 * state % 2 ** 31
            bits.append(int(state.bit_count() >= 16))
        return np.array(bits, dtype=np.int8)
    if source is None:
        raise ValueError(f'SOURCE_FILE required for MODE={mode}')
    source = Path(source)
    if mode == 'bits':
        chars = ''.join(source.read_text(encoding='utf8').split())
        if len(chars) < n or any((c not in '01' for c in chars)):
            raise ValueError('Need at least INSERT_LENGTH binary symbols in SOURCE_FILE')
        bits = []
        for c in chars[:n]:
            bits.append(int(c))
        return np.array(bits, dtype=np.int8)
    if mode == 'text':
        words = re.findall("[A-Za-z]+(?:['’][A-Za-z]+)?", source.read_text(encoding='utf8'))
        if len(words) < n + 1:
            raise ValueError('Need at least INSERT_LENGTH+1 words')
        values = []
        for word in words[:n + 1]:
            values.append(len(word))
    elif mode == 'hrv':
        values = pd.read_csv(source, header=None).iloc[:, 0].dropna().to_numpy(dtype=float)
        if len(values) < n + 1:
            raise ValueError('Need at least INSERT_LENGTH+1 intervals')
        values = values[:n + 1]
    else:
        raise ValueError('MODE must be randu/text/hrv/bits')
    left, right = (np.asarray(values[:-1]), np.asarray(values[1:]))
    return np.where(right > left, 1, np.where(right < left, 0, rng.integers(0, 2, size=n))).astype(np.int8)
LN2 = np.log(2.0)

def log_star(n):
    acc, value = (0.0, float(n))
    while value > 1:
        value = np.log2(value)
        if value > 0:
            acc += value
    return acc

def prefix_context_counts(x, depth):
    N = len(x)
    result = []
    context_at = []
    for j in range(depth + 1):
        contexts = np.zeros(N, dtype=np.int32)
        for k in range(j):
            contexts[j:] |= x[j - 1 - k:N - 1 - k].astype(np.int32) << k
        context_at.append(contexts)
        rows = (contexts[j:] * 2 + x[j:]).astype(np.int32)
        C = np.zeros((1 << j, 2, N + 1), dtype=np.int32)
        for key in range(2 * (1 << j)):
            C[key // 2, key % 2, j + 1:] = np.cumsum(rows == key, dtype=np.int32)
        result.append(C)
    return (result, context_at)

def ctw_log_prob(index, context_at, starts, ends, depth):
    node_probs = []
    for j in range(depth + 1):
        count = index[j][:, :, ends] - index[j][:, :, starts + j]
        a, b = (count[:, 0, :], count[:, 1, :])
        log_kt = (gammaln(a + 0.5) + gammaln(b + 0.5) - gammaln(a + b + 1) - 2 * gammaln(0.5)) / LN2
        node_probs.append(log_kt)
    weighted = node_probs[-1]
    for j in range(depth - 1, -1, -1):
        child = weighted[:1 << j] + weighted[1 << j:]
        prefix_context = context_at[j][starts + j]
        child[prefix_context, np.arange(len(starts))] -= 1.0
        weighted = np.logaddexp((node_probs[j] - 1) * LN2, (child - 1) * LN2) / LN2
    return weighted[0]

def scan_all_lengths(x, L, D, truth_start, truth_end):
    index, context_at = prefix_context_counts(x, D)
    N = len(x)
    best = np.full(N, np.inf)
    best_len = np.zeros(N, dtype=np.int32)
    best_depth = np.zeros(N, dtype=np.int8)
    best_matched = {'score': float('inf'), 'start': None, 'end': None, 'depth': None, 'iou': None}
    for length in range(1, L + 1):
        starts = np.arange(N - length + 1)
        ends = starts + length
        local = np.full(len(starts), np.inf)
        depths = np.zeros(len(starts), dtype=np.int8)
        for depth in range(min(D, length - 1) + 1):
            log_q = ctw_log_prob(index, context_at, starts, ends, depth)
            score = -log_q + log_star(depth + 1) + log_star(length) + HEADER_BITS - length
            chosen = score < local
            local[chosen], depths[chosen] = (score[chosen], depth)
        overlap = np.maximum(0, np.minimum(ends, truth_end) - np.maximum(starts, truth_start))
        iou = overlap / (length + (truth_end - truth_start) - overlap)
        qualifying = iou >= 0.5
        if np.any(qualifying):
            indices = np.flatnonzero(qualifying)
            k = indices[np.argmin(local[indices])]
            if local[k] < best_matched['score']:
                best_matched = {'score': float(local[k]), 'start': int(starts[k]), 'end': int(ends[k]), 'depth': int(depths[k]), 'iou': float(iou[k])}
        chosen = local < best[:len(starts)]
        best[:len(starts)][chosen] = local[chosen]
        best_len[:len(starts)][chosen] = length
        best_depth[:len(starts)][chosen] = depths[chosen]
        if length % 100 == 0:
            print(f'scanned lengths 1..{length} / {L}')
    return (best, best_len, best_depth, best_matched)
cell_t0 = time.perf_counter()
matches = sorted(ROOT.rglob('40000tosses.xlsx'))
if len(matches) != 1:
    raise FileNotFoundError(f'Expected one 40000tosses.xlsx under {ROOT}; found {len(matches)}: {matches[:8]}')
coins = coin_bits(matches[0])
print('Exact coin-data check:', len(coins), 'tosses;', int(coins.sum()), 'heads;', f'p̂={coins.mean():.6f}')
insertion = make_insert(MODE, SOURCE_FILE, INSERT_LENGTH, RANDU_SEED, TIE_SEED)
start = START_1_BASED - 1
assert 0 <= start <= len(coins)
mixed = np.concatenate((coins[:start], insertion, coins[start:]))
assert len(mixed) == len(coins) + len(insertion)
assert 1 <= PLOT_FIRST <= PLOT_LAST <= len(mixed)
segment = mixed[PLOT_FIRST - 1:PLOT_LAST]
truth_start = START_1_BASED - PLOT_FIRST
truth_end = truth_start + len(insertion)
assert 0 <= truth_start < truth_end <= len(segment), 'Insert must fit scan window'
example = np.array([0, 1, 1, 0, 1, 0, 0], dtype=np.int8)
example_idx, example_ctx = prefix_context_counts(example, 2)
example_logq = ctw_log_prob(example_idx, example_ctx, np.array([0]), np.array([7]), 2)[0]
assert np.isclose(example_logq, np.log2(9 / 4096), atol=1e-12), ('CTW initialization disagrees with the 0110100 example in Willems (1998)', example_logq, np.log2(9 / 4096))
print('CTW startup check: P(0110100 | D=2) = 9/4096 [Willems 1998]')
t0 = time.perf_counter()
delta, best_len, best_depth, matched = scan_all_lengths(segment, MAX_LENGTH, MAX_DEPTH, truth_start, truth_end)
scan_seconds = time.perf_counter() - t0
walk = np.cumsum(2 * mixed - 1)
xx = np.arange(PLOT_FIRST, PLOT_LAST + 1)
figure, (walk_ax, score_ax) = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
walk_ax.plot(xx, walk[PLOT_FIRST - 1:PLOT_LAST], lw=0.9, label='Random walk')
score_ax.plot(xx, delta, lw=0.9, color='tab:red', label='ΔL(n), CTW')
score_ax.axhline(0, color='0.5', lw=0.7)
for ax in (walk_ax, score_ax):
    ax.axvspan(START_1_BASED, START_1_BASED + len(insertion) - 1, color='orange', alpha=0.13, label='Inserted region')
    ax.legend(loc='best')
walk_ax.set(ylabel='Cumulative bits', title=f'Coin tosses + {MODE}: exhaustive interval scan')
score_ax.set(xlabel='Sample (toss number)', ylabel='ΔL (bits)')
figure.tight_layout()
figure.savefig(OUT / f'coin_{MODE}_reconstruction.png', dpi=180)
plt.show()
pd.DataFrame({'sample_1based': xx, 'coin_or_insert_bit': segment, 'random_walk': walk[PLOT_FIRST - 1:PLOT_LAST], 'delta_L': delta, 'best_length': best_len, 'best_depth': best_depth}).to_csv(OUT / f'coin_{MODE}_reconstruction.csv', index=False)
finite = np.flatnonzero(np.isfinite(delta))
winner = finite[np.argmin(delta[finite])]
print(f'Minimum ΔL={delta[winner]:.3f} bits at sample {xx[winner]}, length {best_len[winner]}, depth {best_depth[winner]}')
left = max(START_1_BASED, PLOT_FIRST)
right = min(START_1_BASED + len(insertion) - 1, PLOT_LAST)
positive = np.isfinite(delta) & (delta < -1e-10)
outside = (xx < left) | (xx > right)
inside = ~outside
selected_start = int(xx[winner])
selected_end = selected_start + int(best_len[winner]) - 1
inter = max(0, min(selected_end, right) - max(selected_start, left) + 1)
union = selected_end - selected_start + 1 + max(0, right - left + 1) - inter
mixed_interval_count = 0
for length in range(1, MAX_LENGTH + 1):
    mixed_interval_count += len(segment) - length + 1

threshold_start_fractions = {}
for tau in (0, 0.5, 1, 2, 4):
    inside_fraction = None
    outside_fraction = None
    if np.any(inside):
        inside_fraction = float(np.mean(delta[inside] < -max(tau, 1e-10)))
    if np.any(outside):
        outside_fraction = float(np.mean(delta[outside] < -max(tau, 1e-10)))
    threshold_start_fractions[str(tau)] = {
        'inside': inside_fraction,
        'outside': outside_fraction
    }

metrics = {'n_total_coin_tosses': len(coins), 'heads': int(coins.sum()), 'n_total_mixed_bits': len(mixed), 'insertion_operation': 'splice: no original tosses were replaced or removed', 'randu_seed': RANDU_SEED, 'randu_one_fraction': float(np.mean(insertion)), 'insert_start_1based': START_1_BASED, 'insert_end_1based': START_1_BASED + len(insertion) - 1, 'seed_selection_note': 'Seed 5 was chosen after seeing that seed 1 gave no detection; the authors did not publish their seed.', 'n_scanned_samples': len(segment), 'n_intervals_tested': mixed_interval_count, 'max_depth': MAX_DEPTH, 'max_length': MAX_LENGTH, 'scan_seconds': round(scan_seconds, 3), 'intervals_per_second': round(mixed_interval_count / scan_seconds, 1), 'minimum_delta_bits': float(delta[winner]), 'best_start_1based': selected_start, 'best_end_1based': selected_end, 'best_depth': int(best_depth[winner]), 'best_interval_iou_with_insert': float(inter / union) if union else None, 'best_candidate_with_iou_at_least_0_5': {'score_bits': matched['score'], 'start_1based': matched['start'] + PLOT_FIRST if matched['start'] is not None else None, 'end_1based': matched['end'] + PLOT_FIRST - 1 if matched['end'] is not None else None, 'depth': matched['depth'], 'iou': matched['iou']}, 'fraction_start_positions_delta_below_zero_outside_insert': float(np.mean(positive[outside])) if np.any(outside) else None, 'fraction_start_positions_delta_below_zero_inside_insert': float(np.mean(positive[inside])) if np.any(inside) else None, 'threshold_start_fractions': threshold_start_fractions, 'note': 'Start-position fractions are not false-alarm rates: candidate intervals can cross the insertion boundary and tau=0 is uncalibrated.'}
import json
if RUN_COIN_CONTROL:
    baseline = coins
    base_t0 = time.perf_counter()
    base_delta, _, _, _ = scan_all_lengths(baseline, MAX_LENGTH, MAX_DEPTH, truth_start, truth_end)
    base_seconds = time.perf_counter() - base_t0
    baseline_interval_count = 0
    for length in range(1, MAX_LENGTH + 1):
        baseline_interval_count += len(baseline) - length + 1

    baseline_thresholds = {}
    for tau in (0, 0.5, 1, 2, 4):
        baseline_thresholds[str(tau)] = float(np.mean(base_delta < -max(tau, 1e-10)))

    metrics['coin_only_control'] = {'n_intervals_tested': baseline_interval_count, 'scan_seconds': round(base_seconds, 3), 'minimum_delta_bits': float(np.min(base_delta)), 'fraction_start_positions_below_threshold': baseline_thresholds}
    metrics['combined_scan_seconds'] = round(scan_seconds + base_seconds, 3)
metrics['total_cell_seconds'] = round(time.perf_counter() - cell_t0, 3)
(OUT / f'coin_{MODE}_metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
