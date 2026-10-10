from __future__ import annotations
from project_paths import DATA_FILES, RESULTS_ROOT
from pathlib import Path
import hashlib
import json
import math
import re
import time
import numpy as np
import pandas as pd
INPUT_ROOT = DATA_FILES
OUTPUT_ROOT = RESULTS_ROOT
BOOK_FILE = None
BOOK_LENGTH = None
TIE_SEED = 1
EXTRA_NORMAL_BITS = 40000
NORMAL_SEED = 20260923
ANOMALY_LENGTH = 1000
INSERT_AFTER_NORMAL = tuple(range(7000, 70001, 9000))
MAX_LENGTH = 7000
MAX_DEPTH = 4
CHECKPOINT_EVERY_LENGTHS = 1
MAX_RUNTIME_HOURS = None
CHECKPOINT_FILE = OUTPUT_ROOT / 'eight_anomalies_exhaustive_checkpoint.npz'
SUMMARY_FILE = OUTPUT_ROOT / 'eight_anomalies_exhaustive_summary.json'

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
            if not 1 <= i <= 40000:
                continue
            head = str(row[hcol]).strip() == '1'
            tail = str(row[tcol]).strip() == '1'
            if head == tail or i in seen:
                raise ValueError(f'Invalid/duplicate toss {i}')
            seen[i] = int(head)
    if len(seen) != 40000:
        raise ValueError('Missing coin toss indices')
    if set(seen) != set(range(1, 40001)):
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
    text_lipit = []
    for path in INPUT_ROOT.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ('.txt', '.text'):
            continue
        if 'text lipit' in path.name.lower():
            text_lipit.append(path)
    if len(text_lipit) == 1:
        return text_lipit[0]
    raise FileNotFoundError(f'Expected one book text under {INPUT_ROOT}; found {files[:10]} / {text_lipit[:10]}')

def make_book_insert(source, n, tie_seed):
    content = source.read_text(encoding='utf-8-sig')
    content = re.sub('(?im)^Original Copyright[^\\n]*Heritage History[^\\n]*$', '', content)
    words = re.findall("[A-Za-z]+(?:['’][A-Za-z]+)*", content)
    if n is None:
        n = len(words) - 1
    if n < 1 or len(words) < n + 1:
        raise ValueError(f'Need {n + 1} words for {n} comparisons; got {len(words)}')
    lengths_list = []
    for word in words[:n + 1]:
        length = 0
        for character in word:
            if character.isalpha():
                length += 1
        lengths_list.append(length)
    lengths = np.array(lengths_list)
    changes = np.sign(np.diff(lengths))
    bits = (changes > 0).astype(np.uint8)
    ties = np.flatnonzero(changes == 0)
    bits[ties] = np.random.default_rng(tie_seed).integers(0, 2, size=len(ties))
    return (bits, len(words), len(ties))

def splice_sections(normal, book_bits):
    positions = INSERT_AFTER_NORMAL
    if not (len(positions) >= 2 and tuple(sorted(positions)) == tuple(positions) and (len(set(positions)) == len(positions)) and (0 <= positions[0]) and (positions[-1] <= len(normal))):
        raise ValueError('INSERT_AFTER_NORMAL must be distinct ascending normal positions')
    if len(book_bits) < ANOMALY_LENGTH:
        raise ValueError('Need at least 1,000 book-derived bits')
    source_offsets = []
    for i in range(len(positions)):
        offset = math.floor(i * len(book_bits) / len(positions))
        source_offsets.append(offset)
    sections = []
    for offset in source_offsets:
        indices = (np.arange(ANOMALY_LENGTH) + offset) % len(book_bits)
        section = book_bits[indices]
        sections.append(section)
    pieces = []
    truth = []
    cursor = 0
    added = 0
    for i, (pos, section, offset) in enumerate(zip(positions, sections, source_offsets), start=1):
        pieces.extend((normal[cursor:pos], section))
        a = pos + added
        b = a + len(section)
        truth.append({'id': i, 'start_1based': a + 1, 'end_1based': b, 'length': len(section), 'source_offset_0based': offset, 'source_wraps': offset + len(section) > len(book_bits)})
        cursor = pos
        added += len(section)
    pieces.append(normal[cursor:])
    return (np.concatenate(pieces), truth)
LOG2 = math.log(2.0)
LOG2_PI = math.log2(math.pi)

def logstar(n):
    if n <= 1:
        return 0.0
    result = 0.0
    value = math.log2(n)
    while value > 0:
        result += value
        value = math.log2(value)
    return result
depth_code_values = []
for depth in range(MAX_DEPTH + 1):
    depth_code_values.append(logstar(depth))
DEPTH_CODE = np.array(depth_code_values, dtype=np.float64)

def direct_reset_ctw_depth_logs(seq, dmax):
    n = len(seq)
    roots = np.empty(dmax + 1, dtype=np.float64)
    n1 = int(seq.sum())
    a = n - n1
    b = n1
    roots[0] = (math.lgamma(a + 0.5) + math.lgamma(b + 0.5) - math.lgamma(a + b + 1)) / LOG2 - LOG2_PI
    for D in range(1, dmax + 1):
        counts = []
        for j in range(D + 1):
            counts.append(np.zeros((1 << j, 2), dtype=np.int64))
        for t in range(n):
            bit = int(seq[t])
            counts[0][0, bit] += 1
            context = 0
            for j in range(1, min(D, t) + 1):
                context |= int(seq[t - j]) << j - 1
                counts[j][context, bit] += 1
        pe = []
        for j in range(D + 1):
            level = np.empty(1 << j, dtype=np.float64)
            for context in range(1 << j):
                a = int(counts[j][context, 0])
                b = int(counts[j][context, 1])
                level[context] = (math.lgamma(a + 0.5) + math.lgamma(b + 0.5) - math.lgamma(a + b + 1)) / LOG2 - LOG2_PI
            pe.append(level)
        weighted = pe[D].copy()
        for j in range(D - 1, -1, -1):
            parent = np.empty(1 << j, dtype=np.float64)
            for context in range(1 << j):
                split = weighted[context] + weighted[context | 1 << j]
                first = pe[j][context] - 1.0
                second = split - 1.0
                largest = max(first, second)
                parent[context] = largest + math.log2(2.0 ** (first - largest) + 2.0 ** (second - largest))
            weighted = parent
        roots[D] = weighted[0]
    return roots

def save_checkpoint(completed_length, completed_candidates, total_candidates, best_delta, best_start, best_end, best_length, best_depth, book_source_sha256, book_bit_sha256, n):
    np.savez_compressed(CHECKPOINT_FILE, n=n, max_length=MAX_LENGTH, max_depth=MAX_DEPTH, completed_length=completed_length, completed_candidates=completed_candidates, total_candidates=total_candidates, best_delta=best_delta, best_start=best_start, best_end=best_end, best_length=best_length, best_depth=best_depth, book_source_sha256=book_source_sha256, book_bit_sha256=book_bit_sha256)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
start_all = time.perf_counter()
excel_hits = sorted(INPUT_ROOT.rglob('40000tosses.xlsx'))
if len(excel_hits) != 1:
    raise FileNotFoundError(f'Expected one 40000tosses.xlsx; found {len(excel_hits)}: {excel_hits[:5]}')
coins = load_coins(excel_hits[0])
synthetic = np.random.default_rng(NORMAL_SEED).integers(0, 2, size=EXTRA_NORMAL_BITS, dtype=np.uint8)
normal = np.concatenate((coins, synthetic))
book_path = find_book_file()
book_source_sha256 = hashlib.sha256(book_path.read_bytes()).hexdigest()
insert, n_book_words, n_ties = make_book_insert(book_path, BOOK_LENGTH, TIE_SEED)
book_bit_sha256 = hashlib.sha256(insert.tobytes()).hexdigest()
mixed, truth = splice_sections(normal, insert)
assert len(coins) == 40000
assert len(normal) == 80000
assert len(mixed) == 88000
print(f'Real coins: {len(coins):,}, heads: {coins.sum():,}; synthetic fair bits: {len(synthetic):,}, ones: {synthetic.sum():,}; book source bits: {len(insert):,}, ones fraction: {insert.mean():.4f}; mixed: {len(mixed):,}', flush=True)
print(f'Book: {book_path}; words: {n_book_words}; equal-length pairs: {n_ties}; source SHA256: {book_source_sha256}; bit SHA256: {book_bit_sha256}', flush=True)
inserted_sections = []
for item in truth:
    inserted_sections.append((item['start_1based'], item['end_1based']))
print('Inserted sections:', inserted_sections, flush=True)
N = len(mixed)
total_candidates = 0
for length in range(1, MAX_LENGTH + 1):
    total_candidates += N - length + 1
print()
print('Authors-style exhaustive baseline')
print(f'N = {N:,}')
print(f'MAX_LENGTH = {MAX_LENGTH:,}')
print(f'MAX_DEPTH = {MAX_DEPTH}')
print(f'Total candidate intervals = {total_candidates:,}')
print('Indexed CTW = False')
print('Multiscale = False')
print('Reset-context CTW = True')
best_delta = -math.inf
best_start = -1
best_end = -1
best_length = -1
best_depth = -1
completed_length = 0
completed_candidates = 0
if CHECKPOINT_FILE.exists():
    checkpoint = np.load(CHECKPOINT_FILE, allow_pickle=False)
    same_run = int(checkpoint['n']) == N and int(checkpoint['max_length']) == MAX_LENGTH and (int(checkpoint['max_depth']) == MAX_DEPTH) and (str(checkpoint['book_source_sha256']) == book_source_sha256) and (str(checkpoint['book_bit_sha256']) == book_bit_sha256)
    if same_run:
        completed_length = int(checkpoint['completed_length'])
        completed_candidates = int(checkpoint['completed_candidates'])
        best_delta = float(checkpoint['best_delta'])
        best_start = int(checkpoint['best_start'])
        best_end = int(checkpoint['best_end'])
        best_length = int(checkpoint['best_length'])
        best_depth = int(checkpoint['best_depth'])
        print()
        print(f'Resuming after L = {completed_length}')
        print(f'Completed candidates = {completed_candidates:,}')
scan_start = time.perf_counter()
stopped_early = False
for length in range(completed_length + 1, MAX_LENGTH + 1):
    length_start = time.perf_counter()
    candidate_count = N - length + 1
    for start0 in range(candidate_count):
        end0 = start0 + length
        sequence = mixed[start0:end0]
        log_q = direct_reset_ctw_depth_logs(sequence, MAX_DEPTH)
        code_lengths = -log_q + DEPTH_CODE
        depth = int(np.argmin(code_lengths))
        atypical_length = float(code_lengths[depth])
        typical_length = float(len(sequence))
        delta = typical_length - atypical_length
        if delta > best_delta:
            best_delta = delta
            best_start = start0 + 1
            best_end = end0
            best_length = length
            best_depth = depth
        completed_candidates += 1
        if MAX_RUNTIME_HOURS is not None:
            runtime_hours = (time.perf_counter() - scan_start) / 3600
            if runtime_hours >= MAX_RUNTIME_HOURS:
                stopped_early = True
                break
    if stopped_early:
        finished_length = length - 1
    else:
        finished_length = length
    elapsed = time.perf_counter() - scan_start
    rate = completed_candidates / elapsed if elapsed > 0 else 0.0
    remaining = total_candidates - completed_candidates
    eta = remaining / rate if rate > 0 else math.inf
    print(f'L={length:4d}/{MAX_LENGTH} | done={completed_candidates:,}/{total_candidates:,} | length_time={time.perf_counter() - length_start:.3f}s | rate={rate:,.2f} intervals/s | elapsed={elapsed / 3600:.3f}h | ETA={eta / 3600:.3f}h | best={best_delta:.12f} | [{best_start},{best_end}] | D={best_depth}', flush=True)
    if not stopped_early and length % CHECKPOINT_EVERY_LENGTHS == 0:
        save_checkpoint(finished_length, completed_candidates, total_candidates, best_delta, best_start, best_end, best_length, best_depth, book_source_sha256, book_bit_sha256, N)
    if stopped_early:
        save_checkpoint(finished_length, completed_candidates, total_candidates, best_delta, best_start, best_end, best_length, best_depth, book_source_sha256, book_bit_sha256, N)
        break
scan_seconds = time.perf_counter() - scan_start
finished = completed_candidates == total_candidates
summary = {'setup': {'real_coins': len(coins), 'real_coin_heads': int(coins.sum()), 'synthetic_normal_bits': len(synthetic), 'synthetic_normal_ones': int(synthetic.sum()), 'normal_seed': NORMAL_SEED, 'normal_bits': len(normal), 'mixed_bits': len(mixed), 'book_source_bits': len(insert), 'book_words_available': n_book_words, 'book_tie_seed': TIE_SEED, 'book_equal_length_pairs': n_ties, 'book_source_sha256': book_source_sha256, 'book_bit_sha256': book_bit_sha256, 'book_ones_fraction': float(insert.mean()), 'insertion_intervals_1based': truth}, 'exhaustive': {'max_length': MAX_LENGTH, 'max_depth': MAX_DEPTH, 'total_candidates': total_candidates, 'completed_candidates': completed_candidates, 'finished': finished, 'scan_seconds': scan_seconds, 'best_delta': best_delta, 'best_start_1based': best_start, 'best_end_1based': best_end, 'best_length': best_length, 'best_depth': best_depth}, 'total_seconds': time.perf_counter() - start_all}
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print()
print(json.dumps(summary, indent=2))