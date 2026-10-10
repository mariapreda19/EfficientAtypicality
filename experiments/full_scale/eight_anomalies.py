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
EXTRA_NORMAL_BITS = 40000
NORMAL_SEED = 20260923
ANOMALY_LENGTH = 1000
INSERT_AFTER_NORMAL = tuple(range(7000, 70001, 9000))
EPSILON = 0.05
MAX_LENGTH = 7000
MAX_DEPTH = 4
ALPHA = 0.05
P0_ONE = 0.5
RUN_COIN_CONTROL = True
TOP_K = 10
MATCH_IOU = 0.5


def load_coins(path):
    sheet = pd.read_excel(path, sheet_name='Sheet2', header=None, dtype=str)
    seen = {}
    for ncol, scol, hcol, tcol in ((3,4,5,6), (11,12,13,14)):
        for row in sheet.itertuples(index=False, name=None):
            n, initial = str(row[ncol]).strip(), str(row[scol]).strip()
            if not n.isdecimal() or initial not in ('H','T'): continue
            i = int(n)
            if not 1 <= i <= 40000: continue
            head = str(row[hcol]).strip() == '1'
            tail = str(row[tcol]).strip() == '1'
            if head == tail or i in seen: raise ValueError(f'Invalid/duplicate toss {i}')
            seen[i] = int(head)
    if len(seen) != 40000 or set(seen) != set(range(1,40001)):
        raise ValueError('Missing coin toss indices')
    ordered = []
    for i in range(1, 40001):
        ordered.append(seen[i])
    x = np.array(ordered, dtype=np.uint8)
    if int(x.sum()) != 20217: raise ValueError('Heads do not match paper (20,217)')
    return x

def find_book_file():
    if BOOK_FILE is not None:
        path = Path(BOOK_FILE)
        if not path.is_file(): raise FileNotFoundError(f'BOOK_FILE is not a file: {path}')
        return path
    key = 'on_the_shores_of_the_great_sea'
    files = []
    for p in INPUT_ROOT.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ('', '.txt', '.text', '.md'):
            continue
        normalized = str(p).lower().replace('-', '_').replace(' ', '_')
        if key in normalized:
            files.append(p)

    exact = []
    for p in files:
        normalized_stem = p.stem.lower().replace('-', '_').replace(' ', '_')
        if key in normalized_stem:
            exact.append(p)
    if len(exact) == 1: return exact[0]
    if len(files) == 1: return files[0]
    local = Path('upload/Text lipit (2)(6).txt')
    if not files and local.is_file() and not Path('/kaggle/input').exists(): return local
    raise FileNotFoundError(f'Expected one book text under {INPUT_ROOT}; found {files[:10]}')

def make_book_insert(source, n, tie_seed):
    content = source.read_text(encoding='utf-8-sig')

    content = re.sub(r'(?im)^Original Copyright[^\n]*Heritage History[^\n]*$', '', content)
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", content)
    if n is None: n = len(words)-1
    if n < 1 or len(words) < n+1:
        raise ValueError(f'Need {n+1} words for {n} comparisons; got {len(words)}')
    lengths_list = []
    for word in words[:n+1]:
        length = 0
        for character in word:
            if character.isalpha():
                length += 1
        lengths_list.append(length)
    lengths = np.array(lengths_list)
    changes = np.sign(np.diff(lengths))
    bits = (changes > 0).astype(np.uint8)
    ties = np.flatnonzero(changes == 0)
    bits[ties] = np.random.default_rng(tie_seed).integers(0,2,size=len(ties))
    return bits, len(words), len(ties)

def scales(N, L, eps, dmax):
    if not 0 < eps <= 0.5 or not 1 <= L <= N: raise ValueError('Invalid epsilon/L')
    m, result = 1, []
    while m <= L:
        h = max(1, math.floor(eps*m + 1e-12))
        j_first = (dmax + h - 1)//h
        j_last = (N-m)//h
        if j_first <= j_last:
            result.append((m, h, j_first, j_last, j_last-j_first+1))
        m = math.ceil((1+eps)*m - 1e-12)
    return result

def occurrence_index(x, dmax):
    positions = defaultdict(lambda: array('I'))
    N = len(x)
    for t in range(N):
        bit = int(x[t]); positions[(0,0,bit)].append(t)
        context = 0
        for j in range(1, min(dmax,t)+1):
            context |= int(x[t-j]) << (j-1)
            positions[(j,context,bit)].append(t)
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
            value = (math.lgamma(a + .5) + math.lgamma(b + .5)
                     - math.lgamma(a + b + 1)) / LOG2 - LOG2_PI
            level.append(value)
        pe.append(level)

    roots = []
    for d in range(dmax + 1):
        weighted = pe[d]
        for j in range(d - 1, -1, -1):
            next_level = []
            for ctx in range(1 << j):
                first = pe[j][ctx] - 1
                second = weighted[ctx] + weighted[ctx | (1 << j)] - 1
                larger = max(first, second)
                value = larger + math.log2(1.0 + 2.0 ** (-abs(first - second)))
                next_level.append(value)
            weighted = next_level
        roots.append(weighted[0])
    return roots

def indexed_ctw_depth_logs(index, start, end, dmax):
    counts = []
    for j in range(dmax+1):
        nodes = []
        for ctx in range(1<<j):
            pair = []
            for bit in (0,1):
                positions = index.get((j,ctx,bit), ())
                pair.append(bisect_left(positions,end)-bisect_left(positions,start))
            nodes.append(pair)
        counts.append(nodes)
    return ctw_depth_logs_from_counts(counts, dmax)

def direct_ctw_depth_logs(x, start, end, dmax):
    counts = []
    for j in range(dmax + 1):
        level = []
        for _ in range(1 << j):
            level.append([0, 0])
        counts.append(level)
    for t in range(start,end):
        bit = int(x[t]); counts[0][0][bit] += 1
        ctx = 0
        for j in range(1,dmax+1):
            ctx |= int(x[t-j]) << (j-1)
            counts[j][ctx][bit] += 1
    return ctw_depth_logs_from_counts(counts,dmax)

def check_oracle(x, index, dmax, checks=12):
    rng = np.random.default_rng(20260923)
    starts = [dmax, dmax+1, len(x)//2]
    random_starts = rng.integers(dmax, len(x) - 20, size=checks)
    for t in random_starts:
        starts.append(int(t))
    worst = 0.0
    for start in starts:
        end = min(len(x),start+int(rng.integers(1,30)))
        a = indexed_ctw_depth_logs(index,start,end,dmax)
        b = direct_ctw_depth_logs(x,start,end,dmax)
        for u, v in zip(a, b):
            error = abs(u - v)
            if error > worst:
                worst = error
    if worst > 1e-9: raise AssertionError(f'Indexed CTW differs from direct: {worst} bits')
    return worst

def check_normalization():


    history = [0,1,0,1]
    for d in range(3):
        for length in (1,2,3):
            total = 0.0
            for suffix in itertools.product((0,1),repeat=length):
                x = np.array(history+list(suffix),dtype=np.uint8)
                total += 2.0**direct_ctw_depth_logs(x,len(history),len(x),d)[d]
            if not math.isclose(total,1.0,abs_tol=1e-12):
                raise AssertionError(f'CTW is not normalized: d={d}, m={length}, total={total}')

def run_uad(x, name):
    N=len(x)
    grid=scales(N,MAX_LENGTH,EPSILON,MAX_DEPTH)
    M = 0
    for grid_item in grid:
        M += grid_item[-1]
    if not M: raise ValueError('No candidates')
    total_t0=time.perf_counter()
    index=occurrence_index(x,MAX_DEPTH)
    index_seconds=time.perf_counter()-total_t0
    oracle_error=check_oracle(x,index,MAX_DEPTH)
    prior=-np.log2([1-P0_ONE,P0_ONE])

    prefix=np.r_[0.,np.cumsum(prior[x],dtype=np.float64)]
    log_depth=math.log2(MAX_DEPTH+1)
    log_M=math.log2(M)
    threshold=math.log2(1/ALPHA)
    heap=[]; best_uniform=None; best_scale=None
    best_start_score=np.full(N,np.nan)
    best_start_record=[None]*N

    log_e_total=-math.inf
    scanned=0; scan_t0=time.perf_counter()
    for k,(m,h,j_first,j_last,Nk) in enumerate(grid):

        scale_penalty=(math.log2(Nk)+2*math.log2(k+1)
                       +math.log2(math.pi**2/6))
        for j in range(j_first,j_last+1):
            start=j*h; end=start+m
            log_q=indexed_ctw_depth_logs(index,start,end,MAX_DEPTH)
            d=max(range(MAX_DEPTH+1),key=lambda z: log_q[z]-log_depth)
            raw=float(prefix[end]-prefix[start]+log_q[d])
            adjusted=raw-log_depth
            au=adjusted-log_M
            asc=adjusted-scale_penalty
            rec={'start_1based':start+1,'end_1based':end,'length':m,
                 'depth':d,'raw_bits':raw,'A_uniform_bits':au,
                 'A_scale_bits':asc}
            if best_uniform is None or au>best_uniform['A_uniform_bits']: best_uniform=rec
            if best_scale is None or asc>best_scale['A_scale_bits']: best_scale=rec
            if math.isnan(best_start_score[start]) or au>best_start_score[start]:
                best_start_score[start]=au
                best_start_record[start]=rec
            key=(au,scanned,rec)
            if len(heap)<TOP_K: heapq.heappush(heap,key)
            elif au>heap[0][0]: heapq.heapreplace(heap,key)
            log_e_total=np.logaddexp2(log_e_total,au)
            scanned+=1
        if k%12==0 or k==len(grid)-1:
            print(f'{name}: {k+1}/{len(grid)} scales, {scanned:,}/{M:,} candidates',flush=True)
    scan_seconds=time.perf_counter()-scan_t0

    sum_scale_weights = 0.0
    for k in range(len(grid)):
        sum_scale_weights += 6 / (math.pi**2 * (k + 1)**2)

    top_uniform = []
    sorted_heap = sorted(heap, reverse=True)
    for item in sorted_heap:
        top_uniform.append(item[2])

    result={'name':name,'n':N,'epsilon':EPSILON,'rho':(1+EPSILON)**-2,
            'max_length':MAX_LENGTH,'max_depth':MAX_DEPTH,'scales':len(grid),
            'candidates':M,'depth_candidates':M*(MAX_DEPTH+1),
            'candidate_weight_mode':'uniform over intervals × uniform over depths',
            'sum_uniform_weights':1.0,
            'sum_scale_weights':sum(6/(math.pi**2*(k+1)**2) for k in range(len(grid))),
            'alpha':ALPHA,'threshold_bits':threshold,
            'index_seconds':index_seconds,'scan_seconds':scan_seconds,
            'total_uad_seconds':time.perf_counter()-total_t0,
            'max_index_direct_error_bits':oracle_error,
            'best_uniform':best_uniform,'best_scale_weighted':best_scale,
            'detected_uniform':best_uniform['A_uniform_bits']>=threshold,
            'detected_scale':best_scale['A_scale_bits']>=threshold,
            'log2_total_e_value_uniform':float(log_e_total),
            'top_uniform':[r for _,_,r in sorted(heap,reverse=True)],
            'best_start_scores_uniform':best_start_score,
            'best_start_record':best_start_record}
    return result

def splice_sections(normal, book_bits):
    positions=INSERT_AFTER_NORMAL
    if not (len(positions)>=2 and tuple(sorted(positions))==tuple(positions)
            and len(set(positions))==len(positions)
            and 0<=positions[0] and positions[-1]<=len(normal)):
        raise ValueError('INSERT_AFTER_NORMAL must be distinct ascending normal positions')
    if len(book_bits)<ANOMALY_LENGTH: raise ValueError('Need at least 1,000 book-derived bits')


    source_offsets = []
    for i in range(len(positions)):
        offset = math.floor(i * len(book_bits) / len(positions))
        source_offsets.append(offset)

    sections = []
    for offset in source_offsets:
        indices = (np.arange(ANOMALY_LENGTH) + offset) % len(book_bits)
        section = book_bits[indices]
        sections.append(section)
    pieces=[];truth=[];cursor=0;added=0
    for i,(pos,section,offset) in enumerate(zip(positions,sections,source_offsets),start=1):
        pieces.extend((normal[cursor:pos],section))
        a=pos+added;b=a+len(section)
        truth.append({'id':i,'start_1based':a+1,'end_1based':b,
                      'length':len(section),'source_offset_0based':offset,
                      'source_wraps':offset+len(section)>len(book_bits)})
        cursor=pos;added+=len(section)
    pieces.append(normal[cursor:])
    return np.concatenate(pieces),truth

def select_predictions(result):
    candidates = []
    for record in result['best_start_record']:
        if record is None:
            continue
        if record['A_uniform_bits'] >= result['threshold_bits']:
            candidates.append(record)

    candidates.sort(key=lambda record: record['A_uniform_bits'], reverse=True)

    selected = []
    for record in candidates:
        overlaps = False
        for old in selected:
            separated = (record['end_1based'] < old['start_1based']
                         or old['end_1based'] < record['start_1based'])
            if not separated:
                overlaps = True
                break
        if not overlaps:
            selected.append(dict(record))
    return selected

def match_after_scan(predictions, truth):
    neighbors = []

    for actual in truth:
        current = []
        actual_start = actual['start_1based'] - 1
        actual_end = actual['end_1based']

        for p in range(len(predictions)):
            pred = predictions[p]
            pred_start = pred['start_1based'] - 1
            pred_end = pred['end_1based']

            intersection = max(0, min(pred_end, actual_end) - max(pred_start, actual_start))
            union = (pred_end - pred_start) + (actual_end - actual_start) - intersection
            value = intersection / union

            if value > MATCH_IOU:
                current.append((value, p))

        current.sort(reverse=True)
        neighbors.append(current)

    assigned = {}

    def augment(t, seen):
        for value, p in neighbors[t]:
            if p in seen:
                continue
            seen.add(p)
            if p not in assigned or augment(assigned[p], seen):
                assigned[p] = t
                return True
        return False

    truth_order = list(range(len(truth)))
    truth_order.sort(key=lambda k: len(neighbors[k]))

    for t in truth_order:
        augment(t, set())

    matches = {}
    for p, t in assigned.items():
        pred = predictions[p]
        actual = truth[t]

        pred_start = pred['start_1based'] - 1
        pred_end = pred['end_1based']
        actual_start = actual['start_1based'] - 1
        actual_end = actual['end_1based']

        intersection = max(0, min(pred_end, actual_end) - max(pred_start, actual_start))
        union = (pred_end - pred_start) + (actual_end - actual_start) - intersection
        value = intersection / union
        matches[t] = (p, value)

    for p in range(len(predictions)):
        record = predictions[p]

        if p in assigned:
            record['matched_truth_id'] = truth[assigned[p]]['id']
        else:
            record['matched_truth_id'] = None

        best_iou = None
        for actual in truth:
            pred_start = record['start_1based'] - 1
            pred_end = record['end_1based']
            actual_start = actual['start_1based'] - 1
            actual_end = actual['end_1based']

            intersection = max(0, min(pred_end, actual_end) - max(pred_start, actual_start))
            union = (pred_end - pred_start) + (actual_end - actual_start) - intersection
            value = intersection / union

            if best_iou is None or value > best_iou:
                best_iou = value

        record['best_iou_to_truth'] = best_iou

    per_anomaly = []

    for t in range(len(truth)):
        actual = truth[t]
        match = matches.get(t)

        best_iou = 0.0
        for pred in predictions:
            pred_start = pred['start_1based'] - 1
            pred_end = pred['end_1based']
            actual_start = actual['start_1based'] - 1
            actual_end = actual['end_1based']

            intersection = max(0, min(pred_end, actual_end) - max(pred_start, actual_start))
            union = (pred_end - pred_start) + (actual_end - actual_start) - intersection
            value = intersection / union

            if value > best_iou:
                best_iou = value

        matched_prediction = None
        if match is not None:
            matched_prediction = dict(predictions[match[0]])
            matched_prediction['iou_with_truth'] = match[1]

        item = dict(actual)
        item['detected_iou_over_threshold'] = match is not None
        item['matched_prediction'] = matched_prediction
        item['best_iou_any_prediction'] = best_iou
        per_anomaly.append(item)

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth) - tp

    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'iou_threshold_strictly_greater_than': MATCH_IOU,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall_fraction_of_anomalies_detected': recall,
        'f1': f1,
        'per_anomaly': per_anomaly
    }

start_all=time.perf_counter()
check_normalization()
excel_hits = sorted(INPUT_ROOT.rglob('40000tosses.xlsx'))
if len(excel_hits) != 1:
    raise FileNotFoundError(
        f'Expected one 40000tosses.xlsx; found {len(excel_hits)}: {excel_hits[:5]}'
    )
coins = load_coins(excel_hits[0])
synthetic=np.random.default_rng(NORMAL_SEED).integers(0,2,size=EXTRA_NORMAL_BITS,dtype=np.uint8)
normal=np.concatenate((coins,synthetic))
book_path=find_book_file()
book_source_sha256=hashlib.sha256(book_path.read_bytes()).hexdigest()
insert,n_book_words,n_ties=make_book_insert(book_path,BOOK_LENGTH,TIE_SEED)
book_bit_sha256=hashlib.sha256(insert.tobytes()).hexdigest()
mixed,truth=splice_sections(normal,insert)
assert len(coins)==40000 and len(normal)==80000 and len(mixed)==len(normal)+8*ANOMALY_LENGTH
print(f'Real coins: {len(coins):,}, heads: {coins.sum():,}; '
      f'synthetic fair bits: {len(synthetic):,}, ones: {synthetic.sum():,}; '
      f'book source bits: {len(insert):,}, ones fraction: {insert.mean():.4f}; '
      f'mixed: {len(mixed):,}',flush=True)
print(f'Book: {book_path}; words: {n_book_words}; equal-length pairs: {n_ties}; '
      f'source SHA256: {book_source_sha256}; bit SHA256: {book_bit_sha256}',flush=True)
inserted_sections = []
for item in truth:
    inserted_sections.append((item['start_1based'], item['end_1based']))
print('Inserted sections:', inserted_sections, flush=True)
mixed_result=run_uad(mixed,'mixed')
control_result=run_uad(normal,'normal_control') if RUN_COIN_CONTROL else None
predictions=select_predictions(mixed_result)
comparison=match_after_scan(predictions,truth)
control_predictions=select_predictions(control_result) if control_result else []

mixed_json = {}
for key, value in mixed_result.items():
    if key not in ('best_start_scores_uniform', 'best_start_record'):
        mixed_json[key] = value

control_json = None
if control_result is not None:
    control_json = {}
    for key, value in control_result.items():
        if key not in ('best_start_scores_uniform', 'best_start_record'):
            control_json[key] = value

summary={'setup':{'real_coins':len(coins),'real_coin_heads':int(coins.sum()),
                  'synthetic_normal_bits':len(synthetic),'synthetic_normal_ones':int(synthetic.sum()),
                  'normal_seed':NORMAL_SEED,'normal_bits':len(normal),
                  'mixed_bits':len(mixed),'anomaly_length':ANOMALY_LENGTH,
                  'number_of_anomalies':len(truth),'book_source_bits':len(insert),
                  'book_reused':len(truth)*ANOMALY_LENGTH>len(insert),
                  'book_file':str(book_path),
                  'book_source_sha256':book_source_sha256,
                  'book_words_available':n_book_words,
                  'book_tie_seed':TIE_SEED,'book_equal_length_pairs':n_ties,
                  'book_bit_sha256':book_bit_sha256,
                  'book_ones_fraction':float(insert.mean()),
                  'insert_after_normal_position':INSERT_AFTER_NORMAL,
                  'insertion_intervals_1based':truth,
                  'frozen_p0_one':P0_ONE,
                  'depth_penalty_bits':math.log2(MAX_DEPTH+1),
                  'note':'All available words after removing matching Heritage History '
                         'page footers: next longer→1, shorter→0, '
                         'equal→reproducible fair bit. Apostrophes are excluded '
                         'from word length. Exact excerpt, tokenizer, insertion '
                         'and tie seed are not published in the paper. '
                         'Only 2,366 book-derived bits are available, so the eight '
                         '1,000-bit inserts reuse overlapping source windows; they '
                         'are not independent anomalies. The extra 40,000 normal '
                         'bits are seeded synthetic fair coin flips, distinct from '
                         'the 40,000 observed tosses. Text is not assumed i.i.d. '
                         'for the power theorem.'},
         'mixed':mixed_json,
         'predictions':predictions,'multi_anomaly_evaluation':comparison,
         'coin_control':control_json,
         'coin_control_number_of_detections':len(control_predictions) if control_result else None,
         'total_cell_seconds':None}
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(predictions).to_csv(OUTPUT_ROOT/'uad_eight_book_predictions.csv',index=False)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,6),sharex=True)
walk=np.cumsum(2*mixed.astype(np.int16)-1)
ix=np.arange(1,len(mixed)+1)
ax1.plot(ix,walk,lw=.8)
ax1.set_ylabel('Cumulative ±1 walk')
ax2.plot(ix,mixed_result['best_start_scores_uniform'],lw=.7,color='tab:red')
ax2.axhline(math.log2(1/ALPHA),color='black',ls='--',lw=.8,label='global alpha threshold')
ax2.set(xlabel='Bit position (1-based)',ylabel='Best penalized A per start')
for i,t in enumerate(truth):
    for ax in (ax1,ax2):
        ax.axvspan(t['start_1based'],t['end_1based'],color='orange',alpha=.16,
                   label='inserted anomaly' if i==0 and ax is ax2 else None)
for i,p in enumerate(predictions[:20]):
    ax2.axvspan(p['start_1based'],p['end_1based'],facecolor='none',edgecolor='tab:blue',
                lw=1,label='detected interval (top 20)' if i==0 else None)
ax2.legend();fig.tight_layout()
fig.savefig(OUTPUT_ROOT/'uad_eight_book.png',dpi=160)
plt.show()
summary['total_cell_seconds']=time.perf_counter()-start_all
(OUTPUT_ROOT/'uad_eight_book_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

evaluation_summary = {}
for key, value in comparison.items():
    if key != 'per_anomaly':
        evaluation_summary[key] = value

print(json.dumps({
    'mixed_candidates':mixed_result['candidates'],
    'control_candidates':control_result['candidates'] if control_result else None,
    'mixed_scan_seconds':round(mixed_result['scan_seconds'],3),
    'control_scan_seconds':round(control_result['scan_seconds'],3) if control_result else None,
    'true_anomaly_intervals':truth,
    'predicted_intervals':predictions,
    'evaluation':evaluation_summary,
    'per_anomaly':comparison['per_anomaly'],
    'control_number_of_detections':len(control_predictions) if control_result else None,
    'alpha':ALPHA,'threshold_bits':mixed_result['threshold_bits'],
    'total_cell_seconds':round(summary['total_cell_seconds'],3)
},indent=2))