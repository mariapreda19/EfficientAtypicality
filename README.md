# Efficient Multiscale Atypicality Detection with Indexed Context-Tree Weighting

This repository contains implementations of the exhaustive CTW-based atypicality reference, indexed/multiscale variants, synthetic validation experiments, and a larger eight-anomaly benchmark.

## 1. Environment setup

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Always invoke scripts as modules** (`python -m experiments.<group>.<script>`) from the repository root. 

## 2. Reproduce the reference and indexed experiments (Table 1 from manuscript)

The scripts under `experiments/reproduction/` reconstruct examples from the earlier exhaustive atypicality implementation and compare them with the indexed/multiscale approach. **Very important! Run the original before the indexed variant** where a baseline `.npz` is used: the indexed scripts inspect the baseline if present and otherwise may run without a matched runtime comparison.

```bash
# RANDU
python -m experiments.reproduction.randu_original
python -m experiments.reproduction.randu_reduced

# HRV
python -m experiments.reproduction.hrv_original
python -m experiments.reproduction.hrv_reduced
python -m experiments.reproduction.hrv_validation

# Bacterial DNA
python -m experiments.reproduction.bacterial_original
python -m experiments.reproduction.bacterial_indexed

# Viral DNA
python -m experiments.reproduction.viral_original
python -m experiments.reproduction.viral_indexed

# Additional text reproduction
python -m experiments.reproduction.text_original
python -m experiments.reproduction.text_reduced
```

## 3. Reproduce the larger-scale benchmarks

These scripts use the real `40000tosses.xlsx` data and implement separate full-sequence experiments:

```bash
# 40k-coin RANDU insertion, indexed multiscale
python -m experiments.full_scale.randu_indexed

# 40k-coin text insertion, indexed multiscale
python -m experiments.full_scale.text_indexed

# Eight injected text-derived anomalies, indexed multiscale
python -m experiments.full_scale.eight_anomalies
```

The paired exhaustive entry points are:

```bash
python -m experiments.full_scale.randu_exhaustive
python -m experiments.full_scale.text_exhaustive
python -m experiments.full_scale.eight_anomalies_exhaustive
```

**Important!** exhaustive scanning here can take hours or days. In particular, the eight-anomaly scan covers `N=88,000` bits, `L=7,000`, hence `591,503,500` candidate intervals. In `eight_anomalies_exhaustive.py`, the default `MAX_RUNTIME_HOURS = None` means *no time limit*. Set this constant to `2` near the top of the file to reproduce an approximately two-hour baseline rather than an attempted full run. The script writes `results/eight_anomalies_exhaustive_checkpoint.npz` and `results/eight_anomalies_exhaustive_summary.json`.

The source text is `data/on_the_shores_of_the_great_sea.txt`; scripts record text-derived bitstream hashes and insertion coordinates. The indexed eight-anomaly run generates, among other files:

```text
results/uad_eight_book_metrics.json
results/uad_eight_book_predictions.csv
results/uad_eight_book.png
```

Other full-scale scripts output `results/uad_coin_randu_metrics.json`, `results/uad_coin_book_metrics.json`, CSV predictions and PNG figures. The exhaustive RANDU/text scripts output corresponding `coin_randu_metrics.json` and `coin_book_metrics.json` along with plots/CSV results.

## 4. Reproduce the synthetic/statistical experiments

Run them from the repository root:

```bash
# Experiment 1: conditional vs ordinary atypicality;
# null, context shift, visible and hidden mechanism shift
python -m experiments.synthetic.conditional_vs_ordinary

# Experiment 2: detection power vs anomaly length and Bernoulli strength;
# empirical power compared with the finite-sample sufficient condition
python -m experiments.synthetic.power_vs_length

# Experiment 3: multiscale localization and exactness/runtime of indexed CTW
python -m experiments.synthetic.multiscale_and_ctw

# Experiment 4: global false-alarm calibration, weighted vs unweighted scans
python -m experiments.synthetic.false_alarm
```

Selected outputs:

```text
experiment1_results/experiment1_summary.csv
experiment2_results/empirical_90_power_thresholds.csv
experiment3_results/experiment3b_ctw_accuracy.csv
experiment3_results/experiment3b_ctw_summary.csv
experiment4_results/experiment4_evalue_summary.csv
experiment4_results/experiment4_false_alarm_scales_alpha005.csv
```
