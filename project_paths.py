from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
DATA_FILES = DATA_ROOT
KNOWN_TYPICAL = DATA_ROOT / "Known Typical Model"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
