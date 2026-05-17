from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 23_479
N_FEATURES = 500
RAW_COLUMNS = 505
DROP_COLUMNS = [0, 1, 2, 3, 504]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Run200_Wave_0_1.txt"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SUBMISSION_PATH = REPO_ROOT / "submission.csv"
