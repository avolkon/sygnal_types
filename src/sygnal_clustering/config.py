from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 23_479
N_FEATURES = 500
DROP_COLUMNS = [0, 1, 2, 3, 504]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "Разработка" / "Эксперименты"
SUBMISSIONS_DIR = EXPERIMENTS_DIR / "submissions"

DATA_PATH = REPO_ROOT / "Run200_Wave_0_1.txt"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_V2_DIR = REPO_ROOT / "artifacts_v2"
ARTIFACTS_V3_DIR = REPO_ROOT / "artifacts_v3"

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"
SUBMISSION2_PATH = SUBMISSIONS_DIR / "submission2.csv"
SUBMISSION3_PATH = SUBMISSIONS_DIR / "submission3.csv"
SUBMISSION3A_PATH = SUBMISSIONS_DIR / "submission3a.csv"
SUBMISSION3B_PATH = SUBMISSIONS_DIR / "submission3b.csv"
SUBMISSION3C_PATH = SUBMISSIONS_DIR / "submission3c.csv"
SUBMISSION4_PATH = SUBMISSIONS_DIR / "submission4.csv"

KAGGLE_LEADERBOARD_BEST = EXPERIMENTS_DIR / "kaggle_leaderboard_best_0.44571.png"
KAGGLE_LEADERBOARD_FIRST = EXPERIMENTS_DIR / "kaggle_leaderboard_first_submission.png"
