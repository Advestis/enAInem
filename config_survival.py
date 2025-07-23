DATASET_NAMES = [
    "veterans_lung_cancer",
    "flchain",
    "whas500",
    "breast_cancer",
    "gbsg2",
    "aids",
    "ds1",
    "ds2",
    "ds3",
]
N_TRIALS = 25
# MAX_ITER_NTF = 2
MAX_ITER_NTF = 0
MIN_WEIGHTS_CHANGE = 1.0e-2
USE_CONCORDANCE_INDEX_IPCW = True
NTF_SEED_NAMES = ["nmf", "cox", "flat"]
PERCENTILES_BOUNDS = [10, 25, 50, 75, 90]
MAX_N_COMPONENTS = len(PERCENTILES_BOUNDS) + 1
BOOTSTRAP_COX = True
SIMPLE_IMPUTER = True

