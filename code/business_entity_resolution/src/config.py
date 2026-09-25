"""
Configuration and constants for Business Entity Resolution Pipeline.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "Deputy_pipe" / "Datasets" / "student_resource" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_S1 = DATASET_DIR / "train" / "train_source1.tsv"
TRAIN_S2 = DATASET_DIR / "train" / "train_source2.tsv"
TRAIN_S3 = DATASET_DIR / "train" / "train_source3.tsv"
TRAIN_GT = DATASET_DIR / "train" / "train_ground_truth.tsv"

TEST_S1 = DATASET_DIR / "test" / "test_source1.tsv"
TEST_S2 = DATASET_DIR / "test" / "test_source2.tsv"
TEST_S3 = DATASET_DIR / "test" / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# Country partitions
VALID_COUNTRIES = ["US", "India", "France"]

# Delimiters
DELIM_TSV = "\t"
DELIM_LIST = ","

# Blocking & Matching Hyperparameters
BLOCKING_TOP_K = 10
MAX_CANDIDATES_PER_ENTITY = 15
SIMILARITY_MIN_THRESHOLD = 0.35

# Matcher Decision Threshold
# Calibrated for F_0.5 macro optimization
MATCH_PROBABILITY_THRESHOLD = 0.72
SINGLETON_HIGH_CONFIDENCE_THRESHOLD = 0.85
