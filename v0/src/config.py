from __future__ import annotations

import random
from pathlib import Path

# Added by Codex: derive project paths relative to this file so the v0 pipeline
# can run on any machine without hard-coded home-directory paths.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data_clean"
RAW_DATA_DIR = DATA_DIR / "raw_clean"
PROCESSED_DATA_DIR = DATA_DIR / "processed_clean"

GAMES_CSV = RAW_DATA_DIR / "games.csv"
METADATA_JSON = RAW_DATA_DIR / "games_metadata.json"
USERS_CSV = RAW_DATA_DIR / "users.csv"
INTERACTIONS_CSV = RAW_DATA_DIR / "recommendations.csv"

PROCESSED_GAMES_CSV = PROCESSED_DATA_DIR / "processed_games_v0_clean.csv"
PROCESSED_INTERACTIONS_CSV = PROCESSED_DATA_DIR / "processed_sampled_interactions_v0_clean.csv"
TRAIN_SIMULATIONS_CSV = PROCESSED_DATA_DIR / "train_user_simulations_v0_clean.csv"
TEST_SIMULATIONS_CSV = PROCESSED_DATA_DIR / "test_user_simulations_v0_clean.csv"
USER_PROFILES_CSV = PROCESSED_DATA_DIR / "user_profiles_v0_clean.csv"

DEFAULT_RANDOM_SEED = 42

MIN_INTERACTIONS = 20
TARGET_ACTIVE_USERS = 3000
USER_CHUNK_SIZE = 500_000
INTERACTIONS_CHUNK_SIZE = 250_000

METADATA_WT = 0.20
BINARY_WT = 0.15
TAG_WEIGHT = 0.35
DESCRIPTION_WT = 0.30

HOURS_WT = 0.7
HELPFUL_WT = 0.2
FUNNY_WT = 0.1

TAG_MIN_DF = 5
TAG_MAX_DF = 0.7
DESCRIPTION_MIN_DF = 10
DESCRIPTION_MAX_DF = 0.6
DESCRIPTION_MAX_FEATURES = 5000
MAX_SVD_COMPONENTS = 128

TRAIN_RATIO = 0.7

TOP_K_CANDIDATES = 300
MIN_CANDIDATE_POOL = 500
TOP_N = 20
USER_SIMILARITY_BATCH_SIZE = 256

SIMILARITY_WT = 0.7
POPULARITY_WT = 0.2
RECENCY_WT = 0.1

K_EVAL = 10

COLD_START_POSITIVE_INTERACTIONS_WT = 0.5
COLD_START_POSITIVE_RATIO_WT = 0.3
COLD_START_USER_REVIEWS_WT = 0.2


def build_rng(seed: int = DEFAULT_RANDOM_SEED) -> random.Random:
    return random.Random(seed)


RNG = build_rng()
