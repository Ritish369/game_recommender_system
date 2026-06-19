# config.py — All constants, paths, hyperparameters, and ablation control for v1.
# Every value here is sourced from canonical notebook cell 0 and the HYBRID_ALPHA sweep.

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ── Project paths ──
# Derive v1/ root relative to this file so the pipeline runs from any cwd.
BASE_DIR = Path(__file__).resolve().parents[1]  # v1/
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ABLATION_RESULTS_PATH = BASE_DIR / "ablation_results.json"

# ── Reproducibility ──
RANDOM_SEED = 42


# ── Data sampling ──
# Reservoir-sample 3K active users (≥20 reviews) to keep memory tractable.
TARGET_SAMPLED_ACTIVE_USERS = 3_000
MIN_RAW_REVIEWS = 20
USER_CHUNK_SIZE = 500_000
INTERACTION_CHUNK_SIZE = 250_000

# ── Temporal split (per-user chronological, not random) ──
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test gets remainder: 1.0 - TRAIN_RATIO - VAL_RATIO = 0.15

# ── Collaborative filtering thresholds ──
# Items need ≥5 positive train interactions; users need ≥5 events + ≥2 positives.
MIN_TRAIN_ITEM_POSITIVES = 5
MIN_TRAIN_USER_EVENTS = 5
MIN_TRAIN_USER_POSITIVES = 2

# ── Sequence modeling ──
MAX_HISTORY_LENGTH = 50  # right-padded; attention is O(L²) so cap matters

# ── Training ──
BATCH_SIZE = 512
EMBEDDING_DIM = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
LR_SCHEDULER_PATIENCE = 3  # ReduceLROnPlateau epochs without improvement
GRAD_CLIP_MAX_NORM = 1.0

# ── Evaluation ──
HYBRID_ALPHA = 0.2  # optimal from sweep on no_user_id model (test R@10: 0.0856)
EVAL_K = 10
EVAL_USER_BATCH_SIZE = 256

# ── Engagement features ──
# Raw columns (heavy-tailed) → log1p transforms for downstream scaling.
ENGAGEMENT_COLUMNS = ["hours", "helpful", "funny"]
ENGAGEMENT_LOG_COLUMNS = [f"{col}_log" for col in ENGAGEMENT_COLUMNS]

# ── Implicit rating weights (engagement → single confidence signal) ──
HOURS_WT = 0.75
HELPFUL_WT = 0.15
FUNNY_WT = 0.10

# ── Feature block weights (L2-normalized before concatenation) ──
# Tuned on v0 validation; descriptions carry the most signal.
METADATA_WEIGHT = 0.15
BINARY_WEIGHT = 0.15
TAG_WEIGHT = 0.25
DESCRIPTION_WEIGHT = 0.45

# ── Ablation control ──
# no_user_id: best config — removes per-user embedding, forces history-only generalization.
# Other variants: "full", "no_popularity", "mean_pool"
ABLATION = "no_user_id"


def init_environment() -> None:
    """Initialize directory structure and reproducible random seeds.
    
    Call once at pipeline start. Safe to call multiple times (idempotent).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
