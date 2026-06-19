# loaders.py — Raw data I/O and reservoir sampling for v1.
# Loads Steam CSV/JSON datasets and streams 3K active users via reservoir sampling.

from __future__ import annotations

import random

import pandas as pd

from .config import (
    INTERACTION_CHUNK_SIZE,
    MIN_RAW_REVIEWS,
    RANDOM_SEED,
    RAW_DIR,
    TARGET_SAMPLED_ACTIVE_USERS,
    USER_CHUNK_SIZE,
)


def load_raw_games() -> pd.DataFrame:
    """Load the Steam games catalog (50K+ titles)."""
    return pd.read_csv(RAW_DIR / "games.csv")


def load_raw_metadata() -> pd.DataFrame:
    """Load games metadata JSON (tags + descriptions per game)."""
    return pd.read_json(RAW_DIR / "games_metadata.json", lines=True)


def reservoir_sample_active_users() -> tuple[set[int], pd.DataFrame]:
    """Reservoir-sample 3K active users (≥20 reviews) in O(n) time, O(k) memory.

    Streams the 14M-user CSV in chunks, selecting a uniform random subset
    of eligible users. Then loads only those users' interactions.

    Returns (sampled_user_ids, interactions_df).
    """
    rng = random.Random(RANDOM_SEED)  # self-seeded; config.init_environment() handles global seeds
    reservoir: list[int] = []
    eligible_users_seen = 0

    for user_chunk in pd.read_csv(
        RAW_DIR / "users.csv",
        usecols=["user_id", "reviews"],
        chunksize=USER_CHUNK_SIZE,
    ):
        eligible_user_ids = user_chunk.loc[
            user_chunk["reviews"] >= MIN_RAW_REVIEWS, "user_id"
        ]
        for user_id in eligible_user_ids:
            eligible_users_seen += 1
            if len(reservoir) < TARGET_SAMPLED_ACTIVE_USERS:
                reservoir.append(int(user_id))
            else:
                replacement_idx = rng.randint(0, eligible_users_seen - 1)
                if replacement_idx < TARGET_SAMPLED_ACTIVE_USERS:
                    reservoir[replacement_idx] = int(user_id)

    sampled_user_ids = set(reservoir)

    # Load only interactions for sampled users (streaming to stay memory-safe).
    chunks = []
    for interaction_chunk in pd.read_csv(
        RAW_DIR / "recommendations.csv",
        chunksize=INTERACTION_CHUNK_SIZE,
    ):
        sampled_chunk = interaction_chunk[
            interaction_chunk["user_id"].isin(sampled_user_ids)
        ]
        if not sampled_chunk.empty:
            chunks.append(sampled_chunk)

    interactions = pd.concat(chunks, ignore_index=True)
    return sampled_user_ids, interactions
