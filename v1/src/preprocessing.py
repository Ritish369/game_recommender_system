# preprocessing.py — Interaction and metadata cleaning for v1.
# Dedup, dtype enforcement, log-transforms, positive labels, catalog alignment.

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ENGAGEMENT_COLUMNS, ENGAGEMENT_LOG_COLUMNS


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate reviews, coerce dtypes, drop rows with missing keys."""
    interactions = interactions.drop_duplicates(
        subset=["review_id"], keep="first"
    ).copy()
    interactions.drop(columns=["review_id"], inplace=True)
    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")
    interactions.dropna(subset=["user_id", "app_id", "date"], inplace=True)
    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["app_id"] = interactions["app_id"].astype("int64")
    # interactions["is_recommended"] = interactions["is_recommended"].astype("bool")
    return interactions


def apply_engagement_log_transforms(interactions: pd.DataFrame) -> pd.DataFrame:
    """Log1p-transform hours, helpful, funny to compress heavy-tailed distributions."""
    for raw_col, log_col in zip(ENGAGEMENT_COLUMNS, ENGAGEMENT_LOG_COLUMNS):
        interactions[log_col] = np.log1p(interactions[raw_col].clip(lower=0))
    interactions.drop(columns=ENGAGEMENT_COLUMNS, inplace=True)
    return interactions


def sort_chronologically(interactions: pd.DataFrame) -> pd.DataFrame:
    """Sort by (user_id, date) so temporal splits are contiguous slices."""
    interactions = interactions.sort_values(
        ["user_id", "date"]
    ).reset_index(drop=True)
    return interactions


def add_positive_label(interactions: pd.DataFrame) -> pd.DataFrame:
    """Convert is_recommended (bool) → is_positive (bool).
    
    is_positive defines InfoNCE training targets and held-out evaluation ground truth.
    Using native bool avoids NumPy/PyArrow backend mismatch with pandas 2.0+.
    """
    interactions["is_positive"] = interactions["is_recommended"].astype("bool")
    interactions.drop(columns=["is_recommended"], inplace=True)
    return interactions


def clean_metadata(
    games: pd.DataFrame, games_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate, fill missing text, add has_description/has_tags flags."""
    games = games.drop_duplicates(subset=["app_id"], keep="first").copy()
    games_metadata = games_metadata.drop_duplicates(
        subset=["app_id"], keep="first"
    ).copy()

    games_metadata["description"] = (
        games_metadata["description"].fillna("").astype(str)
    )
    games_metadata["tags"] = games_metadata["tags"].apply(
        lambda tags: tags if isinstance(tags, list) else []
    )
    # Availability flags so the item tower can distinguish missing metadata from zeros.
    games_metadata["has_description"] = (
        games_metadata["description"].str.strip() != ""
    ).astype("bool")
    games_metadata["has_tags"] = (
        games_metadata["tags"].str.len() > 0
    ).astype("bool")
    games_metadata["has_text_metadata"] = (
        games_metadata["has_description"]
        | games_metadata["has_tags"]
    ).astype("bool")
    return games, games_metadata


def intersect_catalogs(
    games: pd.DataFrame,
    games_metadata: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep only items present in BOTH games table and metadata table.
    
    Saves pre-intersection catalog as fallback_games for cold-start recommendations.
    """
    fallback_games = games.copy()
    
    valid_catalog_app_ids = set(games_metadata["app_id"]) & set(games["app_id"])
    games = games[games["app_id"].isin(valid_catalog_app_ids)].copy()
    games_metadata = games_metadata[
        games_metadata["app_id"].isin(valid_catalog_app_ids)
    ].copy()
    interactions = interactions[
        interactions["app_id"].isin(valid_catalog_app_ids)
    ].copy()

    return games, games_metadata, interactions, fallback_games


def filter_to_interacted_catalog(
    games: pd.DataFrame,
    games_metadata: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop games no sampled user has touched — no collaborative signal."""
    interacted_app_ids = set(interactions["app_id"])
    games = games[games["app_id"].isin(interacted_app_ids)].copy()
    games_metadata = games_metadata[
        games_metadata["app_id"].isin(interacted_app_ids)
    ].copy()
    games.reset_index(drop=True, inplace=True)
    games_metadata.reset_index(drop=True, inplace=True)
    return games, games_metadata


def validate_catalog(
    games: pd.DataFrame, games_metadata: pd.DataFrame
) -> dict:
    """Assert catalog integrity: unique app_ids, non-null fields, metadata coverage."""
    assert games_metadata["description"].notna().all()
    assert games_metadata["tags"].apply(lambda t: isinstance(t, list)).all()
    assert games["app_id"].is_unique
    assert games_metadata["app_id"].is_unique
    return {
        "has_description": float(games_metadata["has_description"].mean()),
        "has_tags": float(games_metadata["has_tags"].mean()),
        "has_text_metadata": float(games_metadata["has_text_metadata"].mean()),
        "num_games": len(games),
    }
