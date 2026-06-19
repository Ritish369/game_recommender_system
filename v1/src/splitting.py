# splitting.py — Temporal splitting and collaborative filtering for v1.
# Per-user chronological 70/15/15 split, engagement transforms, and iterative catalog
# stabilization so only items/users with enough collaborative signal survive.

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .config import (
    ENGAGEMENT_LOG_COLUMNS,
    FUNNY_WT,
    HELPFUL_WT,
    HOURS_WT,
    MIN_TRAIN_ITEM_POSITIVES,
    MIN_TRAIN_USER_EVENTS,
    MIN_TRAIN_USER_POSITIVES,
    TRAIN_RATIO,
    VAL_RATIO,
)


def temporal_split_by_user(
    frame: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological per-user split: earliest 70% train, next 15% val, last 15% test.
    
    Sorts each user's history by date, then slices. Every user contributes to all
    three splits. Random split would leak future data — this prevents it.
    """
    train_parts, val_parts, test_parts = [], [], []
    for _, user_history in tqdm(
        frame.groupby("user_id"),
        desc="Temporal user split",
    ):
        user_history = user_history.sort_values("date")
        history_size = len(user_history)
        train_end = max(1, int(history_size * train_ratio))
        val_end = max(train_end, int(history_size * (train_ratio + val_ratio)))
        train_parts.append(user_history.iloc[:train_end])
        val_parts.append(user_history.iloc[train_end:val_end])
        test_parts.append(user_history.iloc[val_end:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def fit_engagement_transform(
    train_frame: pd.DataFrame,
) -> tuple[dict, MinMaxScaler]:
    """Fit 99th-percentile clipping + MinMaxScaler on train only (no leakage).
    
    Engagement metrics (hours, helpful, funny) are heavy-tailed. Returns
    (clip_upper_dict, fitted_scaler) for use with add_engagement_strength.
    """
    clip_upper = {
        col: train_frame[col].quantile(0.99)
        for col in ENGAGEMENT_LOG_COLUMNS
    }
    clipped_train = train_frame[ENGAGEMENT_LOG_COLUMNS].copy()
    for col in ENGAGEMENT_LOG_COLUMNS:
        clipped_train[col] = clipped_train[col].clip(upper=clip_upper[col])
    scaler = MinMaxScaler()
    scaler.fit(clipped_train)
    return clip_upper, scaler


def add_engagement_strength(
    frame: pd.DataFrame,
    clip_upper: dict,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Compute implicit_rating = 0.75·hours + 0.15·helpful + 0.10·funny.
    
    Negative (not-recommended) interactions get implicit_rating=0 so they don't
    push user profiles toward disliked items.
    """
    frame = frame.copy()
    clipped = frame[ENGAGEMENT_LOG_COLUMNS].copy()
    for col in ENGAGEMENT_LOG_COLUMNS:
        clipped[col] = clipped[col].clip(upper=clip_upper[col])
    scaled = scaler.transform(clipped)
    frame["implicit_rating"] = (
        HOURS_WT * scaled[:, 0]
        + HELPFUL_WT * scaled[:, 1]
        + FUNNY_WT * scaled[:, 2]
    )
    frame.loc[frame["is_positive"] == 0, "implicit_rating"] = 0.0
    return frame


def retain_train_supported_catalog(
    train_frame: pd.DataFrame,
    min_item_positives: int = MIN_TRAIN_ITEM_POSITIVES,
    min_user_events: int = MIN_TRAIN_USER_EVENTS,
    min_user_positives: int = MIN_TRAIN_USER_POSITIVES,
) -> pd.DataFrame:
    """Iteratively prune items/users without enough signal for collaborative learning.
    
    Cross-filters: removing items may drop users below threshold and vice versa.
    Loops until DataFrame shape converges. Typically shrinks ~15K items → ~2.5K warm.
    """
    retained = train_frame.copy()
    previous_shape = None
    while previous_shape != retained.shape:
        previous_shape = retained.shape
        # Prune items with too few positive interactions.
        item_positive_counts = (
            retained[retained["is_positive"] == 1]
            .groupby("app_id")
            .size()
        )
        supported_items = set(
            item_positive_counts[
                item_positive_counts >= min_item_positives
            ].index
        )
        retained = retained[retained["app_id"].isin(supported_items)].copy()
        # Prune users with too few events or positives.
        user_stats = retained.groupby("user_id").agg(
            event_count=("app_id", "size"),
            positive_count=("is_positive", "sum"),
        )
        supported_users = set(
            user_stats[
                (user_stats["event_count"] >= min_user_events)
                & (user_stats["positive_count"] >= min_user_positives)
            ].index
        )
        retained = retained[retained["user_id"].isin(supported_users)].copy()
    return retained


def filter_splits_to_warm_catalog(
    warm_train: pd.DataFrame,
    val_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    games: pd.DataFrame,
    games_metadata: pd.DataFrame,
    item_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter val/test/games/metadata/features to the stabilized warm catalog."""
    retained_user_ids = set(warm_train["user_id"])
    retained_app_ids = set(warm_train["app_id"])
    val = val_interactions[
        val_interactions["user_id"].isin(retained_user_ids)
        & val_interactions["app_id"].isin(retained_app_ids)
    ].copy()
    test = test_interactions[
        test_interactions["user_id"].isin(retained_user_ids)
        & test_interactions["app_id"].isin(retained_app_ids)
    ].copy()
    games = games[games["app_id"].isin(retained_app_ids)].copy()
    games_metadata = games_metadata[
        games_metadata["app_id"].isin(retained_app_ids)
    ].copy()
    item_features_df = item_features_df[
        item_features_df["app_id"].isin(retained_app_ids)
    ].copy()
    return val, test, games, games_metadata, item_features_df


def drop_engagement_log_columns(
    *frames: pd.DataFrame,
) -> None:
    """Drop engagement log columns from each frame after implicit_rating is computed."""
    for df in frames:
        df.drop(columns=ENGAGEMENT_LOG_COLUMNS, inplace=True, errors="ignore")
