from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    COLD_START_POSITIVE_INTERACTIONS_WT,
    COLD_START_POSITIVE_RATIO_WT,
    COLD_START_USER_REVIEWS_WT,
    MIN_CANDIDATE_POOL,
    POPULARITY_WT,
    RECENCY_WT,
    SIMILARITY_WT,
    TOP_K_CANDIDATES,
    TOP_N,
    USER_SIMILARITY_BATCH_SIZE,
)
from .loaders import drop_unnamed_columns


def _min_max_scale(series: pd.Series) -> pd.Series:
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / denominator


def generate_candidate_similarities(
    user_profiles: pd.DataFrame,
    games: pd.DataFrame,
    top_k_candidates: int = TOP_K_CANDIDATES,
    min_candidate_pool: int = MIN_CANDIDATE_POOL,
    batch_size: int = USER_SIMILARITY_BATCH_SIZE,
) -> dict[float, list[tuple[int, float]]]:
    user_profiles = drop_unnamed_columns(user_profiles.copy())
    games = drop_unnamed_columns(games.copy())
    user_cols = [col for col in user_profiles.columns if col != "user_id"]
    game_cols = [col for col in games.columns if col not in ["app_id", "date_release"]]

    user_matrix = user_profiles[user_cols].to_numpy()
    game_matrix = games[game_cols].to_numpy()
    user_ids = user_profiles["user_id"].to_numpy()
    game_ids = games["app_id"].to_numpy()

    candidate_pool_size = min(max(top_k_candidates, min_candidate_pool), game_matrix.shape[0])

    user_item_similarities: dict[float, list[tuple[int, float]]] = {}
    for start_idx in tqdm(
        range(0, len(user_ids), batch_size),
        desc="Candidate similarity batches...",
    ):
        end_idx = start_idx + batch_size
        similarity_batch = user_matrix[start_idx:end_idx] @ game_matrix.T

        for row_idx, similarities in enumerate(similarity_batch):
            user_id = user_ids[start_idx + row_idx]
            ranked_indices = np.argsort(similarities)[-candidate_pool_size:][::-1]
            user_item_similarities[user_id] = [
                (int(game_ids[idx]), float(similarities[idx]))
                for idx in ranked_indices
            ]

    return user_item_similarities


def build_seen_items_lookup(interactions_data: pd.DataFrame) -> dict[float, set[int]]:
    interactions_data = drop_unnamed_columns(interactions_data.copy())
    return (
        interactions_data.groupby("user_id")["app_id"]
        .apply(lambda items: set(items.astype(int)))
        .to_dict()
    )


def filter_seen_items(
    user_id: float,
    user_item_similarities: dict[float, list[tuple[int, float]]],
    seen_items_lookup: dict[float, set[int]],
) -> list[tuple[int, float]]:
    seen_items = seen_items_lookup.get(user_id, set())
    return [
        (item_id, similarity)
        for item_id, similarity in user_item_similarities.get(user_id, [])
        if item_id not in seen_items
    ]


def build_recommendations(
    user_profiles: pd.DataFrame,
    games: pd.DataFrame,
    train_interactions: pd.DataFrame,
    top_k_candidates: int = TOP_K_CANDIDATES,
    top_n: int = TOP_N,
    similarity_weight: float = SIMILARITY_WT,
    popularity_weight: float = POPULARITY_WT,
    recency_weight: float = RECENCY_WT,
) -> tuple[dict[float, list[tuple[int, float]]], dict[float, list[tuple[int, float]]]]:
    games = drop_unnamed_columns(games.copy())
    train_interactions = drop_unnamed_columns(train_interactions.copy())
    user_item_similarities = generate_candidate_similarities(
        user_profiles=user_profiles,
        games=games,
        top_k_candidates=top_k_candidates,
    )
    seen_items_lookup = build_seen_items_lookup(train_interactions)

    popularity = _min_max_scale(games.set_index("app_id")["user_reviews"]).fillna(0)
    recency = _min_max_scale(pd.to_datetime(games.set_index("app_id")["date_release"])).fillna(0)
    popularity_dict = popularity.to_dict()
    recency_dict = recency.to_dict()

    user_recs: dict[float, list[tuple[int, float]]] = {}

    # Added by Codex: filter seen items before reranking so popularity/recency
    # only help choose among genuinely recommendable unseen games.
    for user_id in tqdm(user_item_similarities.keys(), desc="Recommendations built..."):
        candidates = filter_seen_items(user_id, user_item_similarities, seen_items_lookup)
        reranked = []

        for item_id, cosine_score in candidates:
            final_score = (
                similarity_weight * cosine_score
                + popularity_weight * popularity_dict.get(item_id, 0)
                + recency_weight * recency_dict.get(item_id, 0)
            )
            reranked.append((item_id, float(final_score)))

        user_recs[user_id] = sorted(reranked, key=lambda pair: pair[1], reverse=True)[:top_n]

    return user_recs, user_item_similarities


def recommend_cold_start(
    games_data: pd.DataFrame,
    interactions_data: pd.DataFrame,
    n: int = TOP_N,
) -> pd.DataFrame:
    games_data = drop_unnamed_columns(games_data.copy())
    interactions_data = drop_unnamed_columns(interactions_data.copy())
    positive_popularity = (
        interactions_data[interactions_data["is_recommended"] == 1]
        .groupby("app_id")
        .size()
        .rename("positive_interactions")
        .reset_index()
    )

    cold_start_candidates = games_data.merge(positive_popularity, on="app_id", how="left")
    cold_start_candidates["positive_interactions"] = cold_start_candidates[
        "positive_interactions"
    ].fillna(0)

    cold_start_candidates["cold_start_score"] = (
        cold_start_candidates["positive_interactions"].rank(pct=True)
        * COLD_START_POSITIVE_INTERACTIONS_WT
        + cold_start_candidates["positive_ratio"].rank(pct=True) * COLD_START_POSITIVE_RATIO_WT
        + cold_start_candidates["user_reviews"].rank(pct=True) * COLD_START_USER_REVIEWS_WT
    )

    return (
        cold_start_candidates.sort_values("cold_start_score", ascending=False)[
            ["app_id", "cold_start_score"]
        ]
        .head(n)
        .reset_index(drop=True)
    )
