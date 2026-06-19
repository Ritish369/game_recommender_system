# cold_start.py — Cold-start recommendation for v1.
# Items filtered out during catalog stabilization (insufficient collaborative signal)
# are ranked by metadata quality signals for discoverability.

from __future__ import annotations

import pandas as pd


def recommend_cold_start_v1(
    games_data: pd.DataFrame,
    train_interactions: pd.DataFrame,
    n: int = 10,
    item_id_to_idx: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Recommend top-N cold-start items using popularity + quality metadata signals.
    
    Items without learned embeddings (insufficient interactions) are ranked by:
    60% positive interaction volume + 25% positive ratio + 15% review count.
    Uses the fallback_games catalog + pre-warm-filter train data.
    """
    positive_popularity = (
        train_interactions[train_interactions["is_positive"] == 1]
        .groupby("app_id")
        .size()
        .rename("positive_interactions")
        .reset_index()
    )
    candidates = games_data.merge(
        positive_popularity, on="app_id", how="left",
    )
    candidates["positive_interactions"] = (
        candidates["positive_interactions"].fillna(0)
    )
    candidates["cold_start_score"] = (
        0.60 * candidates["positive_interactions"].rank(pct=True)
        + 0.25 * candidates["positive_ratio"].rank(pct=True)
        + 0.15 * candidates["user_reviews"].rank(pct=True)
    )
    if item_id_to_idx is not None:
        candidates["warm_item_idx"] = candidates["app_id"].map(item_id_to_idx)
    else:
        candidates["warm_item_idx"] = None

    return (
        candidates
        .sort_values("cold_start_score", ascending=False)
        [["app_id", "warm_item_idx", "cold_start_score"]]
        .head(n)
        .reset_index(drop=True)
    )
