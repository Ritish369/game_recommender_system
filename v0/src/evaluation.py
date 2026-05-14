from __future__ import annotations

import pandas as pd

from .config import K_EVAL


def evaluate_recommendations(
    user_recs: dict[float, list[tuple[int, float]]],
    test_interactions: pd.DataFrame,
    k: int = K_EVAL,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    positive_test = test_interactions[test_interactions["is_recommended"] == 1].copy()
    relevant_items_by_user = positive_test.groupby("user_id")["app_id"].apply(set).to_dict()

    metrics: list[dict[str, float]] = []
    for user_id, recs in user_recs.items():
        relevant_items = relevant_items_by_user.get(user_id, set())
        if not relevant_items:
            continue

        recommended_items = [app_id for app_id, _ in recs[:k]]
        hits = len(set(recommended_items) & relevant_items)

        metrics.append(
            {
                "user_id": user_id,
                f"hit_rate@{k}": int(hits > 0),
                f"precision@{k}": hits / k,
                f"recall@{k}": hits / len(relevant_items),
            }
        )

    eval_results = pd.DataFrame(metrics)
    diagnostics = pd.Series(
        {
            "users_with_profiles": len(user_recs),
            "users_with_positive_test_items": len(relevant_items_by_user),
            "users_evaluated": len(eval_results),
        }
    )

    if eval_results.empty:
        eval_summary = pd.DataFrame(
            {"score": [0.0, 0.0, 0.0]},
            index=[f"hit_rate@{k}", f"precision@{k}", f"recall@{k}"],
        )
    else:
        eval_summary = eval_results.drop(columns="user_id").mean().to_frame(name="score")

    return diagnostics, eval_results, eval_summary
