# baselines.py — Three evaluation baselines for v1.
# Popularity: top-K most-interacted items (upper bound on non-personalized).
# Random-expected: analytical expected value of random selection (statistical floor).
# Content-centroid: cosine similarity to mean of user's history features (v0-style).

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import EVAL_K
from .evaluation import available_candidate_indices


def compute_item_popularity_log(
    train_df,  # pd.DataFrame with item_idx and is_positive columns
    num_items: int,
) -> torch.Tensor:
    """Compute log(1 + positive count) per item for hybrid scoring and baselines."""
    train_pos_counts = (
        train_df[train_df["is_positive"] == 1]
        .groupby("item_idx")
        .size()
    )
    popularity_log = torch.zeros(num_items, dtype=torch.float32)
    for item_idx, count in train_pos_counts.items():
        popularity_log[int(item_idx)] = math.log(1.0 + float(count))
    return popularity_log


def evaluate_popularity_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    popular_items: list[int],
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]:
    """Top-K most-interacted items per user (excluding already-seen).
    
    If the model can't beat this, it hasn't learned anything useful.
    """
    eligible_users = [
        u for u in positive_targets
        if user_histories.get(u) and u in candidate_cutoffs
    ]
    hit_rates: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for user_idx in eligible_users:
        available = set(
            available_candidate_indices(
                user_idx, seen_items, candidate_cutoffs, item_release_dates,
            )
        )
        recommendations = [
            idx for idx in popular_items if idx in available
        ][:k]
        targets = positive_targets[user_idx]
        hits = len(set(recommendations) & set(targets))
        hit_rates.append(float(hits > 0))
        precisions.append(hits / k)
        recalls.append(hits / len(targets))

    n = len(recalls)
    return {
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "users_evaluated": n,
    }


def evaluate_random_expected_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]:
    """Analytical expected value of random recommendation (hypergeometric).
    
    ExpectedHits = K × n/M where n=positives, M=candidates. Deterministic — no sampling.
    """
    eligible_users = [
        u for u in positive_targets
        if user_histories.get(u) and u in candidate_cutoffs
    ]
    hit_rates: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for user_idx in eligible_users:
        available_count = len(
            available_candidate_indices(
                user_idx, seen_items, candidate_cutoffs, item_release_dates,
            )
        )
        relevant_count = len(positive_targets[user_idx])
        draw_count = min(k, available_count)

        # Probability of at least one hit: 1 - P(all misses).
        miss_probability = 1.0
        for offset in range(draw_count):
            miss_probability *= (
                available_count - relevant_count - offset
            ) / (available_count - offset)
        expected_hits = draw_count * relevant_count / available_count
        hit_rates.append(1.0 - miss_probability)
        precisions.append(expected_hits / k)
        recalls.append(expected_hits / relevant_count)

    n = len(recalls)
    return {
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "users_evaluated": n,
    }


def evaluate_content_centroid_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    item_features: torch.Tensor,
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]:
    """v0-style content-based: average user's history features, rank by cosine similarity.
    
    Tests whether static content features alone can match learned collaborative retrieval.
    """
    eligible_users = [
        u for u in positive_targets
        if user_histories.get(u) and u in candidate_cutoffs
    ]
    feature_matrix = F.normalize(item_features, dim=1)
    device = item_features.device
    hit_rates: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for user_idx in eligible_users:
        history = torch.tensor(user_histories[user_idx], dtype=torch.long, device=device)
        user_profile = F.normalize(
            feature_matrix[history].mean(dim=0),
            dim=0,
        )
        available = available_candidate_indices(
            user_idx, seen_items, candidate_cutoffs, item_release_dates,
        )
        available_tensor = torch.tensor(available, dtype=torch.long, device=device)
        scores = feature_matrix[available_tensor] @ user_profile
        top_n = min(k, len(available))
        recommended = available_tensor[scores.topk(top_n).indices].tolist()

        targets = positive_targets[user_idx]
        hits = len(set(recommended) & set(targets))
        hit_rates.append(float(hits > 0))
        precisions.append(hits / k)
        recalls.append(hits / len(targets))

    n = len(recalls)
    return {
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "users_evaluated": n,
    }
