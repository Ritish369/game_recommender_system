# evaluation.py — Full-catalog evaluation for v1 TwoTowerModel.
# Release-aware, seen-item-filtered recall/precision/hit-rate against the warm catalog.

from __future__ import annotations

import numpy as np
import torch

from .config import EVAL_K, EVAL_USER_BATCH_SIZE, HYBRID_ALPHA, MAX_HISTORY_LENGTH


def padded_history_batch(
    user_ids: list[int],
    history_map: dict[int, list[int]],
    padding_idx: int,
    max_history_length: int = MAX_HISTORY_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad user histories to max_history_length for batched encode_user().
    
    Returns (history_item_ids [B, max_len], lengths [B]).
    """
    user_histories: list[list[int]] = []
    lengths: list[int] = []
    for user_idx in user_ids:
        history = history_map.get(user_idx, [])[-max_history_length:]
        lengths.append(len(history))
        user_histories.append(
            history + [padding_idx] * (max_history_length - len(history))
        )
    return (
        torch.tensor(user_histories, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def available_candidate_indices(
    user_idx: int,
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict[int, np.datetime64 | np.ndarray],
    item_release_dates: np.ndarray,
) -> np.ndarray:
    """Return item indices available for a user: released AND not already seen.
    
    Release-aware: excludes items released after the user's cutoff date.
    Seen-item-aware: excludes items the user has already interacted with.
    """
    released = item_release_dates <= np.datetime64(candidate_cutoffs[user_idx])
    available = np.flatnonzero(released)
    seen = seen_items.get(user_idx, set())
    if seen:
        available = np.array(
            [idx for idx in available if idx not in seen],
            dtype=np.int64,
        )
    return available


@torch.no_grad()
def evaluate_filtered(
    model: torch.nn.Module,
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict[int, np.datetime64 | np.ndarray],
    item_features: torch.Tensor,
    item_release_dates: np.ndarray,
    device: torch.device,
    k: int = EVAL_K,
    user_batch_size: int = EVAL_USER_BATCH_SIZE,
    hybrid_alpha: float = HYBRID_ALPHA,
    item_popularity_log: torch.Tensor | None = None,
    padding_idx: int | None = None,
) -> dict[str, float]:
    """Score each eligible user against full warm catalog, compute retrieval metrics.
    
    Validation uses train history; test uses train+val history.
    Release-aware AND seen-item masking prevents future-data leakage and
    rediscovery reward.
    """
    model.eval()
    item_features = item_features.to(device)
    if item_popularity_log is not None:
        item_popularity_log = item_popularity_log.to(device)
    all_item_bias = model.item_bias.weight.squeeze(-1)  # [num_items]

    eligible_users = [
        u for u in positive_targets
        if user_histories.get(u) and u in candidate_cutoffs
    ]
    hit_rates: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    # Precompute all item embeddings (static features only — no item IDs needed).
    all_item_vecs = model.encode_all_items(item_features)  # [num_items, 128]

    for start in range(0, len(eligible_users), user_batch_size):
        batch_users = eligible_users[start : start + user_batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

        history_item_ids, _ = padded_history_batch(
            batch_users, user_histories, padding_idx if padding_idx is not None else 0,
        )
        user_vectors = model.encode_user(
            user_tensor,
            history_item_ids.to(device),
        )

        # Exact dot-product scoring against all warm items.
        scores = user_vectors @ all_item_vecs.T  # [B, num_items]
        bias = all_item_bias.unsqueeze(0)  # [1, num_items]
        scores = scores + bias

        # Hybrid blending: add popularity log signal at inference.
        if hybrid_alpha > 0 and item_popularity_log is not None:
            scores = scores + hybrid_alpha * item_popularity_log.unsqueeze(0)

        # Per-user candidate masking (release-aware + seen-item).
        for row, user_idx in enumerate(batch_users):
            available = available_candidate_indices(
                user_idx, seen_items, candidate_cutoffs, item_release_dates,
            )
            if len(available) == 0:
                continue

            available_tensor = torch.tensor(available, dtype=torch.long, device=device)
            user_scores = scores[row][available_tensor]
            top_k = min(k, len(available))
            top_indices = available_tensor[user_scores.topk(top_k).indices]

            targets = positive_targets[user_idx]
            hits = len(set(top_indices.tolist()) & set(targets))
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
