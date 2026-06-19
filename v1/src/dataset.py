# dataset.py — Data structures and indexing for v1 training.
# Contiguous ID remapping, history/seen/heldout maps, prefix examples,
# SteamSequenceTwoTowerDataset, and DataLoader creation.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, MAX_HISTORY_LENGTH, RANDOM_SEED


def add_contiguous_indices(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    """Map user_id and app_id → contiguous 0..N-1 indices for embedding lookups.
    
    Returns (frame with user_idx/item_idx columns, user_id_to_idx, item_id_to_idx).
    """
    frame = frame.copy()
    frame["user_idx"], user_cats = pd.factorize(frame["user_id"])
    frame["item_idx"], item_cats = pd.factorize(frame["app_id"])
    user_id_to_idx = dict(zip(user_cats, range(len(user_cats))))
    item_id_to_idx = dict(zip(item_cats, range(len(item_cats))))
    return frame, user_id_to_idx, item_id_to_idx


def positive_history_map(
    *frames: pd.DataFrame,
) -> dict[int, list[int]]:
    """Build {user_idx: [item_idx, ...]} from positive interactions only (train only).
    
    The user tower builds each user's representation from items they positively
    interacted with. History is chronological (frame is already sorted).
    """
    combined = pd.concat(frames, ignore_index=True)
    positive = combined[combined["is_positive"] == 1]
    result: dict[int, list[int]] = {}
    for user_idx, group in positive.groupby("user_idx"):
        result[int(user_idx)] = list(group["item_idx"])
    return result


def seen_item_map(
    *frames: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build {user_idx: {item_idx, ...}} of ALL items each user has interacted with.
    
    Masks already-seen items at evaluation — prevents rewarding rediscovery.
    """
    combined = pd.concat(frames, ignore_index=True)
    result: dict[int, set[int]] = {}
    for user_idx, group in combined.groupby("user_idx"):
        result[int(user_idx)] = set(group["item_idx"])
    return result


def heldout_positive_map(
    frame: pd.DataFrame,
    seen_items: dict[int, set[int]],
) -> dict[int, list[int]]:
    """Build positive targets for val/test, excluding items already in seen_items.
    
    Evaluation metrics require ground-truth positives not seen in training.
    """
    positive = frame[frame["is_positive"] == 1]
    result: dict[int, list[int]] = {}
    for user_idx, group in positive.groupby("user_idx"):
        targets = [
            int(idx)
            for idx in group["item_idx"]
            if idx not in seen_items.get(int(user_idx), set())
        ]
        if targets:
            result[int(user_idx)] = targets
    return result


def filter_targets_released_by_cutoff(
    positive_targets: dict[int, list[int]],
    candidate_cutoffs: dict[int, pd.Timestamp],
    item_release_dates: np.ndarray,
) -> dict[int, list[int]]:
    """Remove heldout positives for items released after the user's cutoff date.
    
    A user cannot interact with a game before its release. Prevents penalizing
    the model for not recommending items that didn't exist yet.
    """
    result: dict[int, list[int]] = {}
    for user_idx, targets in positive_targets.items():
        cutoff = candidate_cutoffs.get(user_idx)
        if cutoff is None:
            continue
        cutoff_np = np.datetime64(cutoff)
        filtered = [
            t for t in targets
            if int(t) < len(item_release_dates)
            and item_release_dates[int(t)] <= cutoff_np
        ]
        if filtered:
            result[user_idx] = filtered
    return result


def build_prefix_examples(
    history_map: dict[int, list[int]],
    max_history_length: int = MAX_HISTORY_LENGTH,
) -> list[dict]:
    """Generate multiple training examples per user via sliding window over history.
    
    For user with history [A,B,C,D]: prefix[1]=history=[A], target=B; 
    prefix[2]=history=[A,B], target=C; etc. Teaches the model to recommend
    given any prefix, making it robust to users with few interactions.
    """
    examples: list[dict] = []
    for user_idx, history in history_map.items():
        for t in range(1, len(history)):
            # Items beyond max_history_length are dropped (O(L²) attention cost cap).
            prior = history[max(0, t - max_history_length) : t]
            target = history[t]
            examples.append({
                "user_idx": user_idx,
                "history_item_ids": prior,
                "positive_item_idx": target,
            })
    return examples


class SteamSequenceTwoTowerDataset(Dataset):
    """PyTorch Dataset wrapping prefix examples with right-padded history sequences."""

    def __init__(
        self,
        prefix_examples: list[dict],
        num_items: int,
        max_history_length: int = MAX_HISTORY_LENGTH,
    ):
        self.examples = prefix_examples
        self.max_history_length = max_history_length
        self.padding_item_idx = num_items  # one past the last real item index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        history = ex["history_item_ids"][-self.max_history_length:]
        history_len = len(history)
        # Right-pad with padding_item_idx to exactly max_history_length.
        padded = history + [self.padding_item_idx] * (
            self.max_history_length - history_len
        )
        return {
            "user_id": torch.tensor(ex["user_idx"], dtype=torch.long),
            "history_item_ids": torch.tensor(padded, dtype=torch.long),
            "pos_item_id": torch.tensor(ex["positive_item_idx"], dtype=torch.long),
        }


def create_dataloaders(
    train_examples: list[dict],
    num_items: int,
    batch_size: int = BATCH_SIZE,
    max_history_length: int = MAX_HISTORY_LENGTH,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Create training DataLoader with standard collation (no custom collate_fn needed).
    
    Returns (train_loader, full_train_loader, num_users, padding_item_idx).
    The full_train_loader (no shuffle, all examples) is useful for inspection.
    """
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    dataset = SteamSequenceTwoTowerDataset(
        train_examples, num_items, max_history_length
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    padding_item_idx = dataset.padding_item_idx
    # num_users = largest user_idx seen in examples + 1
    num_users = max(ex["user_idx"] for ex in train_examples) + 1
    return loader, full_loader, num_users, padding_item_idx
