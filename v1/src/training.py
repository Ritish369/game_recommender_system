# training.py — Training logic for v1 TwoTowerModel.
# In-batch InfoNCE with false-negative masking, LogQ correction, learnable temperature,
# gradient clipping, AdamW + ReduceLROnPlateau, early stopping, and checkpoint management.

from __future__ import annotations

import json
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    ABLATION,
    ABLATION_RESULTS_PATH,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    EVAL_K,
    GRAD_CLIP_MAX_NORM,
    HYBRID_ALPHA,
    LEARNING_RATE,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    MODELS_DIR,
    WEIGHT_DECAY,
)
from .evaluation import evaluate_filtered


def compute_log_q(
    train_examples: list[dict],
    num_items: int,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """Compute corrected LogQ debiasing scores (Yi et al. 2019).
    
    Standard logQ: log(p_j) where p_j = empirical item frequency.
    Corrected: log(1 - (1-p_j)^B) — accounts for an item appearing as negative
    in up to B-1 other rows within the same batch.
    """
    item_freq = torch.zeros(num_items, dtype=torch.float32)
    for ex in train_examples:
        item_freq[ex["positive_item_idx"]] += 1
    item_freq = item_freq.clamp(min=1)
    item_prob = item_freq / item_freq.sum()
    log_q = torch.log(1.0 - (1.0 - item_prob) ** batch_size)
    return log_q


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    item_features: torch.Tensor,
    positive_matrix: torch.Tensor,
    device: torch.device,
    log_q: torch.Tensor | None = None,
) -> float:
    """One epoch of in-batch InfoNCE training with false-negative masking.

    Each batch forms a B×B score matrix. Diagonal = true positives;
    off-diagonal = in-batch negatives. False negatives (known positives
    appearing off-diagonal) are masked to -1e9. LogQ corrects for
    popular-item over-representation in in-batch negatives.
    """
    model.train()
    item_features = item_features.to(device)
    positive_matrix = positive_matrix.to(device)
    if log_q is not None:
        log_q = log_q.to(device)

    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        if len(batch["user_id"]) < 2:
            continue  # singleton batch — no in-batch contrast possible

        user_ids = batch["user_id"].to(device)
        history_item_ids = batch["history_item_ids"].to(device)
        pos_item_ids = batch["pos_item_id"].to(device)

        optimizer.zero_grad(set_to_none=True)

        user_vectors = model.encode_user(
            user_ids, history_item_ids,
        )
        pos_item_vectors = model.encode_item(
            pos_item_ids, item_features[pos_item_ids],
        )

        # B×B logit matrix: logits[i][j] = user_i · item_j
        logits = user_vectors @ pos_item_vectors.T
        logits = logits + model.item_bias(pos_item_ids).squeeze(-1).unsqueeze(0)

        # Mask false negatives: (user, item) pairs that appear in train but are off-diagonal.
        false_negative_mask = positive_matrix[user_ids][:, pos_item_ids]
        false_negative_mask.fill_diagonal_(False)
        logits = logits.masked_fill(false_negative_mask, -1e9)

        targets = torch.arange(len(user_ids), device=device)
        if log_q is not None:
            adjusted_logits = logits / model.temperature() - log_q[pos_item_ids].unsqueeze(0)
        else:
            adjusted_logits = logits / model.temperature()
        loss = F.cross_entropy(adjusted_logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    item_features: torch.Tensor,
    train_pos_mask: torch.Tensor,
    train_examples: list[dict],
    num_items: int,
    val_positive_targets: dict,
    val_user_histories: dict,
    train_seen: dict,
    val_candidate_cutoffs: dict,
    item_popularity_log: torch.Tensor,
    warm_item_release_dates: np.ndarray,
    device: torch.device,
    padding_idx: int,
    hybrid_alpha: float = HYBRID_ALPHA,
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    ablation: str = ABLATION,
) -> dict:
    """Full training loop: InfoNCE training, validation, checkpointing, early stopping.

    Monitors validation Recall@10 (not loss) because InfoNCE loss is batch-dependent
    and doesn't reflect full-catalog retrieval quality.

    Returns dict with best_state_dict, best_epoch, best_val_recall, train_losses,
    val_recalls, total_time_s, final_lr, and experiment metadata.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="max", patience=LR_SCHEDULER_PATIENCE,
    )

    # Compute corrected LogQ once (depends only on train item frequencies).
    log_q = compute_log_q(train_examples, num_items, BATCH_SIZE)

    best_state_dict = None
    best_val_recall = -float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_recalls: list[float] = []

    print("=" * 65)
    print(f"{'EXPERIMENT TRACKING':^65}")
    print("=" * 65)
    print(f"  Model: Two-Tower (embed_dim={EMBEDDING_DIM}, hidden=256, heads=4)")
    print(f"  Ablation: {ablation}")
    print("  Optimizer: AdamW (lr=3e-4, wd=1e-4)")
    print("  Scheduler: ReduceLROnPlateau (mode=max, patience=3)")
    print(f"  LogQ: corrected (1-(1-p)^B), B={BATCH_SIZE}")
    print(f"  Early stopping: patience={early_stopping_patience} epochs")
    print(f"  Train examples: {len(train_examples):,} | Items: {num_items:,}")
    print("=" * 65)

    train_start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            item_features=item_features,
            positive_matrix=train_pos_mask,
            device=device,
            log_q=log_q,
        )
        val_metrics = evaluate_filtered(
            model=model,
            positive_targets=val_positive_targets,
            user_histories=val_user_histories,
            seen_items=train_seen,
            candidate_cutoffs=val_candidate_cutoffs,
            item_features=item_features,
            item_release_dates=warm_item_release_dates,
            device=device,
            k=EVAL_K,
            hybrid_alpha=hybrid_alpha,
            item_popularity_log=item_popularity_log,
            padding_idx=padding_idx,
        )
        val_recall = val_metrics["Recall@10"]
        scheduler.step(val_recall)

        train_losses.append(loss)
        val_recalls.append(val_recall)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - train_start_time
        is_best = "*" if val_recall > best_val_recall else " "

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch
            best_state_dict = deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | loss={loss:.4f} | "
            f"val R@10={val_recall:.4f} | lr={lr:.2e} | "
            f"{elapsed:.0f}s{is_best}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    total_time = time.time() - train_start_time
    print("\n" + "=" * 65)
    print(f"Training complete. Best: epoch {best_epoch}, val R@10={best_val_recall:.4f}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 65)

    # ── Restore best checkpoint ──
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── Save best checkpoint to disk ──
    checkpoint_path = MODELS_DIR / f"two_tower_{ablation}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, checkpoint_path)
    print(f"Best checkpoint saved to {checkpoint_path}")

    # ── Persist ablation results ──
    result_entry = {
        "ablation": ablation,
        "best_epoch": best_epoch,
        "best_val_recall": best_val_recall,
        "total_train_time_s": total_time,
        "final_lr": lr,
    }
    if ABLATION_RESULTS_PATH.exists():
        existing = json.loads(ABLATION_RESULTS_PATH.read_text())
    else:
        existing = []
    existing.append(result_entry)
    ABLATION_RESULTS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"Ablation results saved to {ABLATION_RESULTS_PATH}")

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_recall": best_val_recall,
        "train_losses": train_losses,
        "val_recalls": val_recalls,
        "total_time_s": total_time,
        "final_lr": lr,
        "checkpoint_path": checkpoint_path,
    }
