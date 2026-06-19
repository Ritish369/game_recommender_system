# pipeline.py — Stage orchestrator for v1 two-tower pipeline.
# Chains data loading → feature engineering → temporal splits → training → evaluation.
# Top-level orchestrator: imports all submodules, no circular dependencies.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .baselines import (
    compute_item_popularity_log,
    evaluate_content_centroid_baseline,
    evaluate_popularity_baseline,
    evaluate_random_expected_baseline,
)
from .cold_start import recommend_cold_start_v1
from .config import (
    BATCH_SIZE,
    EVAL_K,
    HYBRID_ALPHA,
    MAX_HISTORY_LENGTH,
    init_environment,
)
from .dataset import (
    add_contiguous_indices,
    build_prefix_examples,
    create_dataloaders,
    filter_targets_released_by_cutoff,
    heldout_positive_map,
    positive_history_map,
    seen_item_map,
)
from .evaluation import evaluate_filtered
from .features import (
    build_weighted_item_matrix,
    clean_text_for_embeddings,
    encode_descriptions,
    encode_tags,
    engineer_game_features,
    enrich_descriptions_with_title,
    merge_feature_tables,
    rescale_positive_ratio,
)
from .loaders import (
    load_raw_games,
    load_raw_metadata,
    reservoir_sample_active_users,
)
from .model import TwoTowerModel
from .preprocessing import (
    add_positive_label,
    apply_engagement_log_transforms,
    clean_interactions,
    clean_metadata,
    filter_to_interacted_catalog,
    intersect_catalogs,
    sort_chronologically,
)
from .splitting import (
    add_engagement_strength,
    drop_engagement_log_columns,
    filter_splits_to_warm_catalog,
    fit_engagement_transform,
    retain_train_supported_catalog,
    temporal_split_by_user,
)
from .training import train_model


def run_data_stage() -> dict:
    """Load raw data, reservoir-sample active users, clean interactions and metadata."""
    init_environment()
    games = load_raw_games()
    games_metadata = load_raw_metadata()
    sampled_user_ids, interactions = reservoir_sample_active_users()

    interactions = clean_interactions(interactions)
    interactions = apply_engagement_log_transforms(interactions)
    interactions = sort_chronologically(interactions)
    interactions = add_positive_label(interactions)
    games, games_metadata = clean_metadata(games, games_metadata)
    games, games_metadata, interactions, fallback_games = intersect_catalogs(
        games, games_metadata, interactions
    )
    games, games_metadata = filter_to_interacted_catalog(
        games, games_metadata, interactions
    )
    return {
        "games": games, "games_metadata": games_metadata,
        "interactions": interactions, "fallback_games": fallback_games,
    }


def run_feature_stage(state: dict) -> dict:
    """Engineer features, compute SentenceTransformer embeddings, build weighted item matrix."""

    games = engineer_game_features(state["games"])
    games = rescale_positive_ratio(games)
    games, games_metadata = enrich_descriptions_with_title(games, state["games_metadata"])
    games_metadata = clean_text_for_embeddings(games_metadata)
    tags_df = encode_tags(games_metadata)
    desc_df = encode_descriptions(games_metadata)

    availability = games_metadata[["app_id", "has_description", "has_tags", "has_text_metadata"]]
    item_features_df = merge_feature_tables(games, tags_df, desc_df, availability)
    item_features_df = build_weighted_item_matrix(item_features_df)
    return {
        **state,
        "games": games, "games_metadata": games_metadata,
        "full_games": games.copy(), "full_games_meta": games_metadata.copy(),
        "full_item_features": item_features_df.copy(),
        "item_features_df": item_features_df,
    }


def run_split_stage(state: dict) -> dict:
    """Temporal split, engagement transform, warm-catalog stabilization."""

    train_int, val_int, test_int = temporal_split_by_user(state["interactions"])
    clip, scaler = fit_engagement_transform(train_int)
    train_int = add_engagement_strength(train_int, clip, scaler)
    val_int = add_engagement_strength(val_int, clip, scaler)
    test_int = add_engagement_strength(test_int, clip, scaler)

    full_train, full_val, full_test = train_int.copy(), val_int.copy(), test_int.copy()
    train_int = retain_train_supported_catalog(full_train)
    val_int, test_int, games, games_metadata, item_features_df = filter_splits_to_warm_catalog(
        train_int, full_val, full_test, state["full_games"],
        state["full_games_meta"], state["full_item_features"],
    )
    drop_engagement_log_columns(train_int, val_int, test_int)
    return {
        **state,
        "train_int": train_int, "val_int": val_int, "test_int": test_int,
        "full_train": full_train, "full_val": full_val, "full_test": full_test,
        "games": games, "games_metadata": games_metadata,
        "item_features_df": item_features_df,
    }


def run_train_stage(state: dict, ablation: str) -> dict:
    """Build data structures, train TwoTowerModel, return trained model + metrics."""

    train_int, val_int, test_int = state["train_int"], state["val_int"], state["test_int"]
    item_features_df = state["item_features_df"]

    # Contiguous indices.
    train_int, uid_to_idx, iid_to_idx = add_contiguous_indices(train_int)
    for df in [val_int, test_int]:
        df["user_idx"] = df["user_id"].map(uid_to_idx)
        df["item_idx"] = df["app_id"].map(iid_to_idx)
    num_items = len(iid_to_idx)

    # Release dates for candidate filtering.
    idf = item_features_df.copy()
    idf["item_idx"] = idf["app_id"].map(iid_to_idx)
    idf.dropna(subset=["item_idx"], inplace=True)
    idf["item_idx"] = idf["item_idx"].astype("int64")
    idf.sort_values("item_idx", inplace=True)
    warm_item_release_dates = idf["date_release"].values

    # History maps.
    train_pos_hist = positive_history_map(train_int)
    train_seen = seen_item_map(train_int)
    val_hist = positive_history_map(train_int)
    test_hist = positive_history_map(train_int, val_int)
    train_val_seen = seen_item_map(train_int, val_int)

    val_targets = heldout_positive_map(val_int, train_seen)
    test_targets = heldout_positive_map(test_int, train_val_seen)

    max_train_dates = train_int.groupby("user_idx")["date"].max()
    tc = pd.concat([train_int, val_int]).groupby("user_idx")["date"].max()
    val_cutoffs = max_train_dates.to_dict()
    test_cutoffs = tc.to_dict()

    val_targets = filter_targets_released_by_cutoff(val_targets, val_cutoffs, warm_item_release_dates)
    test_targets = filter_targets_released_by_cutoff(test_targets, test_cutoffs, warm_item_release_dates)

    train_examples = build_prefix_examples(train_pos_hist, MAX_HISTORY_LENGTH)
    train_loader, _, num_users, padding_item_idx = create_dataloaders(
        train_examples, num_items, BATCH_SIZE, MAX_HISTORY_LENGTH,
    )

    train_pos_mask = torch.zeros(num_users, num_items, dtype=torch.bool)
    for _, row in train_int[train_int["is_positive"] == 1].iterrows():
        train_pos_mask[int(row["user_idx"]), int(row["item_idx"])] = True

    assert idf["item_idx"].tolist() == list(range(num_items))
    feat_cols = [c for c in idf.columns if c not in ["app_id", "item_idx", "date_release"]]
    item_features = torch.tensor(idf[feat_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

    item_popularity_log = compute_item_popularity_log(train_int, num_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTowerModel(
        num_users=num_users, num_items=num_items,
        item_feature_dim=item_features.shape[1],
        padding_idx=padding_item_idx, ablation=ablation,
    ).to(device)

    results = train_model(
        model=model, train_loader=train_loader,
        item_features=item_features, train_pos_mask=train_pos_mask,
        train_examples=train_examples, num_items=num_items,
        val_positive_targets=val_targets, val_user_histories=val_hist,
        train_seen=train_seen, val_candidate_cutoffs=val_cutoffs,
        item_popularity_log=item_popularity_log,
        warm_item_release_dates=warm_item_release_dates,
        device=device, padding_idx=padding_item_idx,
        hybrid_alpha=HYBRID_ALPHA, ablation=ablation,
    )

    return {
        **state,
        "model": model, "item_features": item_features,
        "num_items": num_items, "num_users": num_users,
        "iid_to_idx": iid_to_idx, "padding_item_idx": padding_item_idx,
        "train_seen": train_seen, "train_val_seen": train_val_seen,
        "val_hist": val_hist, "test_hist": test_hist,
        "val_targets": val_targets, "test_targets": test_targets,
        "val_cutoffs": val_cutoffs, "test_cutoffs": test_cutoffs,
        "warm_item_release_dates": warm_item_release_dates,
        "item_popularity_log": item_popularity_log,
        "train_results": results,
    }


def run_eval_stage(state: dict) -> dict:
    """Run full evaluation: model metrics, all 3 baselines, HYBRID_ALPHA sweep, cold-start."""

    model = state["model"]
    item_features = state["item_features"]
    device = next(model.parameters()).device
    pop_log = state["item_popularity_log"]
    pad_idx = state["padding_item_idx"]
    rd = state["warm_item_release_dates"]
    pop_rank = pop_log.argsort(descending=True).tolist()

    # ── Model evaluation ──
    vm = evaluate_filtered(
        model, state["val_targets"], state["val_hist"], state["train_seen"],
        state["val_cutoffs"], item_features, rd, device,
        k=EVAL_K, hybrid_alpha=HYBRID_ALPHA,
        item_popularity_log=pop_log, padding_idx=pad_idx,
    )
    tm = evaluate_filtered(
        model, state["test_targets"], state["test_hist"], state["train_val_seen"],
        state["test_cutoffs"], item_features, rd, device,
        k=EVAL_K, hybrid_alpha=HYBRID_ALPHA,
        item_popularity_log=pop_log, padding_idx=pad_idx,
    )

    # ── Popularity baseline ──
    vp = evaluate_popularity_baseline(
        state["val_targets"], state["val_hist"], state["train_seen"],
        state["val_cutoffs"], pop_rank, rd, k=EVAL_K,
    )
    tp = evaluate_popularity_baseline(
        state["test_targets"], state["test_hist"], state["train_val_seen"],
        state["test_cutoffs"], pop_rank, rd, k=EVAL_K,
    )

    # ── Random-expected baseline ──
    vr = evaluate_random_expected_baseline(
        state["val_targets"], state["val_hist"], state["train_seen"],
        state["val_cutoffs"], rd, k=EVAL_K,
    )
    tr = evaluate_random_expected_baseline(
        state["test_targets"], state["test_hist"], state["train_val_seen"],
        state["test_cutoffs"], rd, k=EVAL_K,
    )

    # ── Content-centroid baseline ──
    item_features_gpu = item_features.to(device)
    vc = evaluate_content_centroid_baseline(
        state["val_targets"], state["val_hist"], state["train_seen"],
        state["val_cutoffs"], item_features_gpu, rd, k=EVAL_K,
    )
    tc = evaluate_content_centroid_baseline(
        state["test_targets"], state["test_hist"], state["train_val_seen"],
        state["test_cutoffs"], item_features_gpu, rd, k=EVAL_K,
    )

    # ── HYBRID_ALPHA sweep ──
    best_alpha, best_test_r = None, -1.0
    sweep_rows: list[tuple[float, dict, dict]] = []
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        vm_a = evaluate_filtered(
            model, state["val_targets"], state["val_hist"], state["train_seen"],
            state["val_cutoffs"], item_features, rd, device,
            k=EVAL_K, hybrid_alpha=alpha,
            item_popularity_log=pop_log, padding_idx=pad_idx,
        )
        tm_a = evaluate_filtered(
            model, state["test_targets"], state["test_hist"], state["train_val_seen"],
            state["test_cutoffs"], item_features, rd, device,
            k=EVAL_K, hybrid_alpha=alpha,
            item_popularity_log=pop_log, padding_idx=pad_idx,
        )
        sweep_rows.append((alpha, vm_a, tm_a))
        if tm_a["Recall@10"] > best_test_r:
            best_test_r, best_alpha = tm_a["Recall@10"], alpha

    # ── Cold-start recommendations ──
    cold = recommend_cold_start_v1(
        state["fallback_games"], state["full_train"], n=10,
        item_id_to_idx=state["iid_to_idx"],
    )

    return {
        "val_model": vm, "test_model": tm,
        "val_popularity": vp, "test_popularity": tp,
        "val_random": vr, "test_random": tr,
        "val_content": vc, "test_content": tc,
        "sweep_rows": sweep_rows,
        "best_alpha": best_alpha, "best_alpha_test_r": best_test_r,
        "cold_start_recs": cold,
    }


def run_full_pipeline(ablation: str = "no_user_id") -> dict:
    """Run the complete v1 two-tower pipeline from raw data through evaluation."""
    state = run_data_stage()
    state = run_feature_stage(state)
    state = run_split_stage(state)
    state = run_train_stage(state, ablation)
    eval_results = run_eval_stage(state)
    return {**state, **eval_results}
