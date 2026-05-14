from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    PROCESSED_GAMES_CSV,
    PROCESSED_INTERACTIONS_CSV,
    TEST_SIMULATIONS_CSV,
    TRAIN_SIMULATIONS_CSV,
    USER_PROFILES_CSV,
)
from .evaluation import evaluate_recommendations
from .features import build_item_vectors, extract_features
from .loaders import (
    load_processed_games,
    load_processed_interactions,
    load_raw_datasets,
    load_test_simulations,
    load_train_simulations,
    load_user_profiles,
)
from .preprocessing import process_games_and_metadata, process_interactions
from .recommenders import build_recommendations, recommend_cold_start
from .simulation import train_test_users, user_profiles


def ensure_processed_dir(processed_dir: Path = PROCESSED_DATA_DIR) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def build_processed_artifacts(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    games, metadata, interactions = load_raw_datasets()
    processed_games, processed_metadata = process_games_and_metadata(games, metadata)
    processed_interactions = process_interactions(interactions)

    tags_vectors, description_vectors = extract_features(processed_metadata)
    item_vectors = build_item_vectors(processed_games, tags_vectors, description_vectors)

    if save:
        ensure_processed_dir()
        item_vectors.to_csv(PROCESSED_GAMES_CSV, index=False)
        processed_interactions.to_csv(PROCESSED_INTERACTIONS_CSV, index=False)

    return item_vectors, processed_interactions


def build_user_artifacts(
    games: pd.DataFrame,
    processed_interactions: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_simu_users, test_simu_users = train_test_users(games, processed_interactions)
    profiles = user_profiles(games, train_simu_users)

    if save:
        ensure_processed_dir()
        train_simu_users.to_csv(TRAIN_SIMULATIONS_CSV, index=False)
        test_simu_users.to_csv(TEST_SIMULATIONS_CSV, index=False)
        profiles.to_csv(USER_PROFILES_CSV, index=False)

    return train_simu_users, test_simu_users, profiles


def run_recommendation_stage(
    games: pd.DataFrame,
    train_simu_users: pd.DataFrame,
    test_simu_users: pd.DataFrame,
    profiles: pd.DataFrame,
) -> dict[str, object]:
    user_recs, user_item_similarities = build_recommendations(
        user_profiles=profiles,
        games=games,
        train_interactions=train_simu_users,
    )
    diagnostics, eval_results, eval_summary = evaluate_recommendations(
        user_recs=user_recs,
        test_interactions=test_simu_users,
    )
    cold_start_recs = recommend_cold_start(games, train_simu_users)

    return {
        "user_recs": user_recs,
        "user_item_similarities": user_item_similarities,
        "evaluation_diagnostics": diagnostics,
        "evaluation_results": eval_results,
        "evaluation_summary": eval_summary,
        "cold_start_recs": cold_start_recs,
    }


def run_from_processed_artifacts() -> dict[str, object]:
    games = load_processed_games()
    train_simu_users = load_train_simulations()
    test_simu_users = load_test_simulations()
    profiles = load_user_profiles()

    return run_recommendation_stage(
        games=games,
        train_simu_users=train_simu_users,
        test_simu_users=test_simu_users,
        profiles=profiles,
    )


def run_full_pipeline(save: bool = True) -> dict[str, object]:
    games, processed_interactions = build_processed_artifacts(save=save)
    train_simu_users, test_simu_users, profiles = build_user_artifacts(
        games=games,
        processed_interactions=processed_interactions,
        save=save,
    )
    recommendation_outputs = run_recommendation_stage(
        games=games,
        train_simu_users=train_simu_users,
        test_simu_users=test_simu_users,
        profiles=profiles,
    )

    return {
        "games": games,
        "processed_interactions": processed_interactions,
        "train_simu_users": train_simu_users,
        "test_simu_users": test_simu_users,
        "profiles": profiles,
        **recommendation_outputs,
    }
