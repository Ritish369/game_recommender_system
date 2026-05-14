from __future__ import annotations

import argparse

from .loaders import load_processed_games, load_processed_interactions
from .pipeline import (
    build_processed_artifacts,
    build_user_artifacts,
    run_from_processed_artifacts,
    run_full_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cleaned v0 MVP recommender pipeline.")
    parser.add_argument(
        "--stage",
        choices=["processed", "users", "recommend", "full"],
        default="recommend",
        help="Pipeline stage to run. Default uses existing processed artifacts for recommendation/evaluation.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run artifact-building stages without writing CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save = not args.no_save

    if args.stage == "processed":
        games, interactions = build_processed_artifacts(save=save)
        print({"processed_games_shape": games.shape, "processed_interactions_shape": interactions.shape})
        return

    if args.stage == "users":
        games = load_processed_games()
        interactions = load_processed_interactions()
        train, test, profiles = build_user_artifacts(games, interactions, save=save)
        print({"train_shape": train.shape, "test_shape": test.shape, "profiles_shape": profiles.shape})
        return

    if args.stage == "full":
        outputs = run_full_pipeline(save=save)
    else:
        outputs = run_from_processed_artifacts()

    print(outputs["evaluation_diagnostics"].to_dict())
    print(outputs["evaluation_summary"].to_dict()["score"])
    print({"cold_start_rows": len(outputs["cold_start_recs"])})


if __name__ == "__main__":
    main()
