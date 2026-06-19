# main.py — CLI entry point for v1 two-tower pipeline.
# Thin CLI wrapper — all stage logic lives in pipeline.py.

from __future__ import annotations

import argparse
import json

from .config import ABLATION_RESULTS_PATH
from .pipeline import (
    run_data_stage,
    run_eval_stage,
    run_feature_stage,
    run_full_pipeline,
    run_split_stage,
    run_train_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the v1 two-tower game recommender pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["data", "features", "splits", "train", "evaluate", "full"],
        default="full",
        help="Pipeline stage to run. full runs the complete pipeline.",
    )
    parser.add_argument(
        "--ablation",
        choices=["no_user_id", "full", "no_popularity", "mean_pool"],
        default="no_user_id",
        help="Ablation variant (overrides config.ABLATION).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation = args.ablation

    if args.stage == "full":
        results = run_full_pipeline(ablation)
        print_results(results)
        return

    state: dict = {}

    if args.stage == "data":
        state = run_data_stage()

    if args.stage == "features":
        if not state:
            state = run_data_stage()
        state = run_feature_stage(state)

    if args.stage == "splits":
        if not state:
            state = run_data_stage()
            state = run_feature_stage(state)
        state = run_split_stage(state)

    if args.stage == "train":
        if not state:
            state = run_data_stage()
            state = run_feature_stage(state)
            state = run_split_stage(state)
        state = run_train_stage(state, ablation)

    if args.stage == "evaluate":
        if not state:
            state = run_data_stage()
            state = run_feature_stage(state)
            state = run_split_stage(state)
            state = run_train_stage(state, ablation)
        results = run_eval_stage(state)
        print_results(results)


def print_results(results: dict) -> None:
    """Print full evaluation: comparison table, HYBRID_ALPHA sweep, ablation history."""
    k = 10  # EVAL_K

    # ── Comparison table ──
    print()
    print("=" * 90)
    print(f"{'COMPARISON TABLE':^90}")
    print("=" * 90)
    print(f"{'Method':<18} {'Val HR@10':>11} {'Val R@10':>10} {'Test HR@10':>12} {'Test R@10':>11}")
    print("-" * 90)

    methods = [
        ("Two-Tower", results["val_model"], results["test_model"]),
        ("Popularity", results["val_popularity"], results["test_popularity"]),
        ("Random", results["val_random"], results["test_random"]),
        ("Content", results["val_content"], results["test_content"]),
    ]
    hk = f"HitRate@{k}"
    rk = f"Recall@{k}"
    for name, vm, tm_ in methods:
        print(f"{name:<18} {vm[hk]:>11.4f} {vm[rk]:>10.4f} {tm_[hk]:>12.4f} {tm_[rk]:>11.4f}")
    print("=" * 90)

    # ── HYBRID_ALPHA sweep ──
    print()
    print(f"{'Alpha':>6} {'Val R@10':>10} {'Test R@10':>11}")
    print("-" * 32)
    best_a = results.get("best_alpha")
    for alpha, vm_a, tm_a in results["sweep_rows"]:
        marker = " <--" if alpha == best_a else ""
        print(f"{alpha:>6.1f} {vm_a[rk]:>10.4f} {tm_a[rk]:>11.4f}{marker}")
    print(f"\nBest alpha = {best_a} (test R@10 = {results['best_alpha_test_r']:.4f})")

    # ── Cold-start ──
    print(f"\nCold-start recommendations: {len(results['cold_start_recs'])} items")

    # ── Ablation history ──
    if ABLATION_RESULTS_PATH.exists():
        print()
        all_res = json.loads(ABLATION_RESULTS_PATH.read_text())
        print(f"{'Ablation':<16} {'Best Ep':>8} {'Best Val R@10':>14} {'Time(min)':>10}")
        print("-" * 52)
        for r in sorted(all_res, key=lambda x: x.get("best_val_recall", 0), reverse=True):
            print(f"{r.get('ablation','?'):<16} {str(r.get('best_epoch','?')):>8} {r.get('best_val_recall',0):>14.4f} {r.get('total_train_time_s',0)/60:>9.1f}")


if __name__ == "__main__":
    main()
