from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.train import DEFAULT_RUNS, parse_args as parse_train_args, run as run_training
else:
    from .train import DEFAULT_RUNS, parse_args as parse_train_args, run as run_training


@dataclass(frozen=True)
class TrialConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int


def build_search_space(
    learning_rates: Iterable[float] = (2e-5,),
    batch_sizes: Iterable[int] = (8,),
    epochs: Iterable[int] = (1,),
    max_lengths: Iterable[int] = (128,),
) -> list[TrialConfig]:
    from itertools import product

    return [
        TrialConfig(lr, batch, epoch_count, length)
        for lr, batch, epoch_count, length in product(
            learning_rates, batch_sizes, epochs, max_lengths
        )
    ]


def default_grid(trial_epochs: int) -> list[TrialConfig]:
    """Small, explicit grid: tuning is a visible, reproducible step, not a hidden sweep."""
    return [
        TrialConfig(learning_rate=2e-5, batch_size=32, epochs=trial_epochs, max_length=256),
        TrialConfig(learning_rate=3e-5, batch_size=32, epochs=trial_epochs, max_length=256),
        TrialConfig(learning_rate=4e-5, batch_size=64, epochs=trial_epochs, max_length=256),
        TrialConfig(learning_rate=2e-5, batch_size=32, epochs=trial_epochs, max_length=128),
    ]


def choose_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        result
        for result in results
        if result.get("best_val_f1") is not None
        and float(result["best_val_f1"]) == float(result["best_val_f1"])
    ]
    if not valid:
        raise ValueError("No valid tuning results were produced")
    return max(
        valid,
        key=lambda result: (
            float(result["best_val_f1"]),
            -float(result.get("duration_seconds") or 0.0),
        ),
    )


def run_search(
    configs: Iterable[TrialConfig],
    run_trial: Callable[[TrialConfig, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [run_trial(config, index) for index, config in enumerate(configs, start=1)]


def run_trial(config: TrialConfig, index: int, args: argparse.Namespace) -> dict[str, Any]:
    run_name = f"{args.run_prefix}-t{index}"
    argv = [
        "--run-name", run_name,
        "--runs-dir", str(args.runs_dir),
        "--seed", str(args.seed),
        "--learning-rate", str(config.learning_rate),
        "--batch-size", str(config.batch_size),
        "--epochs", str(config.epochs),
        "--max-length", str(config.max_length),
        "--patience", str(args.trial_patience),
        "--monitor", args.monitor,
        "--skip-test",
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    started = time.time()
    result = run_training(parse_train_args(argv))
    best = result.get("best_val") or {}
    return {
        "trial": index,
        "run_name": run_name,
        **asdict(config),
        "best_val_epoch": best.get("epoch"),
        "best_val_f1": best.get("val_f1"),
        "best_val_loss": best.get("val_loss"),
        "best_val_accuracy": best.get("val_accuracy"),
        "duration_seconds": round(time.time() - started, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search hyperparameters on the validation split")
    parser.add_argument("--execute", action="store_true", help="Actually run trials (otherwise only print the grid)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", default="val_f1", choices=["val_f1", "val_accuracy", "val_loss"])
    parser.add_argument("--trial-epochs", type=int, default=4)
    parser.add_argument("--trial-patience", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--final-patience", type=int, default=5)
    parser.add_argument("--run-prefix", default="tune")
    parser.add_argument("--final-run-name", default="distilbert-final")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = default_grid(args.trial_epochs)
    if not args.execute:
        print(json.dumps({"grid": [asdict(config) for config in configs]}, indent=2))
        print("Add --execute to run the trials and the final model.")
        return

    results = run_search(configs, lambda config, index: run_trial(config, index, args))
    best = choose_best(results)
    summary: dict[str, Any] = {
        "selection_metric": args.monitor,
        "seed": args.seed,
        "trials": results,
        "best": best,
    }

    final_argv = [
        "--run-name", args.final_run_name,
        "--runs-dir", str(args.runs_dir),
        "--seed", str(args.seed),
        "--learning-rate", str(best["learning_rate"]),
        "--batch-size", str(best["batch_size"]),
        "--max-length", str(best["max_length"]),
        "--epochs", str(args.final_epochs),
        "--patience", str(args.final_patience),
        "--monitor", args.monitor,
    ]
    if args.limit is not None:
        final_argv += ["--limit", str(args.limit)]
    final = run_training(parse_train_args(final_argv))
    summary["final"] = {
        key: final.get(key)
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "checkpoint",
            "best_val",
            "length_buckets",
            "length_only_auc",
            "cleaning",
            "leakage",
        )
    }
    summary_path = args.runs_dir / f"{args.run_prefix}-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
