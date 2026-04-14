from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stochpaint

ENV_ID = "StochPaint-v0"


def run_episode(env: gym.Env, seed: int | None = None) -> dict[str, float]:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

    return {
        "coverage_ratio": float(info["coverage_ratio"]),
        "overspray_ratio": float(info["overspray_ratio"]),
        "uniformity_score": float(info["uniformity_score"]),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def build_aggregated_metrics(metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for key, values in metrics.items():
        mean_value, std_value = mean_std(values)
        aggregated[key] = {
            "mean": mean_value,
            "std": std_value,
        }
    return aggregated


def maybe_write_output(output_path: str | None, payload: dict[str, object]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a random policy on StochPaint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of random-agent episodes to run.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducible episode rollouts.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for saving aggregated results.")
    args = parser.parse_args()

    env = gym.make(ENV_ID)
    metrics = {
        "coverage_ratio": [],
        "overspray_ratio": [],
        "uniformity_score": [],
    }

    for episode in range(args.episodes):
        episode_seed = None if args.seed is None else args.seed + episode
        episode_metrics = run_episode(env, seed=episode_seed)
        for key, value in episode_metrics.items():
            metrics[key].append(value)

    env.close()

    aggregated_metrics = build_aggregated_metrics(metrics)
    result = {
        "env_id": ENV_ID,
        "episodes": args.episodes,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated_metrics": aggregated_metrics,
    }
    maybe_write_output(args.output, result)

    print(f"Random-agent evaluation over {args.episodes} episodes")
    if args.seed is not None:
        print(f"Base seed: {args.seed}")
    for key, stats in aggregated_metrics.items():
        print(f"{key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    if args.output:
        print(f"Saved JSON results to: {Path(args.output)}")


if __name__ == "__main__":
    main()
