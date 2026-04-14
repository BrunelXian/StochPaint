from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stochpaint

ENV_ID = "StochPaint-v0"
METRIC_KEYS = ("coverage_ratio", "overspray_ratio", "uniformity_score")


PolicyFn = Callable[[gym.Env, np.ndarray, dict[str, float], int], np.ndarray]


def random_policy(
    env: gym.Env, observation: np.ndarray, info: dict[str, float], step_index: int
) -> np.ndarray:
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def make_spiral_heuristic_policy(env: gym.Env) -> PolicyFn:
    base_env = env.unwrapped
    target_points = np.argwhere(base_env.target_mask)
    center_yx = np.mean(target_points, axis=0)
    center_y, center_x = float(center_yx[0]), float(center_yx[1])
    distances = np.sqrt(
        (target_points[:, 1] - center_x) ** 2 + (target_points[:, 0] - center_y) ** 2
    )
    max_radius = 0.85 * float(np.max(distances)) / max(base_env.grid_size - 1, 1)
    center = np.array(
        [
            center_x / max(base_env.grid_size - 1, 1),
            center_y / max(base_env.grid_size - 1, 1),
        ],
        dtype=np.float32,
    )
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    total_steps = max(int(base_env.episode_length), 1)

    def policy(
        env: gym.Env, observation: np.ndarray, info: dict[str, float], step_index: int
    ) -> np.ndarray:
        progress = (step_index + 0.5) / total_steps
        radius = max_radius * np.sqrt(progress)
        theta = step_index * golden_angle
        action = center + np.array(
            [radius * np.cos(theta), radius * np.sin(theta)],
            dtype=np.float32,
        )
        return np.clip(action, 0.0, 1.0)

    return policy


def run_episode(
    env: gym.Env, policy: PolicyFn, seed: int | None = None
) -> dict[str, float]:
    observation, info = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)
    terminated = False
    truncated = False
    step_index = 0

    while not (terminated or truncated):
        action = policy(env, observation, info, step_index)
        observation, reward, terminated, truncated, info = env.step(action)
        step_index += 1

    return {key: float(info[key]) for key in METRIC_KEYS}


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


def evaluate_baseline(
    env: gym.Env,
    baseline_name: str,
    policy_factory: Callable[[gym.Env], PolicyFn],
    episodes: int,
    seed: int | None,
) -> dict[str, object]:
    policy = policy_factory(env)
    metrics = {key: [] for key in METRIC_KEYS}

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        episode_metrics = run_episode(env, policy=policy, seed=episode_seed)
        for key, value in episode_metrics.items():
            metrics[key].append(value)

    return {
        "baseline": baseline_name,
        "aggregated_metrics": build_aggregated_metrics(metrics),
    }


def build_comparison(
    baseline_results: dict[str, dict[str, object]]
) -> dict[str, dict[str, float]]:
    if "random" not in baseline_results or "heuristic" not in baseline_results:
        return {}

    random_metrics = baseline_results["random"]["aggregated_metrics"]
    heuristic_metrics = baseline_results["heuristic"]["aggregated_metrics"]
    return {
        "heuristic_minus_random": {
            key: float(heuristic_metrics[key]["mean"] - random_metrics[key]["mean"])
            for key in METRIC_KEYS
        }
    }


def print_summary(
    baseline_results: dict[str, dict[str, object]], episodes: int, seed: int | None
) -> None:
    print(f"Baseline evaluation over {episodes} episodes")
    if seed is not None:
        print(f"Base seed: {seed}")

    for baseline_name, result in baseline_results.items():
        print(f"\n[{baseline_name}]")
        aggregated_metrics = result["aggregated_metrics"]
        for key in METRIC_KEYS:
            stats = aggregated_metrics[key]
            print(f"{key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    comparison = build_comparison(baseline_results)
    if comparison:
        print("\n[comparison]")
        for key, delta in comparison["heuristic_minus_random"].items():
            print(f"{key}: heuristic-random={delta:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate minimal baseline policies on StochPaint."
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="circle",
        choices=("circle", "square"),
        help="Target shape to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run per baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducible episode rollouts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for saving aggregated results.",
    )
    args = parser.parse_args()

    env = gym.make(ENV_ID, target_shape=args.target_shape)
    baseline_results = {
        "random": evaluate_baseline(
            env,
            baseline_name="random",
            policy_factory=lambda current_env: random_policy,
            episodes=args.episodes,
            seed=args.seed,
        ),
        "heuristic": evaluate_baseline(
            env,
            baseline_name="heuristic",
            policy_factory=make_spiral_heuristic_policy,
            episodes=args.episodes,
            seed=args.seed,
        ),
    }
    env.close()

    result = {
        "env_id": ENV_ID,
        "target_shape": args.target_shape,
        "episodes": args.episodes,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baselines": baseline_results,
        "comparison": build_comparison(baseline_results),
    }
    maybe_write_output(args.output, result)

    print_summary(baseline_results, episodes=args.episodes, seed=args.seed)
    if args.output:
        print(f"Saved JSON results to: {Path(args.output)}")


if __name__ == "__main__":
    main()
