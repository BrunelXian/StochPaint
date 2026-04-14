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
SUPPORTED_TARGET_SHAPES = ("circle", "square")
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
    env_factory: Callable[[], gym.Env],
    baseline_name: str,
    policy_factory: Callable[[gym.Env], PolicyFn],
    episodes: int,
    seed: int | None,
) -> dict[str, object]:
    metrics = {key: [] for key in METRIC_KEYS}

    for episode in range(episodes):
        env = env_factory()
        policy = policy_factory(env)
        episode_seed = None if seed is None else seed + episode
        episode_metrics = run_episode(env, policy=policy, seed=episode_seed)
        env.close()
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


def build_per_baseline_results(
    per_shape_results: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    per_baseline: dict[str, dict[str, object]] = {}

    for shape, shape_result in per_shape_results.items():
        for baseline_name, baseline_result in shape_result["baselines"].items():
            entry = per_baseline.setdefault(
                baseline_name,
                {"baseline": baseline_name, "shapes": {}},
            )
            entry["shapes"][shape] = baseline_result["aggregated_metrics"]

    return per_baseline


def build_cross_shape_comparisons(
    per_baseline_results: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    comparisons: dict[str, dict[str, object]] = {}

    if len(SUPPORTED_TARGET_SHAPES) < 2:
        return comparisons

    first_shape, second_shape = SUPPORTED_TARGET_SHAPES[:2]
    comparison_key = f"{second_shape}_minus_{first_shape}"

    for baseline_name, baseline_result in per_baseline_results.items():
        shape_metrics = baseline_result["shapes"]
        if first_shape not in shape_metrics or second_shape not in shape_metrics:
            continue

        comparisons[baseline_name] = {
            comparison_key: {
                key: float(
                    shape_metrics[second_shape][key]["mean"]
                    - shape_metrics[first_shape][key]["mean"]
                )
                for key in METRIC_KEYS
            }
        }

    return comparisons


def evaluate_shape(
    target_shape: str, episodes: int, seed: int | None
) -> dict[str, object]:
    env_factory = lambda: gym.make(ENV_ID, target_shape=target_shape)
    baseline_results = {
        "random": evaluate_baseline(
            env_factory=env_factory,
            baseline_name="random",
            policy_factory=lambda current_env: random_policy,
            episodes=episodes,
            seed=seed,
        ),
        "heuristic": evaluate_baseline(
            env_factory=env_factory,
            baseline_name="heuristic",
            policy_factory=make_spiral_heuristic_policy,
            episodes=episodes,
            seed=seed,
        ),
    }
    return {
        "target_shape": target_shape,
        "baselines": baseline_results,
        "comparison": build_comparison(baseline_results),
    }


def print_summary(
    per_shape_results: dict[str, dict[str, object]],
    episodes: int,
    seed: int | None,
) -> None:
    print(f"Baseline evaluation over {episodes} episodes")
    if seed is not None:
        print(f"Base seed: {seed}")

    for shape, shape_result in per_shape_results.items():
        print(f"\n[{shape}]")
        for baseline_name in ("random", "heuristic"):
            aggregated_metrics = shape_result["baselines"][baseline_name]["aggregated_metrics"]
            coverage = aggregated_metrics["coverage_ratio"]["mean"]
            overspray = aggregated_metrics["overspray_ratio"]["mean"]
            uniformity = aggregated_metrics["uniformity_score"]["mean"]
            print(
                f"{baseline_name}: "
                f"coverage={coverage:.4f}, "
                f"overspray={overspray:.4f}, "
                f"uniformity={uniformity:.4f}"
            )

        comparison = shape_result["comparison"]
        if comparison:
            delta = comparison["heuristic_minus_random"]
            print(
                "delta(heuristic-random): "
                f"coverage={delta['coverage_ratio']:+.4f}, "
                f"overspray={delta['overspray_ratio']:+.4f}, "
                f"uniformity={delta['uniformity_score']:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate minimal baseline policies on StochPaint."
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="circle",
        choices=SUPPORTED_TARGET_SHAPES + ("all",),
        help="Target shape to evaluate, or 'all' to run every supported shape.",
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

    target_shapes = (
        list(SUPPORTED_TARGET_SHAPES)
        if args.target_shape == "all"
        else [args.target_shape]
    )
    per_shape_results = {
        shape: evaluate_shape(shape, episodes=args.episodes, seed=args.seed)
        for shape in target_shapes
    }
    per_baseline_results = build_per_baseline_results(per_shape_results)
    comparison_summaries = {
        "within_shape": {
            shape: shape_result["comparison"] for shape, shape_result in per_shape_results.items()
        },
        "across_shapes": build_cross_shape_comparisons(per_baseline_results),
    }

    result = {
        "metadata": {
            "env_id": ENV_ID,
            "requested_target_shape": args.target_shape,
            "evaluated_target_shapes": target_shapes,
            "episodes": args.episodes,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "supported_target_shapes": list(SUPPORTED_TARGET_SHAPES),
        "per_shape_results": per_shape_results,
        "per_baseline_results": per_baseline_results,
        "comparison_summaries": comparison_summaries,
    }
    maybe_write_output(args.output, result)

    print_summary(per_shape_results, episodes=args.episodes, seed=args.seed)
    if args.output:
        print(f"Saved JSON results to: {Path(args.output)}")


if __name__ == "__main__":
    main()
