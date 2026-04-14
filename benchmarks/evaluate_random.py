from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stochpaint


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a random policy on StochPaint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of random-agent episodes to run.")
    args = parser.parse_args()

    env = gym.make("StochPaint-v0")
    metrics = {
        "coverage_ratio": [],
        "overspray_ratio": [],
        "uniformity_score": [],
    }

    for episode in range(args.episodes):
        episode_metrics = run_episode(env, seed=episode)
        for key, value in episode_metrics.items():
            metrics[key].append(value)

    env.close()

    print(f"Random-agent evaluation over {args.episodes} episodes")
    for key, values in metrics.items():
        mean_value, std_value = mean_std(values)
        print(f"{key}: mean={mean_value:.4f}, std={std_value:.4f}")


if __name__ == "__main__":
    main()
