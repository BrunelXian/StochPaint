from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stochpaint
from stable_baselines3 import PPO


ENV_ID = "StochPaint-v0"


def default_output_path(target_shape: str, noise_profile: str, seed: int | None) -> Path:
    seed_label = "none" if seed is None else str(seed)
    return REPO_ROOT / "models" / f"ppo_{target_shape}_{noise_profile}_seed{seed_label}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal PPO baseline on StochPaint.")
    parser.add_argument(
        "--target-shape",
        type=str,
        default="circle",
        choices=("circle", "square"),
        help="Target shape for training.",
    )
    parser.add_argument(
        "--noise-profile",
        type=str,
        default="low_noise",
        choices=("low_noise", "high_noise"),
        help="Noise profile for training.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional model output path without extension.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else default_output_path(
        args.target_shape, args.noise_profile, args.seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        ENV_ID,
        target_shape=args.target_shape,
        noise_profile=args.noise_profile,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(str(output_path))
    env.close()

    print(f"Saved PPO model to: {output_path}.zip")


if __name__ == "__main__":
    main()
