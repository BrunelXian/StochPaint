from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stochpaint


def main() -> None:
    env = gym.make("StochPaint-v0")
    observation, info = env.reset(seed=42)
    total_reward = 0.0
    base_env = env.unwrapped

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(base_env.target_mask, cmap="gray")
    axes[0].set_title("Target Mask")
    axes[0].axis("off")

    axes[1].imshow(observation, cmap="viridis")
    axes[1].set_title(
        f"Final Coating\nreward={total_reward:.2f}, coverage={info['coverage_ratio']:.2%}"
    )
    axes[1].axis("off")

    plt.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = REPO_ROOT / "random_agent_result.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    plt.close(fig)

    print(f"Episode reward: {total_reward:.2f}")
    print(f"Coverage ratio: {info['coverage_ratio']:.4f}")
    print(f"Overspray cells: {info['overspray_cells']}")
    env.close()


if __name__ == "__main__":
    main()
