from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stochpaint.utils.metrics import coverage_ratio, overspray_ratio, uniformity_score


@dataclass(frozen=True)
class BrushConfig:
    sigma: float = 1.75
    particles_per_step: int = 96
    overspray_penalty: float = 0.02


class CoatingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        grid_size: int = 64,
        episode_length: int = 100,
        target_radius_ratio: float = 0.28,
        brush_config: BrushConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.brush = brush_config or BrushConfig()

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        self.target_mask = self._build_circle_mask(target_radius_ratio)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.steps = 0
        self.rng = np.random.default_rng()

    def _build_circle_mask(self, radius_ratio: float) -> np.ndarray:
        center = (self.grid_size - 1) / 2.0
        radius = self.grid_size * radius_ratio
        y_idx, x_idx = np.indices((self.grid_size, self.grid_size), dtype=np.float32)
        distances = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)
        return distances <= radius

    def _action_to_grid_position(self, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(action, 0.0, 1.0)
        return clipped * (self.grid_size - 1)

    def _deposit(self, action: np.ndarray) -> tuple[int, int]:
        center_x, center_y = self._action_to_grid_position(action)
        samples = self.rng.normal(
            loc=(center_x, center_y),
            scale=self.brush.sigma,
            size=(self.brush.particles_per_step, 2),
        )
        indices = np.rint(samples).astype(np.int32)

        in_bounds = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.grid_size)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.grid_size)
        )
        valid = indices[in_bounds]
        if valid.size == 0:
            return 0, 0

        x_coords = valid[:, 0]
        y_coords = valid[:, 1]
        was_covered = self.grid[y_coords, x_coords] > 0.0
        on_target = self.target_mask[y_coords, x_coords]
        self.grid[y_coords, x_coords] += 1.0

        coverage_gain = int(np.count_nonzero((~was_covered) & on_target))
        overspray = int(np.count_nonzero(~on_target))
        return coverage_gain, overspray

    def _get_observation(self) -> np.ndarray:
        return self.grid.copy()

    def _get_info(self) -> dict[str, Any]:
        metrics = {
            "coverage_ratio": coverage_ratio(self.grid, self.target_mask),
            "overspray_ratio": overspray_ratio(self.grid, self.target_mask),
            "uniformity_score": uniformity_score(self.grid, self.target_mask),
        }
        overspray_cells = np.count_nonzero((self.grid > 0.0) & ~self.target_mask)
        return {
            "metrics": metrics,
            **metrics,
            "overspray_cells": overspray_cells,
            "steps": self.steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.grid.fill(0.0)
        self.steps = 0
        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        coverage_gain, overspray = self._deposit(action)

        reward = float(coverage_gain - self.brush.overspray_penalty * overspray)

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.episode_length
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray:
        norm_grid = self.grid / max(np.max(self.grid), 1.0)
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        image[..., 0] = np.where(self.target_mask, 0.15, 0.05)
        image[..., 1] = np.where(self.target_mask, 0.20, 0.05)
        image[..., 2] = np.where(self.target_mask, 0.35, 0.05)

        image[..., 1] += 0.75 * norm_grid
        image[..., 2] += 0.25 * norm_grid
        return np.clip(image, 0.0, 1.0)
