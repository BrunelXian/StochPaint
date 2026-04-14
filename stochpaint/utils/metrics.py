from __future__ import annotations

import numpy as np


def coverage_ratio(grid: np.ndarray, target_mask: np.ndarray) -> float:
    covered_target = np.count_nonzero((grid > 0.0) & target_mask)
    target_area = np.count_nonzero(target_mask)
    return float(covered_target / max(target_area, 1))


def overspray_ratio(grid: np.ndarray, target_mask: np.ndarray) -> float:
    coated_cells = np.count_nonzero(grid > 0.0)
    overspray_cells = np.count_nonzero((grid > 0.0) & ~target_mask)
    return float(overspray_cells / max(coated_cells, 1))


def uniformity_score(grid: np.ndarray, target_mask: np.ndarray) -> float:
    target_deposition = grid[target_mask]
    if target_deposition.size == 0:
        return 0.0

    mean_deposition = float(np.mean(target_deposition))
    if mean_deposition <= 0.0:
        return 0.0

    std_deposition = float(np.std(target_deposition))
    coefficient_of_variation = std_deposition / mean_deposition
    return float(1.0 / (1.0 + coefficient_of_variation))
