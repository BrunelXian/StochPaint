# StochPaint
<h1 align="center">StochPaint</h1>

<p align="center"><strong>Stochastic Coating Environment for Reinforcement Learning</strong></p>
<p align="center"><strong>StochPaint is a stochastic deposition benchmark for studying reinforcement learning under uncertain actuation and partial observability.</strong></p>

[![Benchmark Type](https://img.shields.io/badge/benchmark-RL%20control-blue)](#)
[![Formulation](https://img.shields.io/badge/problem-POMDP-important)](#)
[![Domain Metaphor](https://img.shields.io/badge/domain-stochastic%20deposition-success)](#)

StochPaint is a physically inspired sandbox environment for studying reinforcement learning under stochastic actuation and partial observability.

The current runnable version provides a minimal Gymnasium-compatible coating environment with a stochastic brush, continuous `(x, y)` actions, a 2D coating-grid observation, and a simple random-agent example.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Create the environment through Gymnasium:

```python
import stochpaint
import gymnasium as gym

env = gym.make("StochPaint-v0")
observation, info = env.reset(seed=42)
```

Run the included random agent example:

```bash
python examples/random_agent.py
```

Run a minimal random-agent evaluation:

```bash
python benchmarks/evaluate_random.py
```

Run the evaluation and save a JSON result:

```bash
python benchmarks/evaluate_random.py --episodes 5 --seed 42 --output benchmark_results/random_eval.json
```

The environment can also be used directly via the package:

```python
from stochpaint import CoatingEnv

env = CoatingEnv()
```

## Current Environment

The current implementation includes:

- stochastic particle deposition around the commanded brush location
- continuous 2D action space
- grid-based coating observation
- simple reward based on coverage gain and overspray penalty
- Gymnasium environment registration as `StochPaint-v0`
- basic evaluation metrics: `coverage_ratio`, `overspray_ratio`, and `uniformity_score`

## One-Line Summary

**StochPaint is a stochastic deposition benchmark designed to study reinforcement learning under uncertain actuation and partial observability.**
