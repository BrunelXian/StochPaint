# StochPaint
<h1 align="center">StochPaint</h1>

<p align="center"><strong>Stochastic Coating Environment for Reinforcement Learning</strong></p>
<p align="center"><strong>StochPaint is a stochastic deposition benchmark for studying reinforcement learning under uncertain actuation and partial observability.</strong></p>

[![Benchmark Type](https://img.shields.io/badge/benchmark-RL%20control-blue)](#)
[![Formulation](https://img.shields.io/badge/problem-POMDP-important)](#)
[![Domain Metaphor](https://img.shields.io/badge/domain-stochastic%20deposition-success)](#)

StochPaint is a physically inspired sandbox environment for studying reinforcement learning under stochastic actuation and partial observability.

The environment simulates a deposition process in which an agent controls a noisy brush that deposits particles according to stochastic Gaussian dynamics. The goal is to coat a target shape uniformly while minimizing overspray, stacking, and non-uniform buildup.

## Project Positioning

StochPaint is designed as a benchmark for RL algorithms operating under uncertain control outcomes. It is not a game environment. It is intended as a research-oriented benchmark for:

- stochastic deposition control
- partial observability and POMDP learning
- closed-loop policy learning under uncertainty
- multi-objective tradeoffs between quality, waste, and uniformity

一句话总结：**StochPaint = stochastic deposition control benchmark**.

## Motivation

Many real-world control problems involve uncertain actuation and imperfect sensing.

Examples include:

- additive manufacturing
- robotic coating
- thermal spraying
- powder deposition processes
- closed-loop control under stochastic execution error
- policy learning under partial observability

In these systems:

- actions do not produce deterministic outcomes
- sensors provide incomplete information
- control policies must adapt to stochastic dynamics

StochPaint provides a controlled sandbox for studying these challenges in a reproducible way.

## Why This Project Exists

Many RL benchmarks emphasize game score or idealized robotic dynamics. Industrial control often looks different:

- actuation is stochastic
- sensing is partial
- performance is multi-objective

StochPaint is built around this setting and aims to support benchmark-style evaluation rather than one-off demonstrations.

## Core Idea

Instead of a deterministic brush, each action produces a stochastic particle cloud:

\[
(x, y) \sim \mathcal{N}(\mu, \Sigma)
\]

Here:

- `\mu` is the commanded brush center
- `\Sigma` controls brush spread and shape

This means the agent controls the distribution of outcomes rather than exact placement. That makes the environment a natural benchmark for robust decision-making under uncertainty.

## Environment Overview

The agent controls a brush and attempts to coat a target region.

Primary objectives:

- maximize target coverage
- minimize overspray
- maintain uniform deposition
- avoid excessive stacking

The agent observes coating-related process state, but not full latent metrics, which makes the problem naturally suitable for POMDP reinforcement learning.

## Brush Model

Each brush stroke generates a stochastic particle cloud centered near the commanded action location.

Possible stochastic effects include:

- covariance perturbation
- center drift
- particle count variation

These effects are intended to mimic uncertainty found in real deposition processes.

## RL Formulation

### Observation Space

Typical observations may include:

- coating grid image
- residual coverage map
- previous actions
- optional brush or sensor statistics

Typical resolution:

- `64 x 64` coating grid

The agent does not receive direct access to the full hidden process state.

### Action Space

The default action space is continuous brush positioning:

`(x, y)`

Optional extensions may include:

- `brush_size`
- `intensity`
- `spray_duration`

This allows the benchmark to scale from simple control to richer process parameter optimization.

### Reward

A representative multi-objective reward is:

\[
R = w_1 \cdot \text{coverage} - w_2 \cdot \text{overspray} - w_3 \cdot \text{stacking} - w_4 \cdot \text{variance}
\]

Interpretation:

- `coverage_gain`: reward for newly covered target area
- `overspray_penalty`: penalty for out-of-target deposition
- `stacking_penalty`: penalty for excessive overlap
- `variance_penalty`: penalty for non-uniform coating

This balances efficiency, accuracy, and uniformity.

## Industrial Analogy

| StochPaint concept | Industrial interpretation |
| --- | --- |
| stochastic brush deposition | powder, coating, or thermal spray uncertainty |
| coverage control | process input control |
| overspray | material waste |
| stacking or non-uniform buildup | thickness inconsistency or thermal instability |
| residual uncovered area | defects |

Relevant application metaphors include additive manufacturing, robotic coating, and spray-based process control.

## Benchmark Task Suite

### 1. Shape Generalization

Target geometries may include:

- circle
- star
- polygon
- complex masks

These tasks evaluate how agents handle different spatial structures.

### 2. Noise Robustness

Noise settings may include:

- low noise
- high noise
- drift disturbance
- particle count variation

These tasks test robustness under actuation uncertainty.

### 3. Difficulty Scaling

Benchmark variants can be grouped into:

- easy
- moderate
- stochastic-heavy

Higher difficulty introduces stronger noise and more complex geometry.

### 4. Process Variants

The framework is extensible to process-level variations such as:

- spread or adhesion variants
- layer diffusion heuristics
- custom deposition kernels

## Baseline Algorithms

Recommended baseline algorithms include:

- PPO
- SAC
- TD3

The environment is intended to be Gymnasium-compatible so that it can be used with common RL libraries such as:

- Stable-Baselines3
- RLlib
- CleanRL

Future research directions may also include transformer policies, world models, and planning-based agents.

## Evaluation Metrics

Typical evaluation metrics include:

- Coverage Score: fraction of the target region successfully coated
- Overspray Ratio: fraction of deposited material outside the target area
- Uniformity Score: variance-based measurement of coating thickness consistency
- Sample Efficiency: number of environment steps required to reach a target performance level

Detailed protocols are intended to live in `benchmarks/evaluation_protocol.md`.

## Planned Repository Layout

```text
StochPaint/
├── README.md
├── LICENSE
├── requirements.txt
├── stochpaint/
│   ├── env/
│   │   ├── coating_env.py
│   │   ├── brush_model.py
│   │   ├── reward.py
│   │   └── observation.py
│   ├── physics/
│   │   ├── gaussian_deposition.py
│   │   └── particle_sampler.py
│   ├── tasks/
│   │   ├── shapes.py
│   │   ├── noise_profiles.py
│   │   └── benchmarks.py
│   └── utils/
│       ├── rendering.py
│       └── metrics.py
├── examples/
│   ├── random_agent.py
│   ├── train_ppo.py
│   └── train_sac.py
├── benchmarks/
│   ├── evaluation_protocol.md
│   └── leaderboard.md
├── docs/
│   ├── environment_design.md
│   ├── stochastic_brush.md
│   └── RL_formulation.md
└── assets/
    ├── demo.gif
    └── figures/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/BrunelXian/StochPaint.git
cd StochPaint
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a simple random agent:

```bash
python examples/random_agent.py
```

Train a PPO agent:

```bash
python examples/train_ppo.py
```

Train a SAC agent:

```bash
python examples/train_sac.py
```

## Documentation

More detailed design notes are intended to live in:

- `docs/environment_design.md`
- `docs/stochastic_brush.md`
- `docs/RL_formulation.md`

These documents can cover:

- stochastic deposition modeling
- environment mechanics
- RL problem formulation

## Positioning

StochPaint can support at least three research directions:

1. environment benchmark work for RL and POMDP evaluation
2. algorithmic work on robust RL under stochastic actuation
3. digital twin or manufacturing control transfer studies

## Contributing

Contributions are welcome. Possible directions include:

- new benchmark tasks
- improved physics models
- additional RL baselines
- visualization tools
- new evaluation metrics

Please open an issue or pull request to discuss proposed changes.

## License

This project is released under the MIT License.

## One-Line Summary

**StochPaint is a stochastic deposition benchmark designed to study reinforcement learning under uncertain actuation and partial observability.**
