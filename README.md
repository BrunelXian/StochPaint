# StochPaint

**Stochastic Coating Environment for Reinforcement Learning**

A physically-inspired sandbox environment for studying **reinforcement learning under stochastic actuation and partial observability**.

StochPaint simulates a deposition process where an agent controls a **noisy brush** that deposits particles following stochastic Gaussian dynamics.  
The objective is to **coat a target shape uniformly while minimizing overspray and stacking**.

This environment is designed as a **benchmark for RL algorithms operating under uncertain control outcomes**.

---

# Motivation

Many real-world control problems involve **uncertain actuation** and **imperfect sensing**.

Examples include:

- additive manufacturing
- robotic coating
- thermal spraying
- powder deposition processes

In these systems:

- actions do not produce deterministic outcomes  
- sensors provide incomplete information  
- control policies must adapt to stochastic dynamics  

StochPaint provides a **controlled sandbox** to study these challenges.

---

# Key Idea

Instead of a deterministic brush, each stroke produces a **stochastic particle cloud**.

Particles are sampled from a Gaussian distribution centered at the commanded brush position.

This introduces **uncertain actuation**, meaning the agent controls the *distribution of outcomes* rather than precise placement.

---

# Environment Overview

The agent controls a brush and attempts to coat a target region.

Objectives:

- maximize coverage  
- minimize overspray  
- maintain uniform deposition  
- avoid stacking  

The agent only observes **coating states**, not perfect metrics, making the problem naturally suitable for **POMDP reinforcement learning**.

---

# Brush Model

Each brush stroke generates a stochastic particle cloud.

Particle locations are sampled from a Gaussian distribution centered at the action location.

Possible stochastic effects:

- covariance perturbation  
- center drift  
- particle count variation  

These effects simulate uncertainty found in real deposition processes.

---

# Reinforcement Learning Formulation

## Observation Space

Typical observations include:

- coating grid image
- residual coverage map
- previous actions
- optional sensor statistics

Example resolution:

64 × 64 coating grid


---

## Action Space

The default action space is continuous control of the brush position:

`(x, y)`

Optional extensions can include additional parameters:

- `brush_size`
- `intensity`
- `spray_duration`

These allow experimentation with different control complexities.

---

## Reward

A typical reward formulation encourages efficient and uniform coating.

Example:

reward =  
+ coverage_gain  
- overspray_penalty  
- stacking_penalty  
- variance_penalty  

Explanation:

- **coverage_gain** – reward for newly covered target area  
- **overspray_penalty** – penalty for particles outside the target region  
- **stacking_penalty** – penalty for excessive particle overlap  
- **variance_penalty** – penalty for non-uniform deposition  

This reward balances **efficiency, accuracy, and uniformity**.

---

# Benchmark Tasks

The environment supports multiple benchmark variants.

## Shape Tasks

Different target geometries:

- circle  
- star  
- polygon  
- complex masks  

These evaluate how agents handle **different spatial structures**.

---

## Noise Profiles

Different levels of stochasticity:

- low noise  
- high noise  
- drift disturbance  
- particle count variation  

These test **robustness under actuation uncertainty**.

---

## Difficulty Levels

Tasks can be scaled in complexity:

- easy  
- moderate  
- stochastic-heavy  

Higher difficulty introduces stronger noise and more complex shapes.

---

# Repository Structure

```

StochPaint
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── stochpaint
│ ├── env
│ │ ├── coating_env.py
│ │ ├── brush_model.py
│ │ ├── reward.py
│ │ └── observation.py
│ │
│ ├── physics
│ │ ├── gaussian_deposition.py
│ │ └── particle_sampler.py
│ │
│ ├── tasks
│ │ ├── shapes.py
│ │ ├── noise_profiles.py
│ │ └── benchmarks.py
│ │
│ └── utils
│ ├── rendering.py
│ └── metrics.py
│
├── examples
│ ├── random_agent.py
│ ├── train_ppo.py
│ └── train_sac.py
│
├── benchmarks
│ ├── evaluation_protocol.md
│ └── leaderboard.md
│
├── docs
│ ├── environment_design.md
│ ├── stochastic_brush.md
│ └── RL_formulation.md
│
└── assets
├── demo.gif
└── figures

```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourname/StochPaint.git
cd StochPaint
```

Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```
pip install -r requirements.txt
```
---

Quick Start

Run a simple random agent interacting with the environment:
```
python examples/random_agent.py
```
This will launch the environment and generate random brush strokes to demonstrate the stochastic coating process.

---

Training an RL Agent

Example training scripts are provided using common RL algorithms.

Train a PPO agent:
```
python examples/train_ppo.py
```
Train a SAC agent:
```
python examples/train_sac.py
```
These scripts use Gymnasium-compatible environments, making it easy to integrate with libraries such as:

- Stable-Baselines3
- RLlib
- CleanRL

---

StochPaint is designed as a benchmark environment for reinforcement learning under stochastic actuation.

Evaluation metrics include:

Coverage Score
Percentage of the target region successfully coated.
Overspray Ratio
Fraction of particles deposited outside the target area.
Uniformity Score
Measures coating thickness variance.
Sample Efficiency
Number of environment steps required to reach a target performance.

Detailed evaluation procedures are documented in:
benchmarks/evaluation_protocol.md

---

Example Tasks

The environment includes several predefined benchmark tasks.

Shape Tasks

Different target geometries:

  circle
  star
  polygon
  arbitrary masks
  Noise Profiles

Different stochastic brush behaviors:

  low noise
  high noise
  drift disturbance
  particle count variation
  Difficulty Levels

Tasks can be scaled by stochasticity and geometry complexity:

  easy
  moderate
  stochastic-heavy

These settings enable robustness and generalization evaluation.

---

Documentation

More detailed explanations of the environment design can be found in:

docs/environment_design.md
docs/stochastic_brush.md
docs/RL_formulation.md

These documents describe:

stochastic deposition modeling
environment mechanics
RL problem formulation
Contributing

---

Contributions are welcome.

Possible areas of contribution include:

new benchmark tasks
improved physics models
new RL baselines
visualization tools
evaluation metrics

Please open an issue or pull request to discuss proposed changes.

---

License

This project is released under the MIT License.

---

