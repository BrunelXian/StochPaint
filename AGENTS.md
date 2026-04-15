# AGENTS.md

## Project Stage

StochPaint is currently a minimal runnable benchmark prototype built on Gymnasium.
It includes target-shape variants, noise-profile variants, lightweight random and heuristic baselines, and an initial PPO training/evaluation path.
It is not yet a full benchmark suite.

## Working Style

- Make small, reviewable changes.
- Prefer incremental improvements over broad rewrites.
- Do not implement or document large future features unless explicitly requested.
- Keep the repository aligned with the current implementation stage.

## Environment Compatibility

- Every code change must preserve Gymnasium compatibility.
- The environment must remain creatable with `gym.make("StochPaint-v0")`.
- Do not break the current minimal environment design unless explicitly asked.

## Validation After Changes

After each meaningful task, validate at least the following:

- `gym.make("StochPaint-v0")` works
- `reset()` and `step()` run without errors
- `python examples/random_agent.py` runs successfully
- if PPO-related code changed, `python examples/train_ppo.py` and `python benchmarks/evaluate_random.py --ppo-model ...` must still work

## Documentation Rules

- README should describe only features that are currently implemented.
- Do not present planned ideas as existing functionality.
- Keep documentation concise and repository-specific.

## Dependencies

- Prefer minimal dependencies.
- Do not add new packages unless they are clearly necessary for the requested task.

## Local Machine Path Rules

- Keep project source code under `D:\Projects`.
- Place Python environments, models, datasets, and caches under `D:\XianLab`.
- Do not default large dependencies or caches to `C:\`.
- Before creating any new environment, propose the target path layout first, then execute.
