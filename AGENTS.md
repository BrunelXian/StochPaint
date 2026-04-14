# AGENTS.md

## Project Stage

StochPaint is currently a minimal runnable Gymnasium environment. It is not yet a full benchmark suite.

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

## Documentation Rules

- README should describe only features that are currently implemented.
- Do not present planned ideas as existing functionality.
- Keep documentation concise and repository-specific.

## Dependencies

- Prefer minimal dependencies.
- Do not add new packages unless they are clearly necessary for the requested task.
