from gymnasium.error import Error as GymnasiumError
from gymnasium.envs.registration import register

from stochpaint.env.coating_env import CoatingEnv


try:
    register(
        id="StochPaint-v0",
        entry_point="stochpaint.env.coating_env:CoatingEnv",
    )
except GymnasiumError:
    pass


__all__ = ["CoatingEnv"]
