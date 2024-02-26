"""
A simple example of how to use SB3's PPO algorithm to control a drone from A to B
and then to hover there until time
runs out. This uses a square error reward function.

Run
===

..code-block:: bash
    PYTHONPATH=src python docs/examples/drl_hover_square_error.py

"""
from collections import deque
from typing import Optional
from typing import Tuple

import gymnasium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeAwareObservation
from gymnasium.wrappers import TimeLimit
from jdrones.data_models import State
from jdrones.energy_model import StaticPropellerVariableVelocityEnergyModel
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.types import PropellerAction
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor

matplotlib.use("Agg")
logger.info(f"Starting {__file__}")


class HoverMSERewardWrapperNonlinearDynamicModelDroneEnv(NonlinearDynamicModelDroneEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        high, low = self.observation_space.high, self.observation_space.low
        high[:3] = 10
        low[:3] = -10
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.array((0, 0, 0, 0, 1)),
            high=np.array((1, 1, 1, 1, 1000)),
            dtype=self.action_space.dtype,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        self.hover_tgt = self.observation_space.sample()[:3]
        return super().reset(seed=seed, options=options)

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        action = action[:4] * action[4]
        obs, _, trunc, term, info = super().step(action)
        distance_from_tgt = np.linalg.norm(self.hover_tgt - self.state.pos)
        reward = -np.square(distance_from_tgt)

        return obs, reward, trunc, term, info


def make_env():
    env = HoverMSERewardWrapperNonlinearDynamicModelDroneEnv()
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = TimeLimit(env, int(10 / env.dt))
    env = Monitor(env)
    return env


env = make_vec_env(make_env, n_envs=1)
eval_env = make_vec_env(make_env, n_envs=10)
model = PPO("MlpPolicy", env=env, verbose=0, tensorboard_log="/tmp/jdrones")
eval_callback = EvalCallback(
    eval_env, eval_freq=50000, n_eval_episodes=10, deterministic=True, render=False
)
model.learn(total_timesteps=1e7, progress_bar=True, callback=eval_callback)
