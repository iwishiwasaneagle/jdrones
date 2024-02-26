#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
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
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor

matplotlib.use("Agg")
logger.info(f"Starting {__file__}")


class HoverEnv(NonlinearDynamicModelDroneEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        high, low = self.observation_space.high, self.observation_space.low
        high[:3] = 10
        low[:3] = -10
        high = np.concatenate((high, np.full(3, 5.0)))
        low = np.concatenate((low, np.full(3, -5.0)))
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.full(4, 0),
            high=np.full(4, 1),
            dtype=np.float32,
        )

        self.reset_target()

        self.energy_calculation = StaticPropellerVariableVelocityEnergyModel(
            self.dt, self.model
        )

    def get_observation(self):
        return np.concatenate((self.state, self.hover_tgt))

    def reset_target(self):
        self.hover_tgt = np.random.uniform(
            self.observation_space.low[20:23].min(),
            self.observation_space.high[20:23].max(),
            3,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        _, info = super().reset(seed=seed, options=options)
        self.reset_target()
        self.previous_action = np.zeros(self.action_space.shape)
        self.info["is_success"] = False
        return self.get_observation(), info

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        _, _, term, trunc, _ = super().step(action * 10)

        w1, w2, w3, w4 = -1 / np.sqrt(np.square(10) * 3), 0, -0, -0.1
        speed = np.linalg.norm(self.state.vel)
        energy = self.energy_calculation.energy(speed)
        self.info["energy"] = energy

        control_action = np.sum(np.abs(action))
        dcontrol_action = np.sum(np.abs(self.previous_action - action))
        self.previous_action = action
        self.info["control_action"] = control_action
        self.info["dcontrol_action"] = dcontrol_action

        distance_from_tgt = np.linalg.norm(self.hover_tgt - self.state.pos)
        self.info["distance_from_target"] = distance_from_tgt

        reward = (
            w1 * distance_from_tgt
            + w2 * energy
            + w3 * control_action
            + w4 * dcontrol_action
        )

        if distance_from_tgt < 1.5:
            reward += 25
            self.info["is_success"] = True
            self.reset_target()

        a = 2 * self.observation_space.low[:3]
        b = 2 * self.observation_space.high[:3]
        c = self.state.pos
        if np.any((a > c) | (b < c)):
            reward -= 25
            trunc = True

        return self.get_observation(), reward, term, trunc, self.info


class GraphingCallback(BaseCallback):
    def _on_step(self):
        log = deque()
        env = self.model.get_env()
        obs = env.reset()
        states = None
        t = 0
        while True:
            action, states = self.model.predict(obs, deterministic=True, state=states)
            obs, reward, done, info = env.step(action)
            if np.any(done):
                break
            t += env.get_attr("dt")[0]
            x, y, z = obs[0][:3]
            p1, p2, p3, p4 = obs[0][16:20]
            tx, ty, tz = obs[0][20:23]
            log.append(
                info[0]
                | dict(
                    time=t,
                    x=x,
                    y=y,
                    z=z,
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    p4=p4,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                )
            )

        df = pd.DataFrame(log)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.energy)
        self.logger.record(
            "data/energy",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.distance_from_target)
        self.logger.record(
            "data/distance_from_target",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record(
            "eval/sum_distance_from_target", df.distance_from_target.sum()
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.control_action)
        ax.set(ylabel="u")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.dcontrol_action, c="g")
        ax2.set_ylabel("du", color="g")
        fig.tight_layout()
        self.logger.record(
            "data/control_action",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record("eval/sum_control_action", df.control_action.sum())
        self.logger.record("eval/sum_dcontrol_action", df.dcontrol_action.sum())
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.x, c="g", label="x")
        ax.plot(df.time, df.tx, linestyle="-.", c="g")
        ax.plot(df.time, df.y, c="r", label="y")
        ax.plot(df.time, df.ty, linestyle="-.", c="r")
        ax.plot(df.time, df.z, c="b", label="z")
        ax.plot(df.time, df.tz, linestyle="-.", c="b")
        ax.legend()
        self.logger.record(
            "data/position",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(df.time, df[f"p{i + 1}"], label=f"P{i + 1}")
        ax.legend()
        self.logger.record(
            "data/propeller_rpm",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        return True


def make_env(dt):
    env = HoverEnv(dt=dt)
    env = TimeLimit(env, int(10 / env.get_wrapper_attr("dt")))
    env = TimeAwareObservation(env)
    env = Monitor(env)
    return env


dt = 1 / 50
env = make_vec_env(make_env, n_envs=1, env_kwargs=dict(dt=dt))
eval_env = make_vec_env(make_env, n_envs=10, env_kwargs=dict(dt=dt))
model = PPO(
    "MlpPolicy",
    env=env,
    verbose=0,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])),
    tensorboard_log="/tmp/jdrones",
)
eval_callback = EvalCallback(
    eval_env,
    eval_freq=50000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback_after_eval=GraphingCallback(),
)
model.learn(total_timesteps=1e7, progress_bar=True, callback=eval_callback)
