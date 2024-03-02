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
from jdrones.data_models import State as _State
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.types import PropellerAction
from jdrones.wrappers import EnergyCalculationWrapper
from loguru import logger
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor

matplotlib.use("Agg")
logger.info(f"Starting {__file__}")

POS_LIM = (-10, 10)
TGT_SUB_LIM = (-5, 5)
VEL_LIM = (-100, 100)
RPY_LIM = (-np.pi, np.pi)
ANG_VEL_LIM = (-100, 100)
PROP_OMEGA_LIM = (0, 50)


class State(_State):
    k: int = 29

    @property
    def target(self):
        return self[20:23]

    @target.setter
    def target(self, val):
        self[20:23] = val

    @property
    def target_error(self):
        return self[23:26]

    @target_error.setter
    def target_error(self, val):
        self[23:26] = val

    @property
    def target_error_integral(self):
        return self[26:29]

    @target_error_integral.setter
    def target_error_integral(self, val):
        self[26:29] = val

    def normed(self, limits: list[tuple[float, float]]):
        data = State()
        for i, (value, (lower, upper)) in enumerate(zip(self, limits)):
            data[i] = np.interp(value, (lower, upper), (-1, 1))
        return data


class HoverEnv(gymnasium.Env):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

        self.env = EnergyCalculationWrapper(NonlinearDynamicModelDroneEnv(dt=self.dt))

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(State.k,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.reset_target()

    def get_observation(self):
        state = State()
        state[:20] = self.env.unwrapped.state
        state.target = self.target
        state.target_error = self.target_error
        state.target_error_integral = self.integral_target_error

        normed_state = state.normed(
            [
                POS_LIM,
                POS_LIM,
                POS_LIM,
                (-1, 1),
                (-1, 1),
                (-1, 1),
                (-1, 1),
                RPY_LIM,
                RPY_LIM,
                RPY_LIM,
                VEL_LIM,
                VEL_LIM,
                VEL_LIM,
                ANG_VEL_LIM,
                ANG_VEL_LIM,
                ANG_VEL_LIM,
                PROP_OMEGA_LIM,
                PROP_OMEGA_LIM,
                PROP_OMEGA_LIM,
                PROP_OMEGA_LIM,
                POS_LIM,
                POS_LIM,
                POS_LIM,
                (-20, 20),
                (-20, 20),
                (-20, 20),
                (-100, 100),
                (-100, 100),
                (-100, 100),
            ]
        )
        return normed_state

    def reset_target(self):
        self.target = np.random.uniform(
            *TGT_SUB_LIM,
            3,
        )
        self.target_counter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)

        reset_state = _State()
        reset_state.pos = np.random.uniform(*TGT_SUB_LIM, 3)
        reset_state.vel = np.random.uniform(-1, 1, 3)
        reset_state.rpy = np.random.uniform(-0.2, 0.2, 3)
        reset_state.ang_vel = np.random.uniform(-0.1, 0.1, 3)

        _, info = self.env.reset(options=dict(reset_state=reset_state))
        self.reset_target()
        self.previous_prop_omega = 0
        self.target_error = self.integral_target_error = np.zeros_like(self.target)
        self.info = {"is_success": False}
        return self.get_observation(), info

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        trunc = False
        term = False
        denormed_action = np.interp(action, (-1, 1), PROP_OMEGA_LIM)
        obs, _, _, _, info = self.env.step(denormed_action)
        self.info["action"] = action

        prop_omega = obs.prop_omega
        control_action = np.linalg.norm(obs.prop_omega)
        dcontrol_action = np.linalg.norm(self.previous_prop_omega - prop_omega)
        self.previous_prop_omega = np.copy(prop_omega)
        self.info["control_action"] = control_action
        self.info["dcontrol_action"] = dcontrol_action

        self.target_error = self.target - obs.pos
        self.integral_target_error = (
            self.integral_target_error * 0.9 + self.target_error
        )

        distance_from_tgt = np.linalg.norm(self.target_error)
        self.info["distance_from_target"] = distance_from_tgt

        reward = (
            2  # alive bonus
            + -5e-1 * distance_from_tgt
            + -1e-6 * info["energy"]
            + -1e-4 * control_action
            + 0 * dcontrol_action
            + -1e-6 * np.linalg.norm(self.integral_target_error)
        )
        if distance_from_tgt < 0.5:
            self.target_counter += 1
            reward += 1
            if self.target_counter > int(1 / self.env.dt):
                self.info["is_success"] = True
                self.reset_target()
        elif np.any((POS_LIM[0] > obs[:3]) | (POS_LIM[1] < obs[:3])):
            self.info["is_success"] = False
            trunc = True
            reward -= 10

        return self.get_observation(), float(reward), term, trunc, self.info | info


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
            info_ = info[0]
            reward_ = reward[0]
            obs_ = State(obs[0, : State.k])
            t += env.get_attr("dt")[0]
            a1, a2, a3, a4 = info_.pop("action")
            x, y, z = obs_.pos
            vx, vy, vz = obs_.vel
            p1, p2, p3, p4 = obs_.prop_omega
            tx, ty, tz = obs_.target
            iX, iY, iZ = obs_.target_error_integral
            log.append(
                info_
                | dict(
                    time=t,
                    reward=reward_,
                    x=x,
                    y=y,
                    z=z,
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    p4=p4,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    a1=a1,
                    a2=a2,
                    a3=a3,
                    a4=a4,
                    iX=iX,
                    iY=iY,
                    iZ=iZ,
                )
            )

        df = pd.DataFrame(log)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.energy)
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        self.logger.record(
            "data/energy",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.distance_from_target)
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
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
        ax3 = ax.twinx()
        ax3.plot(df.time, df.reward, c="y")
        ax3.set_ylabel("reward", color="y")
        ax3.spines["right"].set_position(("axes", 1.2))
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
        ax.plot(df.time, df.vx, c="g", label="x")
        ax.plot(df.time, df.vy, c="r", label="y")
        ax.plot(df.time, df.vz, c="b", label="z")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        ax.legend()
        self.logger.record(
            "data/velocity",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.x, c="g", label="x")
        ax.plot(df.time, df.tx, linestyle="-.", c="g")
        ax.plot(df.time, df.y, c="r", label="y")
        ax.plot(df.time, df.ty, linestyle="-.", c="r")
        ax.plot(df.time, df.z, c="b", label="z")
        ax.plot(df.time, df.tz, linestyle="-.", c="b")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
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
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        self.logger.record(
            "data/propeller_rpm",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(df.time, df[f"a{i + 1}"], label=f"A{i + 1}")
        ax.legend()
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        self.logger.record(
            "data/action",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.iX, c="g", label="x")
        ax.plot(df.time, df.iY, c="r", label="y")
        ax.plot(df.time, df.iZ, c="b", label="z")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        ax.legend()
        self.logger.record(
            "data/integral_error",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        return True


def make_env(dt):
    env = HoverEnv(dt=dt)
    env = TimeAwareObservation(env)
    env = TimeLimit(env, int(10 / env.get_wrapper_attr("dt")))
    env = Monitor(env)
    return env


dt = 1 / 100
env = make_vec_env(make_env, n_envs=16, env_kwargs=dict(dt=dt))
eval_env = make_vec_env(make_env, n_envs=10, env_kwargs=dict(dt=dt))
model = PPO(
    "MlpPolicy",
    env=env,
    verbose=0,
    tensorboard_log="log/tensorboard",
)
eval_callback = EvalCallback(
    eval_env,
    eval_freq=500000 // 16,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback_on_new_best=CallbackList(
        [
            GraphingCallback(),
            CheckpointCallback(
                save_freq=1, save_path="log/model", save_replay_buffer=True
            ),
        ]
    ),
)
model.learn(total_timesteps=1e8, progress_bar=True, callback=eval_callback)
