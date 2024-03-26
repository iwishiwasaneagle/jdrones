#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from collections import deque
from typing import Optional

import gymnasium
import numpy as np
import optuna
import pandas as pd
from drl_3d_wp.state import State
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import VecEnv


class GraphingCallback(BaseCallback):

    def _on_step(self):
        log = deque()
        env = self.model.get_env()
        obs = env.reset()
        states = None
        t = 0
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        while True:
            action, states = self.model.predict(
                obs, deterministic=True, episode_start=episode_starts, state=states
            )
            obs, reward, done, info = env.step(action)
            episode_starts = done

            info_ = info[0]
            reward_ = reward[0]
            obs_ = info_["state"]
            tx, ty, tz = info_["target"]
            action_ = action[0]

            if action_.shape == 1:
                action_ = np.array([action_])

            if obs_.shape == 20:
                obs_ = np.array([obs])

            for x in obs_:
                obs_i = State()
                obs_i[:20] = x
                t += env.unwrapped.get_attr("dt")[0]
                x, y, z = obs_i.pos
                roll, pitch, yaw = obs_i.rpy
                droll, dpitch, dyaw = obs_i.ang_vel
                vx, vy, vz = obs_i.vel
                p1, p2, p3, p4 = obs_i.prop_omega
                log.append(
                    info_
                    | dict(
                        time=t,
                        reward=reward_,
                        action=action_,
                        x=x,
                        y=y,
                        z=z,
                        vx=vx,
                        vy=vy,
                        vz=vz,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        p1=p1,
                        p2=p2,
                        p3=p3,
                        p4=p4,
                        tx=tx,
                        ty=ty,
                        tz=tz,
                        droll=droll,
                        dpitch=dpitch,
                        dyaw=dyaw,
                    )
                )
            if np.any(done):
                break

        df = pd.DataFrame(log)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.energy)
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        fig.tight_layout()
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
        fig.tight_layout()
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
        fig.tight_layout()
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
        fig.tight_layout()
        self.logger.record(
            "data/position",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.roll, label="roll")
        ax.plot(df.time, df.pitch, label="pitch")
        ax.plot(df.time, df.yaw, label="yaw")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        ax.legend()
        fig.tight_layout()
        self.logger.record(
            "data/rpy",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(df.time, df.droll, label="droll")
        ax.plot(df.time, df.dpitch, label="dpitch")
        ax.plot(df.time, df.dyaw, label="dyaw")
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        ax.legend()
        fig.tight_layout()
        self.logger.record(
            "data/ang_vel",
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
        fig.tight_layout()
        self.logger.record(
            "data/propeller_rpm",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        action_df = pd.DataFrame(
            df.action.to_list(),
            columns=[f"a{i + 1}" for i in range(len(df.action.iloc[0]))],
        )
        for column in action_df.columns:
            ax.plot(df.time, action_df[column], label=column)
        ax.legend()
        ax2 = ax.twinx()
        ax2.plot(df.time, df.reward, c="y")
        ax2.set_ylabel("reward", color="y")
        fig.tight_layout()
        self.logger.record(
            "data/actions",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        return True


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env | VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        callback_on_new_best: Optional[BaseCallback] = None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            callback_on_new_best=callback_on_new_best,
        )
        self.trial = trial
        self.is_pruned = False

    def _on_step(self) -> bool:
        if (
            self.eval_freq > 0
            and self.n_calls > 0
            and self.n_calls % self.eval_freq == 0
        ):
            super()._on_step()
            self.trial.report(self.last_mean_reward, self.n_calls // self.model.n_envs)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
