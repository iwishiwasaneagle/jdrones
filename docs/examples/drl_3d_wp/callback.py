#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import enum
import os
from collections import deque
from operator import attrgetter
from typing import Optional

import gymnasium
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from drl_3d_wp.state import State
from jdrones.plotting import plot_2d_path
from loguru import logger
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env import VecEnv


class GraphingCallback(BaseCallback):

    def _on_step(self):
        log = deque()
        env = self.model.get_env()
        obs = env.reset()
        states = None
        t = -env.unwrapped.get_attr("dt")[0]
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        while True:
            action, states = self.model.predict(
                obs, deterministic=True, episode_start=episode_starts, state=states
            )
            obs, reward, done, info = env.step(action)
            if np.any(done):
                break
            episode_starts = done

            info_ = info[0]
            reward_ = reward[0]
            action_ = action[0]
            t += env.unwrapped.get_attr("dt")[0]

            for i, k in enumerate(filter(lambda f: f.startswith("env_"), info_.keys())):
                obs_ = info_[k]["state"]
                tx, ty, tz = info_[k]["target"]
                energy_ = info_[k]["energy"]
                distance_from_target_ = info_[k]["distance_from_target"]
                control_action_ = info_[k]["control_action"]
                dcontrol_action_ = info_[k]["dcontrol_action"]

                if action_.shape == 1:
                    action_ = np.array([action_])

                if obs_.shape == 20:
                    obs_ = np.array([obs])

                for x in obs_:
                    obs_i = State()
                    obs_i[:20] = x
                    x, y, z = obs_i.pos
                    roll, pitch, yaw = obs_i.rpy
                    droll, dpitch, dyaw = obs_i.ang_vel
                    vx, vy, vz = obs_i.vel
                    p1, p2, p3, p4 = obs_i.prop_omega
                    log.append(
                        dict(
                            env=k,
                            time=t,
                            reward=reward_,
                            action=action_[i, :],
                            energy=energy_,
                            distance_from_target=distance_from_target_,
                            control_action=control_action_,
                            dcontrol_action=dcontrol_action_,
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

        df = pd.DataFrame(log)

        fig, ax = plt.subplots()

        marker_shape = ["x", "o"]
        for i, env in enumerate(set(df.env)):
            env_df = df.loc[df.env == env]
            df_long = (
                env_df[["time", "x", "y", "z"]]
                .melt(
                    var_name="variable",
                    value_name="value",
                    id_vars="time",
                )
                .sort_values(by=["time"])
                .rename({"time": "t"}, axis="columns")
            )
            df_long["tag"] = "PPO+LQR"
            plot_2d_path(df_long, ax, label=env)

            targets = env_df[["tx", "ty"]].drop_duplicates()
            ax.scatter(targets.tx, targets.ty, c="b", marker=marker_shape[i])
            ax.scatter(
                *env_df[["x", "y"]].iloc[0].to_list(),
                zorder=10,
                c="g",
                marker=marker_shape[i],
            )
            ax.scatter(
                *env_df[["x", "y"]].iloc[-1].to_list(),
                zorder=10,
                c="r",
                marker=marker_shape[i],
            )

        fig.tight_layout()
        self.logger.record(
            "data/position_2d",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="time", y="energy", hue="env", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/energy",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="time", y="reward", hue="env", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/reward",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="time", y="distance_from_target", hue="env", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/distance_from_target",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        _df = (
            df[["time", "vx", "vy", "vz", "env"]]
            .melt(
                var_name="variable",
                value_name="value",
                id_vars=["env", "time"],
            )
            .sort_values(by=["time"])
        )
        sns.lineplot(data=_df, x="time", y="value", hue="env", style="variable", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/velocity",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        _df = (
            df[["time", "x", "tx", "y", "ty", "z", "tz", "env"]]
            .melt(
                var_name="variable",
                value_name="value",
                id_vars=["env", "time"],
            )
            .sort_values(by=["time"])
        )
        _df["is_target"] = _df["variable"].apply(lambda x: x.startswith("t"))
        g = sns.FacetGrid(
            _df,
            row="env",
            sharey=True,
            sharex=True,
        )
        g.map(
            sns.lineplot,
            data=_df,
            x="time",
            y="value",
            hue="is_target",
            style="variable",
            legend="auto",
        )
        self.logger.record(
            "data/position",
            Figure(g.fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(g.fig)

        fig, ax = plt.subplots()
        _df = (
            df[["time", "env", "roll", "pitch", "yaw"]]
            .melt(
                var_name="variable",
                value_name="value",
                id_vars=["env", "time"],
            )
            .sort_values(by=["time"])
        )
        sns.lineplot(data=_df, x="time", y="value", hue="env", style="variable", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/rpy",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        _df = (
            df[["time", "env", "droll", "dpitch", "dyaw"]]
            .melt(
                var_name="variable",
                value_name="value",
                id_vars=["env", "time"],
            )
            .sort_values(by=["time"])
        )
        sns.lineplot(data=_df, x="time", y="value", hue="env", style="variable", ax=ax)
        fig.tight_layout()
        self.logger.record(
            "data/ang_vel",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        _df = (
            df[["time", "env", "p1", "p2", "p3", "p4"]]
            .melt(
                var_name="variable",
                value_name="value",
                id_vars=["env", "time"],
            )
            .sort_values(by=["time"])
        )
        sns.lineplot(data=_df, x="time", y="value", hue="env", style="variable", ax=ax)
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
        action_df["time"] = df["time"]
        action_df["env"] = df["env"]
        _df = action_df.melt(
            var_name="variable",
            value_name="value",
            id_vars=["env", "time"],
        ).sort_values(by=["time"])
        sns.lineplot(data=_df, x="time", y="value", hue="env", style="variable", ax=ax)
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


class BufferNames(str, enum.Enum):
    TARGETS = "targets"
    IS_OOB = "is_oob"
    IS_UNSTABLE = "is_unstable"
    IS_SUCCESS = "is_success"
    ENERGY = "energy"
    DISTANCE_FROM_TGT = "distance_from_target"
    COLLISION = "collision"


class EvalCallbackWithMoreLogging(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._buffers = {}
        self._max_trackers = {}

        self.is_multi_env = None

    def _eval_callback(self, locals_, globals_) -> None:
        requires_done = {
            BufferNames.TARGETS,
            BufferNames.IS_OOB,
            BufferNames.IS_UNSTABLE,
            BufferNames.IS_SUCCESS,
            BufferNames.COLLISION,
        }

        info = locals_["info"]

        if self.is_multi_env is None:
            self.multi_envs = list(filter(lambda f: f.startswith("env_"), info.keys()))
            self.is_multi_env = len(self.multi_envs) > 0

        for item in list(BufferNames):
            if item in requires_done and not locals_["done"]:
                continue

            key = item.value
            if self.is_multi_env:
                for env in self.multi_envs:
                    env_info = info.get(env)
                    if (value := np.copy(env_info.get(key))) is not None:
                        self._buffers.setdefault(key, []).append(value)
            else:
                if (value := np.copy(info.get(key))) is not None:
                    self._buffers.setdefault(key, []).append(value)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "  # noqa: E501
                        "and warning above."
                    ) from e

            # Reset buffer before evaluating
            self._buffers.clear()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._eval_callback,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/rewards", episode_rewards)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/ep_lengths", episode_lengths)

            for key in map(attrgetter("value"), BufferNames):
                buffer = self._buffers.get(key, [])
                if buffer is not None and len(buffer) > 0:
                    cb_str = f"_{key}_callback"
                    cb = getattr(self, cb_str, None)
                    if cb is None:
                        logger.error(f"Callback {cb_str} does not exist")
                        continue
                    cb(buffer)

            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _path_length_callback(self, buffer):
        self.logger.record("eval/wps/length_mean", float(np.mean(buffer)))
        self.logger.record("eval/wps/length_std", float(np.std(buffer)))

    def _generic_mean_callback(self, key, buffer):
        self.logger.record(key, float(np.mean(buffer)))

    def _is_oob_callback(self, buffer):
        self._generic_mean_callback("eval/oob_rate", buffer)

    def _is_unstable_callback(self, buffer):
        self._generic_mean_callback("eval/unstable_rate", buffer)

    def _targets_callback(self, buffer):
        self._generic_mean_callback("eval/mean_ep_targets", buffer)

    def _is_success_callback(self, buffer):
        self._generic_mean_callback("eval/success_rate", buffer)

    def _energy_callback(self, buffer):
        self._generic_mean_callback("eval/step_energy", buffer)

    def _distance_from_target_callback(self, buffer):
        self._generic_mean_callback("eval/mean_step_distance_from_target", buffer)

    def _collision_callback(self, buffer):
        self._generic_mean_callback("eval/collision_rate", buffer)
