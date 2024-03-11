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
import optuna
import pandas as pd
import torch as th
from gymnasium.wrappers import TimeAwareObservation
from gymnasium.wrappers import TimeLimit
from jdrones.data_models import State as _State
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.types import PropellerAction
from jdrones.wrappers import EnergyCalculationWrapper
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler
from sb3_contrib import RecurrentPPO
from sbx import PPO as PPO_SBX
from stable_baselines3 import PPO as PPO_SB3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv

matplotlib.use("Agg")
logger.info(f"Starting {__file__}")

TOTAL_TIMESTEP = int(5e6)
N_EVAL = 1000

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
        self.info = {"is_success": False, "targets": 0}
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
            4  # alive bonus
            + -5e-1 * distance_from_tgt
            + 0 * info["energy"]
            + 0 * control_action
            + 0 * dcontrol_action
            + 0 * np.linalg.norm(self.integral_target_error)
        )
        if distance_from_tgt < 1.5:
            self.target_counter += 1
            reward += 1
            if self.target_counter > int(1 / self.env.dt):
                self.info["is_success"] = True
                self.info["targets"] += 1
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


class DenseNetModule(th.nn.Module):
    def __init__(self, in_dim, activation_fn):
        super().__init__()
        self.linear = th.nn.Linear(in_dim, in_dim)
        self.activation = activation_fn()

    def forward(self, feature):
        x = self.linear(feature)
        x = self.activation(x)
        return th.cat((x, feature), 1)


class DenseNetExtractor(th.nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: int,
        activation_fn,
        device="auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net = []
        value_net = []

        # Iterate through the policy layers and build the policy net
        for n in range(net_arch):
            d = feature_dim * (2**n)
            policy_net.append(DenseNetModule(d, activation_fn))
            value_net.append(DenseNetModule(d, activation_fn))

        self.latent_dim_vf = self.latent_dim_pi = 2 * d

        self.policy_net = th.nn.Sequential(*policy_net).to(device)
        self.value_net = th.nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ActorCriticDenseNetPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        net_arch: Optional[int] = None,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = 2
        super().__init__(*args, net_arch=net_arch, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DenseNetExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


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


def make_env(dt):
    env = HoverEnv(dt=dt)
    env = TimeAwareObservation(env)
    env = TimeLimit(env, int(10 / env.get_wrapper_attr("dt")))
    env = Monitor(env)
    return env


def objective(trial):
    dt = 1 / 100
    net_arch_name = trial.suggest_categorical(
        "net_arch", ["dense", "mlp"]
    )  # , "recurrent"])
    n_envs = trial.suggest_int("n_envs", 1, 16)
    env = make_vec_env(make_env, n_envs=n_envs, env_kwargs=dict(dt=dt))
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3)
    batch_size = trial.suggest_int("batch_size", 2, 512)
    n_steps = trial.suggest_int("n_steps", 512, 4192)
    use_sde = bool(trial.suggest_int("use_sde", 0, 1))
    match net_arch_name:
        case "mlp":
            net_arch_depth = trial.suggest_int("net_arch_mlp_depth", 2, 4)
            net_arch_width = trial.suggest_int("net_arch_mlp_width", 8, 2048)
            model = PPO_SBX(
                "MlpPolicy",
                learning_rate=lr,
                batch_size=batch_size,
                use_sde=use_sde,
                n_steps=n_steps,
                policy_kwargs=dict(
                    net_arch=[
                        net_arch_width,
                    ]
                    * net_arch_depth
                ),
                env=env,
                verbose=0,
                tensorboard_log="logs/tensorboard",
            )
        case "dense":
            net_arch_layers = trial.suggest_int("net_arch_dense_net_layers", 1, 4)
            model = PPO_SB3(
                ActorCriticDenseNetPolicy,
                learning_rate=lr,
                batch_size=batch_size,
                use_sde=use_sde,
                n_steps=n_steps,
                policy_kwargs=dict(net_arch=net_arch_layers),
                env=env,
                verbose=0,
                tensorboard_log="logs/tensorboard",
            )
        case "recurrent":
            net_arch_lstm_layers = trial.suggest_int("net_arch_lstm_net_layers", 1, 3)
            net_arch_lstm_hidden_size = trial.suggest_int(
                "net_arch_lstm_width", 64, 512
            )
            model = RecurrentPPO(
                "MlpLstmPolicy",
                policy_kwargs=dict(
                    lstm_hidden_size=net_arch_lstm_layers,
                    n_lstm_layers=net_arch_lstm_hidden_size,
                ),
                learning_rate=lr,
                batch_size=batch_size,
                use_sde=use_sde,
                n_steps=n_steps,
                env=env,
                verbose=0,
                tensorboard_log="logs/tensorboard",
            )
        case _:
            raise Exception()
    eval_env = make_vec_env(make_env, n_envs=10, env_kwargs=dict(dt=dt))
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        eval_freq=TOTAL_TIMESTEP // (N_EVAL * n_envs),
        n_eval_episodes=100,
        deterministic=True,
        verbose=0,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEP, progress_bar=True, callback=eval_callback
    )
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1000)

    return mean_reward


study_name = "drl_3d_w_energy"
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    storage=f"sqlite:///logs/optuna/{study_name}.db",
    load_if_exists=True,
    sampler=RandomSampler(),
    pruner=MedianPruner(n_startup_trials=int(0.01 * N_EVAL)),
)  # Create a new study.
study.enqueue_trial(
    dict(
        net_arch="mlp",
        net_arch_mlp_depth=2,
        net_arch_mlp_width=8,
        n_envs=16,
        batch_size=512,
    )
)
study.enqueue_trial(
    dict(
        net_arch="mlp",
        net_arch_mlp_depth=3,
        net_arch_mlp_width=10,
        n_envs=15,
        batch_size=400,
    )
)
study.enqueue_trial(
    dict(net_arch="dense", net_arch_dense_net_layers=1, n_envs=16, batch_size=256)
)
study.enqueue_trial(
    dict(
        net_arch="mlp",
        net_arch_mlp_depth=4,
        net_arch_mlp_width=8,
        n_envs=14,
        batch_size=300,
    )
)
study.optimize(objective, n_trials=300)
