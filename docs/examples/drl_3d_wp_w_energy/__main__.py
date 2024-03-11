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
import matplotlib
import optuna
from gymnasium.wrappers import TimeAwareObservation
from gymnasium.wrappers import TimeLimit
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler
from sb3_contrib import RecurrentPPO
from sbx import PPO as PPO_SBX
from stable_baselines3 import PPO as PPO_SB3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from .callback import TrialEvalCallback
from .consts import N_EVAL
from .consts import TOTAL_TIMESTEP
from .env import HoverEnv
from .policies import ActorCriticDenseNetPolicy

matplotlib.use("Agg")
logger.info(f"Starting {__file__}")


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
