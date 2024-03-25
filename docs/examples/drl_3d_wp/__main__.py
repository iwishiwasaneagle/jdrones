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
import warnings

import click
import matplotlib
import optuna
import torch as th
from callback import GraphingCallback
from drl_3d_wp.callback import TrialEvalCallback
from drl_3d_wp.consts import DT
from drl_3d_wp.consts import LOG_PATH
from drl_3d_wp.consts import N_ENVS
from drl_3d_wp.consts import N_EVAL
from drl_3d_wp.consts import OPTUNA_PATH
from drl_3d_wp.consts import TENSORBOARD_PATH
from drl_3d_wp.consts import TOTAL_TIMESTEP
from drl_3d_wp.env import DRL_WP_Env
from drl_3d_wp.env import DRL_WP_Env_LQR
from drl_3d_wp.policies import ActorCriticDenseNetPolicy
from gymnasium.wrappers import TimeAwareObservation
from gymnasium.wrappers import TimeLimit
from loguru import logger
from optuna.pruners import HyperbandPruner
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")
logger.info(f"Starting {__file__}")


def make_env(env_type):
    match env_type:
        case "direct":
            env = DRL_WP_Env(dt=DT)
            env = TimeAwareObservation(env)
            env = TimeLimit(env, int(10 / DT))
        case "LQR":
            env = DRL_WP_Env_LQR(dt=DT, T=10)
    env = Monitor(env, info_keywords=("is_success", "is_oob", "is_unstable", "targets"))
    return env


def build_callback(
    total_timesteps: int,
    eval_callback_cls=EvalCallback,
    eval_callback_kwargs=None,
    make_vec_env_kwargs=None,
):
    if eval_callback_kwargs is None:
        eval_callback_kwargs = {}

    if make_vec_env_kwargs is None:
        make_vec_env_kwargs = {}
    n_eval = eval_callback_kwargs.pop("n_eval", N_EVAL)
    n_envs = eval_callback_kwargs.pop("n_envs", N_ENVS)
    usual_kwargs = dict(
        eval_freq=total_timesteps // (n_eval * n_envs),
        n_eval_episodes=100,
        deterministic=True,
        verbose=1,
        callback_after_eval=GraphingCallback(),
    )

    kwargs = eval_callback_kwargs | usual_kwargs

    eval_env = make_vec_env(make_env, n_envs=10, **make_vec_env_kwargs)
    eval_callback = eval_callback_cls(eval_env, **kwargs)
    return eval_callback


def build_trial_callback(total_timesteps: int, trial: optuna.Trial):
    eval_callback = build_callback(
        total_timesteps,
        eval_callback_cls=TrialEvalCallback,
        eval_callback_kwargs=dict(verbose=0, trial=trial),
    )
    return eval_callback


def build_model(
    *,
    env,
    net_arch_name,
    lr,
    batch_size,
    use_sde,
    n_steps,
    device,
    net_arch_mlp_width=None,
    net_arch_mlp_depth=None,
    net_arch_dense_layers=None,
    net_arch_lstm_layers=None,
    net_arch_lstm_hidden_size=None,
):
    match net_arch_name:
        case "mlp":
            from sbx import PPO as PPO_SBX

            model = PPO_SBX(
                "MlpPolicy",
                device=device,
                learning_rate=lr,
                batch_size=batch_size,
                use_sde=use_sde,
                n_steps=n_steps,
                policy_kwargs=dict(
                    net_arch=[
                        net_arch_mlp_width,
                    ]
                    * net_arch_mlp_depth
                ),
                env=env,
                verbose=0,
                tensorboard_log=TENSORBOARD_PATH,
            )
        case "dense":
            from stable_baselines3 import PPO as PPO_SB3

            model = PPO_SB3(
                ActorCriticDenseNetPolicy,
                device=device,
                learning_rate=lr,
                batch_size=batch_size,
                use_sde=use_sde,
                n_steps=n_steps,
                policy_kwargs=dict(net_arch=net_arch_dense_layers),
                env=env,
                verbose=0,
                tensorboard_log=TENSORBOARD_PATH,
            )
        case "recurrent":
            from sb3_contrib import RecurrentPPO

            model = RecurrentPPO(
                "MlpLstmPolicy",
                device=device,
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
                tensorboard_log=TENSORBOARD_PATH,
            )
        case _:
            raise Exception()
    return model


def objective(trial: optuna.Trial):
    kwargs = dict(
        net_arch_name=trial.suggest_categorical(
            "net_arch", ["dense", "mlp", "recurrent"]
        ),
        env=make_vec_env(
            make_env,
            n_envs=N_ENVS,
            vec_env_cls=DummyVecEnv if N_ENVS == 1 else SubprocVecEnv,
        ),
        lr=trial.suggest_float("learning_rate", 1e-6, 1e-3),
        batch_size=trial.suggest_int("batch_size", 2, 512),
        n_steps=trial.suggest_int("n_steps", 512, 4192),
        use_sde=bool(trial.suggest_int("use_sde", 0, 1)),
    )
    match kwargs.get("net_arch_name"):
        case "mlp":
            kwargs = kwargs | dict(
                net_arch_depth=trial.suggest_int("net_arch_mlp_depth", 2, 4),
                net_arch_width=trial.suggest_int("net_arch_mlp_width", 8, 2048),
            )
        case "dense":
            kwargs = kwargs | dict(
                net_arch_layers=trial.suggest_int("net_arch_dense_net_layers", 1, 4)
            )
        case "recurrent":
            kwargs = kwargs | dict(
                net_arch_lstm_layers=trial.suggest_int(
                    "net_arch_lstm_net_layers", 1, 3
                ),
                net_arch_lstm_hidden_size=trial.suggest_int(
                    "net_arch_lstm_width", 64, 512
                ),
            )
        case _:
            raise Exception()
    eval_callback = build_trial_callback(trial)
    model = build_model(**kwargs)
    model.learn(
        total_timesteps=TOTAL_TIMESTEP, progress_bar=True, callback=eval_callback
    )
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    mean_reward, _ = evaluate_policy(model, eval_callback.eval_env, n_eval_episodes=500)

    return mean_reward


@click.group()
def main():
    logger.info(f"Is cuda available? {th.cuda.is_available()}")


@main.command("learn", context_settings={"show_default": True})
@click.option("--env_type", type=click.Choice(["LQR", "direct"]), default="direct")
@click.option("--vec_env_cls", type=click.Choice(["dummy", "subproc"]), default="dummy")
@click.option("--batch_size", type=int, default=128)
@click.option("--n_steps", type=int, default=4096)
@click.option("--lr", nargs=3, default=(0.0003, 0.0003, 1))
@click.argument("net_arch_name", type=click.Choice(["mlp", "dense", "recurrent"]))
@click.option("-N", "--num_timesteps", type=int, default=TOTAL_TIMESTEP)
@click.option("--use_sde", is_flag=True, default=False)
@click.option("--net_arch_mlp_width", type=int, default=1024)
@click.option("--net_arch_mlp_depth", type=int, default=4)
@click.option("--net_arch_dense_layers", type=int, default=4)
@click.option("--net_arch_lstm_layers", type=int, default=1)
@click.option("--net_arch_lstm_hidden_size", type=int, default=256)
@click.option("--n_eval", type=int, default=N_EVAL)
@click.option("--n_envs", type=int, default=N_ENVS)
@click.option("--wandb_project", default=None, type=str)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda")
def learn(wandb_project, vec_env_cls, env_type, **kwargs):
    N = kwargs.pop("num_timesteps")
    n_eval = kwargs.pop("n_eval")
    n_envs = kwargs.pop("n_envs")
    kwargs["lr"] = get_linear_fn(*kwargs.get("lr"))
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv if vec_env_cls == "dummy" else SubprocVecEnv,
        env_kwargs=dict(env_type=env_type),
    )
    model = build_model(env=env, **kwargs)
    callback = build_callback(
        N,
        eval_callback_kwargs=dict(n_eval=n_eval, n_envs=n_envs),
        make_vec_env_kwargs=dict(env_kwargs=dict(env_type=env_type)),
    )

    if wandb_project is not None:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(
            project=wandb_project,
            dir=LOG_PATH,
            sync_tensorboard=True,
            tags=[env_type, vec_env_cls, kwargs.get("net_arch_name")],
            monitor_gym=True,
            save_code=True,
        )
        callback = CallbackList([callback, WandbCallback()])

    model.learn(total_timesteps=N, progress_bar=True, callback=callback)
    model.save(LOG_PATH)


@main.command("sweep")
@click.option("--study_name", type=str, default="drl_3d_wp")
def sweep(study_name):
    storage_path = (OPTUNA_PATH / study_name).with_suffix(".db")
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        pruner=HyperbandPruner(min_resource=1, max_resource=N_EVAL, reduction_factor=2),
    )  # Create a new study.
    study.optimize(objective, n_trials=300)


main()
