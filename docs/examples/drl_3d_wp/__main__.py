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
import torch as th
from callback import EvalCallbackWithMoreLogging
from callback import GraphingCallback
from drl_3d_wp.consts import DT
from drl_3d_wp.consts import LOG_PATH
from drl_3d_wp.consts import N_ENVS
from drl_3d_wp.consts import N_EVAL
from drl_3d_wp.consts import TENSORBOARD_PATH
from drl_3d_wp.consts import TOTAL_TIMESTEP
from drl_3d_wp.env import DRL_WP_Env_LQR
from drl_3d_wp.policies import ActorCriticDenseNetPolicy
from loguru import logger
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")
logger.info(f"Starting {__file__}")


def make_env(T=10):
    env = DRL_WP_Env_LQR(dt=DT, T=T)
    env = Monitor(env, info_keywords=("is_success", "is_oob", "is_unstable", "targets"))
    return env


def build_callback(
    total_timesteps: int,
    eval_callback_cls=EvalCallbackWithMoreLogging,
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
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
        callback_after_eval=GraphingCallback(),
    )

    kwargs = eval_callback_kwargs | usual_kwargs

    eval_env = make_vec_env(make_env, n_envs=10, **make_vec_env_kwargs)
    eval_callback = eval_callback_cls(eval_env, **kwargs)
    return eval_callback


def build_model(
    *,
    env,
    net_arch_name,
    lr,
    batch_size,
    clip_range,
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
            from stable_baselines3 import PPO as PPO_SB3

            model = PPO_SB3(
                "MlpPolicy",
                device=device,
                learning_rate=lr,
                clip_range=clip_range,
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
                clip_range=clip_range,
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
                clip_range=clip_range,
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


@click.group()
def main():
    logger.info(f"Is cuda available? {th.cuda.is_available()}")


@main.command("learn", context_settings={"show_default": True})
@click.option("--vec_env_cls", type=click.Choice(["dummy", "subproc"]), default="dummy")
@click.option("--batch_size", type=int, default=128)
@click.option("--n_steps", type=int, default=4096)
@click.option("--lr", nargs=3, default=(0.0003, 0.0003, 1))
@click.option("--clip_range", default=0.2, type=click.FloatRange(min=0.01))
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
@click.option("-T", "--max_sim_time", type=click.IntRange(min=10), default=10)
def learn(wandb_project, vec_env_cls, max_sim_time, **kwargs):
    N = kwargs.pop("num_timesteps")
    n_eval = kwargs.pop("n_eval")
    n_envs = kwargs.pop("n_envs")
    kwargs["lr"] = get_linear_fn(*kwargs.get("lr"))
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv if vec_env_cls == "dummy" else SubprocVecEnv,
        env_kwargs=dict(T=max_sim_time),
    )
    model = build_model(env=env, **kwargs)
    callback = build_callback(
        N,
        eval_callback_kwargs=dict(n_eval=n_eval, n_envs=n_envs),
        make_vec_env_kwargs=dict(env_kwargs=dict(T=max_sim_time)),
    )

    if wandb_project is not None:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(
            project=wandb_project,
            dir=LOG_PATH,
            sync_tensorboard=True,
            tags=[vec_env_cls, kwargs.get("net_arch_name")],
            monitor_gym=True,
            save_code=True,
        )
        callback = CallbackList([callback, WandbCallback()])

    model.learn(total_timesteps=N, progress_bar=True, callback=callback)
    model.save(LOG_PATH)


main()
