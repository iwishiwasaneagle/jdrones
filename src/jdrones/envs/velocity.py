#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Dict
from typing import Tuple

from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from jdrones.controllers import PID
from jdrones.controllers import PID_antiwindup
from jdrones.envs.base import BaseControlledEnv
from jdrones.envs.base import BaseDroneEnv
from jdrones.envs.base import PyBulletDroneEnv
from jdrones.maths import apply_rpy
from jdrones.maths import clip_scalar
from jdrones.types import State
from jdrones.types import VelHeadAltAction


class VelHeadAltDroneEnv(BaseControlledEnv):
    def __init__(self, *args, env: BaseDroneEnv = None, dt: float = 1 / 240, **kwargs):
        if env is None:
            env = PyBulletDroneEnv(*args, dt=dt, **kwargs)
        super(VelHeadAltDroneEnv, self).__init__(env=env, dt=dt)

    @staticmethod
    def _init_controllers(dt: float) -> Dict[str, PID]:
        vx_b = PID_antiwindup(1, 0, 0, dt, gain=0.01)
        vy_b = PID_antiwindup(1, 0, 0, dt, gain=0.01)

        return dict(vx_b=vx_b, vy_b=vy_b)

    def step(self, action: VelHeadAltAction) -> Tuple[State, float, bool, bool, dict]:
        vx_b, vy_b, yaw, z = action
        # Convert x-y from inertial to body
        vx_b_m, vy_b_m, _ = apply_rpy(self.env.state.vel, (0, 0, self.env.state.rpy[0]))

        # Calculate control action
        p_act = self.controllers["vx_b"](vx_b_m, vx_b)
        r_act = self.controllers["vy_b"](vy_b_m, vy_b)

        k = 0.1
        p_act = clip_scalar(p_act, -k, k)
        r_act = clip_scalar(r_act, -k, k)

        attalt_act = np.array((r_act, p_act, yaw, z))
        return self.env.step(attalt_act)

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array(
            [
                (-1, 1),  # vx
                (-1, 1),  # vy
                (-np.pi, np.pi),  # Yaw
                (1.0, np.inf),  # Altitude
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )


if __name__ == "__main__":

    import gymnasium
    import jdrones
    import jdrones.types

    import pandas as pd
    import numpy as np

    from tqdm.auto import tqdm

    # In[4]:

    import seaborn as sns
    import matplotlib.pyplot as plt

    # In[5]:

    from collections import deque

    # In[6]:

    T = 50
    dt = 1 / 240
    seed = 1337

    initial_state = jdrones.types.State()
    initial_state.pos = (0, 0, 2)

    # In[7]:

    env = gymnasium.make("VelHeadAltDroneEnv-v0", dt=dt, initial_state=initial_state)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=int(T / dt))

    # In[18]:

    observations = deque()
    errors = deque()

    obs, info = env.reset(seed=seed)
    setpoint = [0, 1, 0, 2]
    trunc, term = False, False
    for i in tqdm(range(int(T / dt) - 1)):
        obs, _, term, trunc, info = env.step(setpoint)
        observations.append(obs)
        errors.append([v.e for k, v in env.controllers.items()])

    fig, ax = plt.subplots()

    data = np.array(errors)

    ax.plot(data[:, 0])
    ax.plot(data[:, 1])

    plt.show()

    # In[19]:

    data = np.array(observations)
    t = np.linspace(0, len(data) * dt, len(data))
    df = pd.DataFrame(
        data,
        columns=[
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "phi",
            "theta",
            "psi",
            "vx",
            "vy",
            "vz",
            "p",
            "q",
            "r",
            "P0",
            "P1",
            "P2",
            "P3",
        ],
        index=t,
    )
    df.index.name = "t"

    df = df.iloc[
        :: int(len(df) / 1000), :
    ]  # Select only every 500th row for performance

    # In[20]:

    df_long = df.melt(
        var_name="variable", value_name="value", ignore_index=False
    ).reset_index()

    # In[21]:

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax = ax.flatten()

    sns.lineplot(
        data=df_long.query("variable in ('x','y','z')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[0],
    )
    ax[0].hlines(
        setpoint[3],
        df.index.min(),
        df.index.max(),
        linestyles="dotted",
        label="setpoint",
    )
    ax[0].legend()

    sns.lineplot(
        data=df_long.query("variable in ('phi','theta','psi')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[1],
    )
    ax[1].legend()

    sns.lineplot(
        data=df_long.query("variable in ('vx','vy','vz')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[2],
    )
    ax[2].legend()

    sns.lineplot(
        data=df_long.query("variable in ('P0','P1','P2','P3')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[3],
    )
    ax[3].legend()

    fig.tight_layout()

    plt.show()

    # In[22]:

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(df.x, df.y)

    fig.tight_layout()

    plt.show()
