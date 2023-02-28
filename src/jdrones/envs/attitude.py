#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Dict
from typing import Tuple

from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from jdrones.controllers import PID
from jdrones.controllers import PID_antiwindup
from jdrones.envs.base import BaseControlledEnv
from jdrones.envs.base import PyBulletDroneEnv
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.maths import clip
from jdrones.types import AttitudeAltitudeAction
from jdrones.types import State


class AttitudeAltitudeDroneEnv(BaseControlledEnv):
    def __init__(self, *args, env: BaseDroneEnv = None, dt: float = 1 / 240, **kwargs):
        if env is None:
            env = PyBulletDroneEnv(*args, dt=dt, **kwargs)
        super(AttitudeAltitudeDroneEnv, self).__init__(env=env, dt=dt)

    @staticmethod
    def _init_controllers(dt: float) -> Dict[str, PID]:
        y = PID_antiwindup(10, 0.1, 0.2, dt, angle=True)
        dy = PID_antiwindup(15, 0.01, 7, dt, gain=-10000)
        # p = PID_antiwindup(6, 0.1, 0.3, dt, angle=True, gain=1000)
        p = PID_antiwindup(6, 0.1, 2, dt, angle=True, gain=-1000)
        # r = PID_antiwindup(6, 0.1, 0.3, dt, angle=True, gain=1000)
        r = PID_antiwindup(6, 0.1, 2, dt, angle=True, gain=1000)

        z = PID_antiwindup(15, 3, 0.1, dt, gain=0.1)
        dz = PID_antiwindup(2, 0, 0.00001, dt, gain=100000)

        return dict(yaw=y, pitch=p, roll=r, daltitude=dz, dyaw=dy, altitude=z)

    def step(
        self, action: AttitudeAltitudeAction
    ) -> Tuple[State, float, bool, bool, dict]:
        roll, pitch, yaw, z = action

        spos = self.env.state.pos
        srpy = self.env.state.rpy
        sang_vel = self.env.state.ang_vel
        svel = self.env.state.vel

        # Calculate control action
        r_act = self.controllers["roll"](srpy[0], roll)

        p_act = self.controllers["pitch"](srpy[1], pitch)

        y_act = self.controllers["yaw"](srpy[2], yaw)
        dy_act = self.controllers["dyaw"](sang_vel[2], y_act)

        z_act = self.controllers["altitude"](spos[2], z)
        dz_act = self.controllers["daltitude"](svel[2], z_act)

        propellerAction = clip(
            self.env.model.rpyT2rpm(r_act, p_act, dy_act, dz_act),
            0,
            np.inf,  # 3 * self.env.model.weight / self.env.model.k_T,
        )

        self.env.info["control"] = dict(
            errors={name: ctrl.e for name, ctrl in self.controllers.items()}
        )

        return self.env.step(propellerAction)

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array(
            [
                (-0.2, 0.2),  # Roll
                (-0.2, 0.2),  # Pitch
                (-np.pi, np.pi),  # Yaw
                (1, np.inf),  # Altitude
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )


if __name__ == "__main__":
    import gymnasium

    import pandas as pd
    import numpy as np

    from tqdm.auto import tqdm

    # In[4]:

    import seaborn as sns
    import matplotlib.pyplot as plt

    # In[5]:

    from collections import deque

    # In[6]:

    T = 40
    dt = 1 / 240
    seed = 1337

    # In[7]:

    env = gymnasium.make("AttitudeAltitudeDroneEnv-v0", dt=dt)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=int(T / dt))

    # In[8]:

    observations = deque()

    obs, info = env.reset(seed=seed)
    setpoint = env.action_space.sample()
    trunc, term = False, False
    for i in tqdm(range(int(T / dt) - 1)):
        if i * dt % 10 == 0:
            setpoint = env.action_space.sample()
        obs, _, term, trunc, info = env.step(setpoint)
        observations.append(obs)

    # In[9]:

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

    # In[10]:

    df_long = df.melt(
        var_name="variable", value_name="value", ignore_index=False
    ).reset_index()

    # In[11]:

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
    ax[1].hlines(
        setpoint[:3],
        df.index.min(),
        df.index.max(),
        linestyles="dotted",
        label="setpoint",
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

    # In[12]:

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(df.x, df.y)

    fig.tight_layout()

    plt.show()
