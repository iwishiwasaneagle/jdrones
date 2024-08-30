#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jdrones.data_models import State
from jdrones.data_models import States
from jdrones.envs import LinearDynamicModelDroneEnv
from jdrones.envs import NonlinearDynamicModelDroneEnv
from tqdm.auto import trange


def _simulate_env(env, T, dt):
    dq = deque()
    env.reset()
    setpoint = np.array([1, 1, 1, 1]) * 20
    for _ in trange(int(T / dt)):
        obs, *_ = env.step(setpoint)
        dq.append(np.copy(obs))
    return dq


T = 100
dt = 1 / 240

initial_state = State()
initial_state.pos = (0, 0, 10)
initial_state.rpy = (0, 0, 0)

nl_env = NonlinearDynamicModelDroneEnv(
    initial_state=initial_state,
    dt=dt,
)
l_env = LinearDynamicModelDroneEnv(
    initial_state=initial_state,
    dt=dt,
)
df_long = pd.concat(
    [
        States(_simulate_env(f, T, dt)).to_df(tag=type(f).__name__, dt=dt)
        for f in [nl_env, l_env]
    ]
).reset_index()

fig, ax = plt.subplots(4, figsize=(10, 8))
ax = ax.flatten()
for i, vars in enumerate(
    ["'x','y','z'", "'phi','theta','psi'", "'vx','vy','vz'", "'P0','P1','P2','P3'"]
):
    sns.lineplot(
        data=df_long.query(f"variable in ({vars})"),
        x="t",
        y="value",
        hue="variable",
        style="tag",
        ax=ax[i],
    )
    ax[i].legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(9, figsize=(14, 10))
ax = ax.flatten()
for i, var in enumerate(("x", "y", "z", "phi", "theta", "psi", "vx", "vy", "vz")):
    sns.lineplot(
        data=df_long.query(f"variable in ('{var}')"),
        x="t",
        y="value",
        hue="variable",
        style="tag",
        ax=ax[i],
        legend=False,
    )
    ax[i].set_ylabel(var)
fig.tight_layout()
plt.show()
