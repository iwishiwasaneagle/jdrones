#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Tuple

import numpy as np
from gymnasium import spaces
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.transforms import euler_to_quat
from jdrones.transforms import euler_to_rotmat
from jdrones.types import PropellerAction
from jdrones.types import VEC5


class RotatingNonlinearDynamicModelDroneEnv(BaseDroneEnv):
    @property
    def action_space(self):
        act_bounds = np.array(
            [
                (0.0, 1e6),  # R1
                (0.0, 1e6),  # R2
                (0.0, 1e6),  # R3
                (0.0, 1e6),  # R4
                (-np.pi, np.pi),  # Alpha
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )

    @staticmethod
    def calc_dstate(action: VEC5, state: State, model: URDFModel):
        Inertias = np.diag(model.I)
        m = model.mass
        g = model.g
        length = model.l
        P = action[:4]
        alpha = action[4]

        T = model.k_T * np.square(P)

        P1, P2, P3, P4 = P
        T1, T2, T3, T4 = T

        unit_z = np.array([0, 0, 1]).reshape((-1, 1))

        R_W_B = euler_to_rotmat(state.rpy)
        R_B_W = np.linalg.inv(R_W_B)

        dstate = np.concatenate(
            [
                state.vel,
                (0, 0, 0, 0),
                state.ang_vel,
                (
                    -m * g * unit_z.T
                    + (
                        R_B_W
                        @ [
                            (T4 - T2) * np.sin(alpha),
                            (T3 - T1) * np.sin(alpha),
                            T.sum() * np.cos(alpha),
                        ]
                    ).T
                ).flatten()
                / m,
                np.linalg.solve(
                    Inertias,
                    (
                        R_B_W
                        @ [
                            length * (T4 - T2) * np.cos(alpha),
                            length * (T3 - T1) * np.cos(alpha),
                            length * T.sum() * np.sin(alpha)
                            + model.k_Q
                            + (-P1 * P1 + P2 * P2 - P3 * P3 + P4 * P4) * np.cos(alpha),
                        ]
                    ),
                ),
                (0, 0, 0, 0),
            ]
        )
        return dstate

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        # Get state
        dstate = self.calc_dstate(action, self.state, self.model)

        # Update step
        self.state += self.dt * dstate

        # Update derived state items
        self.state.prop_omega = action[:4]
        self.state.quat = euler_to_quat(self.state.rpy)

        # Return
        return self.state, 0, False, False, self.info


if __name__ == "__main__":
    from collections import deque

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from jdrones.data_models import States
    from tqdm.auto import trange
    from loguru import logger

    def _simulate_env(env, T, dt):
        dq = deque()
        env.reset()
        trim = np.sqrt(1.4 * 9.81 / (0.1 * 4))
        setpoint = np.array([*(trim,) * 4, 1e-6])
        for _ in trange(int(T / dt)):
            obs, *_ = env.step(setpoint)
            dq.append(np.copy(obs))
        return dq

    T = 5
    dt = 1 / 240

    initial_state = State()
    initial_state.pos = (0, 0, 10)
    initial_state.rpy = (0, 0, 0)

    nl_env = RotatingNonlinearDynamicModelDroneEnv(
        initial_state=initial_state,
        dt=dt,
    )
    logger.debug("Simulating envs")
    df = pd.concat(
        [
            States(_simulate_env(f, T, dt)).to_df(tag=type(f).__name__, N=200, dt=dt)
            for f in [nl_env]
        ]
    ).reset_index()
    logger.debug("Plotting grouped states")
    print(df.head())

    fig, ax = plt.subplots(4, figsize=(10, 8))
    ax = ax.flatten()
    for i, vars in enumerate(
        ["'x','y','z'", "'phi','theta','psi'", "'vx','vy','vz'", "'P0','P1','P2','P3'"]
    ):
        sns.lineplot(
            data=df.query(f"variable in ({vars})"),
            x="t",
            y="value",
            hue="variable",
            style="tag",
            ax=ax[i],
        )
        ax[i].legend()
    fig.tight_layout()
    plt.show()

    logger.debug("Plotting individual states")
    fig, ax = plt.subplots(9, figsize=(14, 10))
    ax = ax.flatten()
    for i, var in enumerate(("x", "y", "z", "phi", "theta", "psi", "vx", "vy", "vz")):
        sns.lineplot(
            data=df.query(f"variable in ('{var}')"),
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
