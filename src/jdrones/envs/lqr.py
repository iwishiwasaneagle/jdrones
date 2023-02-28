#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Any
from typing import Optional

import numpy as np
from gymnasium import spaces
from jdrones.controllers import Controller
from jdrones.controllers import LQR
from jdrones.envs.base import BaseControlledEnv
from jdrones.envs.base import LinearDynamicModelDroneEnv
from jdrones.envs.base import NonlinearDynamicModelDroneEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.types import PositionAction
from jdrones.types import State
from jdrones.types import URDFModel


class LQRDroneEnv(BaseControlledEnv):
    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
        env: NonlinearDynamicModelDroneEnv = None,
        Q=None,
        R=None,
    ):
        if env is None:
            env = NonlinearDynamicModelDroneEnv(
                model=model, initial_state=initial_state, dt=dt
            )

        if Q is None:
            self.Q = np.diag(
                [
                    8.991192078204383e-05,
                    8.806431753269343e-05,
                    8.091799938753304e-05,
                    3.632257365747144e-05,
                    6.764156497790077e-06,
                    2.328062112946342e-05,
                    3.28709705868307e-05,
                    3.238878003477344e-05,
                    4.672184058816576e-05,
                    8.04579280110124e-05,
                    7.189365004554992e-05,
                    8.188944953864152e-05,
                ]
            )
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag(
                [
                    2.094619920797816,
                    4.1435633102757645,
                    42.47949524890384,
                    4.854927340822368e-05,
                ]
            )
        else:
            self.R = R

        super().__init__(env, dt)

    def _init_controllers(self) -> dict[str, Controller]:
        A, B, _ = LinearDynamicModelDroneEnv.get_matrices(self.env.model)
        return dict(lqr=LQR(A, B, self.Q, self.R))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[State, dict]:
        return super().reset(seed=seed, options=options)

    def step(
        self, action: PositionAction
    ) -> tuple[State, float, bool, bool, dict[str, Any]]:

        setpoint = np.concatenate([action, np.zeros(17)])

        action = self.controllers["lqr"](measured=self.env.state, setpoint=setpoint)
        action_with_linearization_assumptions = np.sqrt(
            np.clip(
                self.env.model.rpyT2rpm(
                    [0, 0, 0, self.env.model.mass * self.env.model.g] + action
                ),
                0,
                np.inf,
            )
        )
        obs, _, trunc, term, _ = self.env.step(action_with_linearization_assumptions)

        return obs, 0, trunc, term, {}

    @property
    def action_space(self):
        bounds = np.ones((3, 2)) * np.inf
        bounds[:, 0] *= -1

        return spaces.Box(low=bounds[:, 0], high=bounds[:, 1])


if __name__ == "__main__":

    from collections import deque

    import matplotlib.pyplot as plt
    import seaborn as sns
    from jdrones.types import State, States
    from tqdm.auto import trange

    def _simulate_env(env, T, dt):
        dq = deque()
        env.reset()
        setpoint = [0, 0, 0]
        for _ in trange(int(T / dt)):
            obs, *_ = env.step(setpoint)
            dq.append(np.copy(obs))
        return dq

    T = 50
    dt = 1 / 240

    initial_state = State()
    initial_state.pos = (1, 1, 1)
    initial_state.prop_omega = np.ones(4) * 40

    env = LQRDroneEnv(initial_state=initial_state, dt=dt)

    df_long = States(_simulate_env(env, T, dt)).to_df(tag="LQR", dt=dt)

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
