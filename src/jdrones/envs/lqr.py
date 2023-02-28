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
from jdrones.types import LinearXAction
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
                    0.00011725206793179842,
                    0.00014153036731153807,
                    0.00016948868147463857,
                    5.677476140415448e-06,
                    2.2138672682961878e-05,
                    4.62158397517201e-06,
                    4.665673538059717e-06,
                    2.247699515384433e-05,
                    4.885363158451726e-05,
                    5.6631099121158025e-05,
                    1.746331462178498e-05,
                    6.822376227547066e-05,
                ]
            )
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag(
                [
                    0.28770346584971673,
                    0.6116377458968236,
                    41.55246938247595,
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
        self, action: LinearXAction
    ) -> tuple[State, float, bool, bool, dict[str, Any]]:

        setpoint = State.from_x(action)

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
        bounds = np.ones((12, 2)) * np.inf
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
        setpoint = np.zeros(12)
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
