#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Any
from typing import Optional

import numpy as np
from gymnasium import spaces
from jdrones.data_models import State
from jdrones.data_models import States
from jdrones.data_models import URDFModel
from jdrones.envs.base import BaseControlledEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.types import DType
from libjdrones import LQRDroneEnv as _LQRDroneEnv


class LQRDroneEnv(BaseControlledEnv):
    """
    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("LQRDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<LQRDroneEnv<LQRDroneEnv-v0>>>>
    """

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
    ):
        if initial_state is None:
            initial_state = State()
        self.initial_state = initial_state
        self.env = _LQRDroneEnv(dt, initial_state)

        bounds = np.ones((20, 2), dtype=DType) * np.inf
        bounds[:, 0] *= -1
        self.action_space = spaces.Box(low=bounds[:, 0], high=bounds[:, 1], dtype=DType)
        obs_bounds = np.array(
            [
                # XYZ
                # Position
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # Q 1-4
                # Quarternion rotation
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                # RPY
                # Roll pitch yaw rotation
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                # V XYZ
                # Velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # V RPY
                # Angular velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # P 0-4
                # Propeller speed
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
            ],
            dtype=DType,
        )
        self.observation_space = spaces.Box(
            low=obs_bounds[:, 0], high=obs_bounds[:, 1], dtype=DType
        )

    @property
    def state(self) -> State:
        return State(self.env.state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[State, dict]:
        super().reset(seed=seed, options=options)
        self.info = {}

        if options is not None:
            reset_state = options.get("reset_state", self.initial_state)
        else:
            reset_state = self.initial_state
        self.env.reset(np.copy(reset_state))
        return self.state, self.info

    def step(self, action: State) -> tuple[State, float, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc = self.env.step(action)
        return State(obs), rew, term, trunc, {}


if __name__ == "__main__":
    from collections import deque

    import matplotlib.pyplot as plt
    import seaborn as sns
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
