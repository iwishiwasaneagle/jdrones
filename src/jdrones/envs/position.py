#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import collections
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from jdrones.data_models import State
from jdrones.data_models import States
from jdrones.data_models import URDFModel
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.lqr import LQRDroneEnv
from jdrones.types import PositionAction


class PositionDroneEnv(gymnasium.Env):
    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
        env: LQRDroneEnv = None,
    ):
        if env is None:
            env = LQRDroneEnv(model=model, initial_state=initial_state, dt=dt)
        self.env = env
        self.dt = dt
        self.observation_space = spaces.Sequence(self.env.observation_space)

    @property
    def action_space(self) -> ActType:
        bounds = np.array([[0, 10], [0, 10], [1, 10]])
        return spaces.Box(low=bounds[:, 0], high=bounds[:, 1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[States, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs, _ = self.env.reset(seed=seed, options=options)

        return States([np.copy(obs)]), {}

    def step(
        self, action: PositionAction
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        x_action = State()
        x_action.pos = action
        u = x_action.to_x()

        observations = collections.deque()

        term, trunc, info = False, False, {}
        while not (term or trunc):
            obs, _, term, trunc, info = self.env.step(u)

            observations.append(np.copy(obs))

            e = self.env.controllers["lqr"].e
            dist = np.linalg.norm(np.concatenate([e.pos, e.rpy]))
            if np.any(np.isnan(dist)):
                trunc = True

            if dist < 0.01:
                term = True
                info["error"] = dist

        states = States(observations)
        return states, self.get_reward(states), term, trunc, info

    @staticmethod
    def get_reward(states: States) -> float:
        df = states.to_df(tag="temp")
        df_sums = (
            df.sort_values("t")
            .groupby(df.variable)
            .apply(lambda r: np.trapz(r.value.abs(), x=r.t))
        )

        df_sums[["P0", "P1", "P2", "P3"]] = 0
        df_sums[["qw", "qx", "qy", "qz"]] = 0
        df_sums[["x", "y", "z"]] *= 1000
        df_sums[["phi", "theta", "psi"]] *= 10

        return df_sums.sum()
