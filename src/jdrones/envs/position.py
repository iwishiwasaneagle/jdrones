#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import collections
import itertools
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
from jdrones.trajectory import QuinticPolynomialTrajectory
from jdrones.types import PositionAction
from loguru import logger


class BasePositionDroneEnv(gymnasium.Env):
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


class PolyPositionDroneEnv(BasePositionDroneEnv):
    def step(
        self, action: PositionAction
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        action_as_state = State()
        action_as_state.pos = action

        observations = collections.deque()
        traj = self.calc_traj(self.env.state, action_as_state)

        u: State = action_as_state.copy()

        term, trunc, info = False, False, {}
        counter = itertools.count(-1)
        while not (term or trunc):
            t = next(counter) * self.dt
            if t > traj.T:
                u.pos = action
                u.vel = (0, 0, 0)
            else:
                u.pos = traj.position(t)
                u.vel = traj.velocity(t)

            obs, _, term, trunc, info = self.env.step(u.to_x())

            observations.append(obs.copy())

            dist = np.linalg.norm(self.env.state.pos - action_as_state.pos)
            if np.any(np.isnan(dist)):
                trunc = True

            if dist < 0.01:
                term = True
                info["error"] = dist

        states = States(observations)
        return states, self.get_reward(states), term, trunc, info

    @staticmethod
    def calc_traj(
        cur: State, tgt: State, max_vel: float = 1
    ) -> QuinticPolynomialTrajectory:
        logger.debug(f"Generating polynomial trajectory from {cur.pos} to {tgt.pos}")

        dist = np.linalg.norm(tgt.pos - cur.pos)
        T = dist / max_vel
        logger.debug(f"Total time for polynomial is {T=:.2f}s")

        t = QuinticPolynomialTrajectory(
            start_pos=cur.pos,
            start_vel=cur.vel,
            start_acc=(0, 0, 0),
            dest_pos=tgt.pos,
            dest_vel=tgt.vel,
            dest_acc=(0, 0, 0),
            T=T,
        )
        return t


class LQRPositionDroneEnv(BasePositionDroneEnv):
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
