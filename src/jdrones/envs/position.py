#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import abc
import collections
import itertools
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from jdrones.data_models import State
from jdrones.data_models import States
from jdrones.data_models import URDFModel
from jdrones.envs.base.basecontrolledenv import BaseControlledEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.lqr import LQRDroneEnv
from jdrones.trajectory import QuinticPolynomialTrajectory
from jdrones.types import PositionAction


class BasePositionDroneEnv(gymnasium.Env, abc.ABC):
    """
    Baseclass for other position drone environments. These are ones where step takes
    a :math:`(x,y,z)` argument and makes a drone fly from its current position to there.
    """

    env: BaseControlledEnv

    dt: float

    model: URDFModel

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
        self.model = model
        self.observation_space = spaces.Sequence(self.env.observation_space)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 1]), high=np.array([10, 10, 10])
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[States, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs, _ = self.env.reset(seed=seed, options=options)

        return States([np.copy(obs)]), {}

    @abc.abstractmethod
    def step(
        self, action: PositionAction
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        """
        A step from the viewpoint of a
        :class:`~jdrones.envs.position.BasePositionDroneEnv` is making the drone
        fly from its current position :math:`A` to the target position :math:`B`.


        Parameters
        ----------
        action : float,float,float
            Target coordinates :math:`(x,y,z)`

        Returns
        -------
            states: jdrones.data_models.States
            reward: float
            term: bool
            trunc: bool
            info: dict
        """
        pass

    @staticmethod
    def get_reward(states: States) -> float:
        """
        Calculate the cost of the segment with

        .. math::
            \\begin{gather}
            J = \\sum_{i=0}^{T/dt} Q \\frac{\\vec x_i}{dt}
            \\\\
            Q =
            \\left[
                1000,1000,1000,0,0,0,0,10,10,10,1,1,1,1,1,1,0,0,0,0
            \\right]
            \\\\
            \\vec x =
            \\left[
                x,y,z,q_x,q_y,q_z,q_w,\\phi,\\theta,\\psi,\\dot x,\\dot y,\\dot z,
                p,q,r,P_1,P_2,P_3,P_4
            \\right]
            \\end{gather}

        Where :math:`\\vec x_i` is the state matrix at time-step :math:`i`, and
        :math:`Q` is a cost matrix prioritizing error in position and angle.

        Parameters
        ----------
        states : jdrones.data_models.States
            Iterable containing the observed states at regular intervals :math:`dt`

        Returns
        -------
        float
            The calculated cost
        """
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
    """
    Uses :class:`jdrones.trajectory.QuinticPolynomialTrajectory` to give target
    position and velocity commands at every time point until the target is reached.
    If the time taken exceeds :math:`T`, the original target position is given as a raw
    input as in :class:`~jdrones.envs.position.LQRPositionDroneEnv`. However, if this
    were to happen, the distance is small enough to ensure stability.
    """

    def step(
        self, action: PositionAction
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        action_as_state = State()
        action_as_state.pos = action

        observations = collections.deque()
        traj = self.calc_traj(self.env.state, action_as_state, self.model.max_vel_ms)

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
        """
        Calculate the trajectory for the drone to traverse.

        Total time to traverse the polynomial is defined as

        .. math::
            T = \\frac{||x_{t=T}-x_{t=0}||}{v_\\max}

        to ensure dynamic compatibility.

        Parameters
        ----------
        cur : jdrones.data_models.State
            Current state
        tgt : jdrones.data_models.State
            Target state
        max_vel : float
            Maximum vehicle velocity

        Returns
        -------
        jdrones.trajectory.QuinticPolynomialTrajectory
            The solved trajectory
        """
        dist = np.linalg.norm(tgt.pos - cur.pos)
        T = dist / max_vel

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
    """
    Feeds the raw position setpoint into :class:`jdrones.envs.lqr.LQRDroneEnv` until
    the drones arrives.

    .. warning::
        This is unstable for large inputs (distances of over :math:`10m`) and should
        **not** be used.
    """

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
