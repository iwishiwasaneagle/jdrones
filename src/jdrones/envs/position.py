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
from jdrones.trajectory import BasePolynomialTrajectory
from jdrones.trajectory import FifthOrderPolynomialTrajectory
from jdrones.trajectory import FirstOrderPolynomialTrajectory
from jdrones.types import PositionAction
from jdrones.types import PositionVelocityAction
from jdrones.types import VEC3


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


class PolynomialPositionBaseDronEnv(BasePositionDroneEnv):
    @staticmethod
    def calc_traj(cur: State, tgt: State, max_vel: float = 1):
        raise NotImplementedError

    @staticmethod
    def update_u_from_traj(u: State, traj: BasePolynomialTrajectory, t: float):
        if hasattr(traj, "position"):
            u.pos = traj.position(t)
        if hasattr(traj, "velocity"):
            u.vel = traj.velocity(t)
        return u

    @staticmethod
    def _validate_action_input(
        action: PositionAction | PositionVelocityAction,
    ) -> State:
        """
        Validate the action input. Has to either be only position, or position and
        velocity

        Parameters
        ----------
        action : VEC3 | tuple[VEC3, VEC3]
            Target coordinates :math:`(x,y,z)` or target coordinates and velocity (
            :math:`(v_x,v_y,v_z)`)

        Returns
        -------
        State
            The valid and converted action as a :class:`~jdrones.data_models.State`

        """
        action_as_state = State()
        action_np = np.asarray(action).squeeze()
        if action_np.shape == (3,):
            action_as_state.pos = action_np
        elif action_np.shape == (2, 3):
            action_as_state.pos = action_np[0]
            action_as_state.vel = action_np[1]
        elif action_np.shape == (6,):
            action_as_state.pos = action_np[:3]
            action_as_state.vel = action_np[3:]
        else:
            raise ValueError(
                f"Incorrect shape {action_np.shape}. Expected either " f"(3,) or (2,3)"
            )
        return action_as_state

    def step(
        self, action: PositionAction | PositionVelocityAction
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        """
        A step from the viewpoint of a
        :class:`~jdrones.envs.position.BasePositionDroneEnv` is making the drone
        fly from its current position :math:`A` to the target position :math:`B`.


        Parameters
        ----------
        action : VEC3 | tuple[VEC3, VEC3]
            Target coordinates :math:`(x,y,z)` or target coordinates and velocity (
            :math:`(v_x,v_y,v_z)`)

        Returns
        -------
            states: jdrones.data_models.States
            reward: float
            term: bool
            trunc: bool
            info: dict
        """
        action_as_state = self._validate_action_input(action)

        term, trunc, info = False, False, {}
        if np.allclose(action, self.env.state.pos):
            # Avoid a singularity since the trajectory is scaled with distance
            term = True
            traj = None
        else:
            traj = self.calc_traj(
                self.env.state, action_as_state, self.model.max_vel_ms
            )

        observations = collections.deque()
        observations.append(self.env.state.copy())

        u: State = action_as_state.copy()

        counter = itertools.count(-1)
        while not (term or trunc):
            t = next(counter) * self.dt
            if t > traj.T:
                u.pos = action_as_state.pos
                u.vel = (0, 0, 0)
            else:
                u = self.update_u_from_traj(u, traj, t)

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


class FifthOrderPolyPositionDroneEnv(PolynomialPositionBaseDronEnv):
    """
    Uses :class:`jdrones.trajectory.FifthOrderPolynomialTrajectory` to give target
    position and velocity commands at every time point until the target is reached.
    If the time taken exceeds :math:`T`, the original target position is given as a raw
    input. However, if this were to happen, the distance is small enough to ensure
    stability.

    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("FifthOrderPolyPositionDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<FifthOrderPolyPositionDroneEnv<FifthOrderPolyPositionDroneEnv-v0>>>>

    """

    @staticmethod
    def calc_traj(
        cur: State, tgt: State, max_vel: float = 1
    ) -> FifthOrderPolynomialTrajectory:
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
        jdrones.trajectory.FifthOrderPolynomialTrajectory
            The solved trajectory
        """
        dist = np.linalg.norm(tgt.pos - cur.pos)
        T = dist / max_vel

        t = FifthOrderPolynomialTrajectory(
            start_pos=cur.pos,
            start_vel=cur.vel,
            start_acc=(0, 0, 0),
            dest_pos=tgt.pos,
            dest_vel=tgt.vel,
            dest_acc=(0, 0, 0),
            T=T,
        )
        return t


class FirstOrderPolyPositionDroneEnv(PolynomialPositionBaseDronEnv):
    """
    Uses :class:`jdrones.trajectory.FirstOrderPolynomialTrajectory` to give target
    position commands at every time point until the target is reached.
    If the time taken exceeds :math:`T`, the original target position is given as a raw
    input. However, if this were to happen, the distance is small enough to ensure
    stability.

    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("FirstOrderPolyPositionDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<FirstOrderPolyPositionDroneEnv<FirstOrderPolyPositionDroneEnv-v0>>>>
    """

    @staticmethod
    def calc_traj(
        cur: State, tgt: State, max_vel: float = 1
    ) -> FirstOrderPolynomialTrajectory:
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
        jdrones.trajectory.FirstOrderPolynomialTrajectory
            The solved trajectory
        """
        dist = np.linalg.norm(tgt.pos - cur.pos)
        T = dist / max_vel

        t = FirstOrderPolynomialTrajectory(
            start_pos=cur.pos,
            dest_pos=tgt.pos,
            T=T,
        )
        return t


class FifthOrderPolyPositionWithLookAheadDroneEnv(BasePositionDroneEnv):
    """
    Supplements the :class:`jdrones.env.FifthOrderPolyPositionDroneEnv` by including
    the next waypoints position in the trajectory generation calculation.

    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("FifthOrderPolyPositionWithLookAheadDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<FifthOrderPolyPositionWithLookAheadDroneEnv<FifthOrderPolyPositionWithLookAheadDroneEnv-v0>>>>
    """

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
        env: LQRDroneEnv = None,
    ):
        if env is None:
            env = FifthOrderPolyPositionDroneEnv(
                model=model, initial_state=initial_state, dt=dt, env=env
            )
        super().__init__(model=model, initial_state=initial_state, dt=dt, env=env)

        self.action_space = spaces.Box(
            low=np.array([[0, 0, 1], [0, 0, 1]]), high=10, shape=(2, 3)
        )

    @staticmethod
    def calc_v_at_B(A: VEC3, B: VEC3, C: VEC3, *, V: float, N: float = 3):
        """
        For the velocity vector at :math:`B`, it is ideal to create a smooth
        transition between the segments. One solution is to reduce velocity to 0.
        However, this is very inefficient and not representative of real-world flight,
        especially for the case where there are multiple waypoints in a straight line.
        The preferred, albeit slightly more computationally expensive, solution is to
        use the waypoints before and after to calculate a smooth velocity at :math:`B`.
        This is calculated by

        .. math::
            \\begin{gather}
                \\theta =
                    \\frac{(-\\vec r_{AB}) \\cdot \\vec r_{BC}}
                    {|\\vec r_{AB}| |\\vec r_{BC}|}, \\\\
                \\chi = \\frac{(1-\\theta)}{2}, \\\\
                \\vec v_{B,-} = \\vec v_{B,+} = V \\cdot
                    \\frac{\\vec r_{AC}}
                    {| \\vec r_{AC} |}
                    \\cdot \\chi^N,
            \\end{gather}


        Parameters
        ----------
        A : VEC3
            Current position
        B : VEC3
            Target position
        C : VEC3
            Position after going through :math:`B`
        V : float
            Scaling factor for the magnitude of :math:`\\vec v_B`
        N : float
            Scaling factor for the magnitude of :math:`\\chi`.
            :math:`N = 3` for an aggressive reduction in velocity for non-straight line
            waypoints to closely follow the path. :math:`N = 1` would be more suitable
            for cases where following the path is not as important.
            (Default = 3)

        Returns
        -------
        VEC3
            The velocity at :math:`B`
        """

        rAC = C - A
        rBA = A - B
        rBC = C - B
        rAC_unit = rAC / np.clip(np.linalg.norm(rAC), 1e-6, np.inf)
        theta = rBA.dot(rBC) / (np.linalg.norm(rBA) * np.linalg.norm(rBC))
        chi = np.power((1 - theta) / 2, N)
        return V * rAC_unit * chi

    def step(
        self, action: tuple[PositionAction, PositionAction]
    ) -> tuple[States, float, bool, bool, dict[str, Any]]:
        """
        A step from the viewpoint of a
        :class:`~jdrones.envs.position.
        FifthOrderPolyPositionWithLookAheadDroneEnv` is making the drone
        fly from its current position :math:`A` to the target position :math:`C` via
        :math:`B`.

        .. seealso::
            :meth:`~jdrones.envs.position.
            FifthOrderPolyPositionDroneEnvWithLookAheadDroneEnv.calc_v_at_B`

        Parameters
        ----------
        action : tuple[VEC3, VEC3]
            Target coordinates :math:`B=(x_1,y_1,z_1)` and next target coordinates
            :math:`C=(x_2,y_2,z_2)`

        Returns
        -------
            states: jdrones.data_models.States
            reward: float
            term: bool
            trunc: bool
            info: dict
        """
        A, B, C = np.array([self.env.env.state.pos, *action])
        if np.allclose(B, C):
            v_at_B = (0, 0, 0)
        else:
            v_at_B = self.calc_v_at_B(A, B, C, V=self.model.max_vel_ms * 0.3)
        tgt_pos = B
        tgt_vel = v_at_B
        return self.env.step((tgt_pos, tgt_vel))
