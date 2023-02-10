#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import sys
import time
from collections import deque
from itertools import count
from typing import Callable
from typing import Deque
from typing import List
from typing import Optional
from typing import Tuple

import gymnasium
import numpy as np
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.velocity import VelHeadAltDroneEnv
from jdrones.maths import clip_scalar
from jdrones.maths import euclidean_distance
from jdrones.maths import yaw
from jdrones.types import PositionAction
from jdrones.types import SimulationType
from jdrones.types import State
from jdrones.types import States
from jdrones.types import URDFModel
from jdrones.types import VelHeadAltAction
from loguru import logger

sys.setrecursionlimit(100000)


class PIDTrajectoryDroneEnv(gymnasium.Env):
    """
    Fly the drone to a waypoint using a PID trajectory
    """

    cost_func: Callable[["PIDTrajectoryDroneEnv"], float]
    """Custom cost function to evaluate at the end of every episode"""

    env: VelHeadAltDroneEnv
    """Base drone environment"""

    observations: Deque[State]
    """Log of observations over episode"""

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        simulation_type: SimulationType = SimulationType.DIRECT,
        dt: float = 1 / 240,
        cost_func: Callable[["PIDTrajectoryDroneEnv"], float] = None,
        wrappers: List[gymnasium.Wrapper] = None,
    ):
        self.env = VelHeadAltDroneEnv(
            model=model,
            initial_state=initial_state,
            simulation_type=simulation_type,
            dt=dt,
        )
        if wrappers is not None:
            for wrapper in wrappers:
                logger.debug(f"Applying {wrapper=} to env")
                self.env = wrapper(self.env)
        self.observation_space = self.env.observation_space

        if cost_func is None:
            self.cost_func = lambda s: 0.0

        self.observations = deque()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[States, dict]:
        """
        Reset the simulation to the initial state.

        .. seealso::
            :func:`gymnasium.Env.reset`

        Parameters
        ----------
        seed : int
            Seed to pass to gymnasium RNG
            (Default = None)
        options : dict
            Additional options to pass to gymnasium API
            (Default = None)

        Returns
        -------
        observation : Observation
            Observation of the initial state. It should be analogous to the info
            returned by :meth:`step`.
        info : dict
            This dictionary contains auxiliary information complementing observation.
            It should be analogous to the info returned by :meth:`step`.
        """
        self.observations.clear()
        super().reset(seed=seed, options=options)
        return self.env.reset(seed=seed)

    def step(self, action: PositionAction) -> Tuple[States, float, bool, bool, dict]:
        """
        Run one episode of the environment’s dynamics using the agent actions. The
        episode terminates if the drone gets within :math:`0.1\\m` of the target
        waypoint, or the base env returns :code:`term` or :code:`trunc`.

        - Yaw error is the difference between target and current yaw to the normal
        - :math:`v^b_x` (the body :math:`x` velocity) is set as follows:

        .. math::
            v_{tgt} &= ||\\vec x - \\vec x_{tgt}|| \\\\
            v^b_x &=
            \\begin{cases}
            0.1&v_{tgt}\\leq0.1 \\\\
            0.4&v_{tgt}\\geq0.4 \\\\
            v_{tgt}&else
            \\end{cases} \\\\
            v^b_y &= 0


        .. seealso::
            - :func:`gymnasium.Env.step`

        Parameters
        ----------
        action : PositionAction
            A waypoint action provided by the top-level flight controller in the form
            :math:`(x,y,z)`

        Returns
        -------
        observation : Deque[Observation]
            Observation of the states over the flight from initial position to the
            target position
        reward : float
             The reward as a result of taking the action, calculated by
             :attr:`PIDTrajectoryDroneEnv.cost_func`
        terminated : bool
             Whether the agent reaches the terminal state (as defined under the MDP of
             the task)
        truncated : bool
             Whether the truncation condition outside the scope of the MDP is satisfied
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, learning,
            and logging)
        """

        self.observations.clear()

        dist = euclidean_distance(*self.env.state.pos[:2], *action[:2])

        timer = time.time()
        term, trunc = False, False
        for i in count():
            cur_pos = self.env.state.pos
            cur_yaw = self.env.state.rpy[2]
            new_yaw = yaw(*cur_pos[:2], *action[:2])
            err_yaw = new_yaw - cur_yaw
            new_vel_mag = (1 - np.abs(err_yaw / np.pi)) * clip_scalar(dist, 0.1, 0.3)

            new_action: VelHeadAltAction = np.array(
                (new_vel_mag, 0, new_yaw, action[2])
            )
            obs, _, term, trunc, _ = self.env.step(new_action)

            self.observations.append(obs)
            dist = euclidean_distance(*cur_pos[:2], *action[:2])
            if dist <= 0.1:
                break
            if term or trunc:
                break

            if self.env.simulation_type == SimulationType.GUI:
                time.sleep(self.env.dt)

        total_t = time.time() - timer
        logger.debug(f"Completed {i} steps in {total_t:.2f}s ({i/total_t:.2f} it/s)")
        reward = self.cost_func(self)
        return np.array(self.observations), reward, term, trunc, {}

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array([(-10, 10), (-10, 10), (2, 3)])
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )
