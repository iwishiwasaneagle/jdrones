#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import sys
import time
from itertools import count
from typing import Callable
from typing import Deque
from typing import List
from typing import Optional
from typing import Tuple

import gymnasium
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from jdrones.controllers import PID
from jdrones.envs.base import BaseControlledEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.velocity import VelHeadAltDroneEnv
from jdrones.maths import apply_rpy
from jdrones.maths import clip_scalar
from jdrones.maths import euclidean_distance
from jdrones.types import PositionAction
from jdrones.types import SimulationType
from jdrones.types import State
from jdrones.types import States
from jdrones.types import URDFModel
from jdrones.types import VEC3
from jdrones.types import VelHeadAltAction
from loguru import logger

sys.setrecursionlimit(100000)


class PIDTrajectoryDroneEnv(BaseControlledEnv):
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
        env = VelHeadAltDroneEnv(
            model=model,
            initial_state=initial_state,
            simulation_type=simulation_type,
            dt=dt,
        )
        if wrappers is not None:
            for wrapper in wrappers:
                logger.debug(f"Applying {wrapper=} to env")
                env = wrapper(env)

        super().__init__(env=env, dt=dt)

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

    @staticmethod
    def _init_controllers(dt: float) -> dict[str, PID]:
        body_pos_x = PID_antiwindup(1, 0, 0, dt=dt)
        body_pos_y = PID_antiwindup(1, 0, 0, dt=dt, gain=-1)
        return dict(p_b_x=body_pos_x, p_b_y=body_pos_y)

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

        timer = time.time()
        term, trunc = False, False
        for i in count():
            cur_rpy = self.env.state.rpy
            cur_pos = self.env.state.pos
            cur_pos_b, tgt_pos_b = self._position_world_to_body(
                cur_pos, action, cur_rpy
            )

            u_v_x_b = self.controllers["p_b_x"](cur_pos_b[0], tgt_pos_b[0])
            u_v_y_b = self.controllers["p_b_y"](cur_pos_b[1], tgt_pos_b[1])

            u_v_x_b = clip_scalar(u_v_x_b, -2, 2)
            u_v_y_b = clip_scalar(u_v_y_b, -2, 2)

            new_action: VelHeadAltAction = np.array((u_v_x_b, u_v_y_b, 0, action[2]))
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

    @staticmethod
    def _position_world_to_body(
        cur_pos: VEC3, tgt_pos: VEC3, rpy: VEC3
    ) -> tuple[VEC3, VEC3]:
        """
        Pure function to convert world to body position. Primarily to allow testing

        .. seealso::

            - S. A. A. Moosavian and A. Kalantari, ‘Experimental Slip Estimation for
            Exact Kinematics Modeling and Control of a Tracked Mobile Robot’, Oct. 2008,
            pp. 95–100. doi: 10.1109/IROS.2008.4650798.


        Parameters
        ----------
        cur_pos : (float,float,float)
            Current position in world coords
        tgt_pos : (float,float,float)
            Target position in world coords
        rpy : (float,float,float)
            Roll pitch yaw angles

        Returns
        -------
        cur_pos_ned: (float,float,float)
            Current position in body
        tgt_pos_ned: (float,float,float)
            Current target in body
        """
        # Zero on cur pos
        yaw = np.array((0, 0, rpy[2]))
        return np.array((0, 0, 0)), apply_rpy(tgt_pos - cur_pos, yaw)

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array([(-10, 10), (-10, 10), (2, 3)])
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )


if __name__ == "__main__":

    import gymnasium
    import jdrones
    import jdrones.types
    from jdrones.controllers import PID_antiwindup

    import pandas as pd
    import numpy as np

    from functools import partial
    from collections import deque

    # In[4]:

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    T = 200
    dt = 1 / 240
    seed = 1337

    initial_state = jdrones.types.State()
    initial_state.pos = (0, 0, 2)

    # In[20]:

    env = gymnasium.make(
        "PIDTrajectoryDroneEnv-v0",
        dt=dt,
        initial_state=initial_state,
        wrappers=[partial(gymnasium.wrappers.TimeLimit, max_episode_steps=int(T / dt))],
    )

    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=10)

    # In[21]:

    observations = deque()
    setpoints = deque()

    obs, info = env.reset(seed=seed)
    trunc, term = False, False
    while not (trunc or term):
        setpoint = env.action_space.sample()
        setpoints.append(setpoint)
        obs, _, term, trunc, info = env.step(setpoint)
        observations.append(obs)

    # In[23]:

    dfs = []
    t_prev = 0
    for i, obs in enumerate(observations):
        t = t_prev + np.linspace(0, len(obs) * dt, len(obs))
        df = pd.DataFrame(
            obs,
            columns=[
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "phi",
                "theta",
                "psi",
                "vx",
                "vy",
                "vz",
                "p",
                "q",
                "r",
                "P0",
                "P1",
                "P2",
                "P3",
            ],
            index=t,
        )
        df.index.name = "t"
        df["segment"] = i
        dfs.append(df)
        t_prev = t[-1] + dt

    df = pd.concat(dfs)
    df = df.iloc[:: int(len(df) / 1000), :]  # Select only every 5th row for performance

    # In[24]:

    df_long = df.melt(
        var_name="variable", value_name="value", id_vars=["segment"], ignore_index=False
    ).reset_index()

    # In[25]:

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax = ax.flatten()

    sns.lineplot(
        data=df_long.query("variable in ('x','y','z')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[0],
    )
    ax[0].legend()

    sns.lineplot(
        data=df_long.query("variable in ('phi','theta','psi')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[1],
    )
    ax[1].legend()

    sns.lineplot(
        data=df_long.query("variable in ('vx','vy','vz')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[2],
    )
    ax[2].legend()

    sns.lineplot(
        data=df_long.query("variable in ('P0','P1','P2','P3')"),
        x="t",
        y="value",
        hue="variable",
        ax=ax[3],
    )
    ax[3].legend()

    fig.tight_layout()

    plt.show()

    # In[26]:

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    segments = df.segment.unique()
    setpoints_data = np.array(setpoints)

    cmap = mpl.cm.get_cmap("plasma", len(segments))
    c = np.arange(1, len(segments) + 1)
    dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
    ax.cla()

    for i, s in enumerate(df.segment.unique()):
        df_subset = df[df.segment == s]
        ax.plot(df_subset.x, df_subset.y, c=cmap(i))
        a = ax.scatter(
            setpoints_data[i, 0], setpoints_data[i, 1], color=cmap(i), label="Waypoint"
        )
    b = ax.scatter(df.iloc[0].x, df.iloc[0].y, 100, marker="x", label="Start")
    leg = ax.legend(handles=(a, b))
    for legenhandle in leg.legendHandles:
        legenhandle.set_color("black")

    fig.colorbar(dummie_cax, ticks=c, label="Order")
    fig.tight_layout()
    plt.show()
