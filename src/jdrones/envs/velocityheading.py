from collections import namedtuple
from typing import Optional, Tuple, Dict

import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.vector.utils import spaces

from jdrones.controllers import PID_antiwindup, PID
from jdrones.envs.attitude import AttitudeAltitudeDroneEnv
from jdrones.transforms import euler_to_rotmat
from jdrones.types import (
    AttitudeAltitudeAction,
    VelHeadAltAction,
    VEC3, Action, Observation,
)


class VelHeadAltDroneEnv(AttitudeAltitudeDroneEnv):
    def _init_controllers(self) -> Dict[str, PID]:
        ctrls = super()._init_controllers()

        vx_b = PID_antiwindup(1, 0.05, 1, self.dt, gain=0.1)
        vy_b = PID_antiwindup(1, 0.05, 1, self.dt, gain=-0.1)

        return {**ctrls, "vx_b": vx_b, "vy_b": vy_b}

    @staticmethod
    def _vi_to_vb(vel: VEC3, rpy: VEC3) -> VEC3:
        return np.dot(vel, euler_to_rotmat(rpy))

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        action = VelHeadAltAction(action)
        # Convert x-y from inertial to body
        vx_b_m, vy_b_m, _ = self._vi_to_vb(self.state.vel, self.state.rpy)
        # Calculate control action
        vx_act = self.controllers["vx_b"](vx_b_m, action.vx_b)
        vy_act = self.controllers["vy_b"](vy_b_m, action.vy_b)

        attalt_act = [vy_act, vx_act, action.vy_b, action.z]
        return super().step(attalt_act)

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array(
            [
                (-1, 1),  # vx
                (-1, 1),  # vy
                (-np.pi, np.pi),  # Yaw
                (1.0, np.inf),  # Altitude
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )


if __name__ == "__main__":
    from collections import deque
    from itertools import count

    from gymnasium.wrappers import TimeLimit
    import sys

    from loguru import logger
    from tqdm.auto import tqdm

    from jdrones.envs.dronemodels import DronePlus
    from jdrones.types import State, SimulationType

    sys.setrecursionlimit(100000)

    T = 300
    dt = 1 / 240
    model = DronePlus
    logger.debug(model)

    initial_state = State()
    initial_state.pos = [0, 0, 2]
    env = VelHeadAltDroneEnv(
        model=model,
        initial_state=initial_state,
        simulation_type=SimulationType.DIRECT,
        dt=dt,
    )
    env = TimeLimit(env, max_episode_steps=int(T / dt))

    env.reset()

    trunc, term = (False,) * 2
    c = count()

    pbar = tqdm(range(int(T / dt)), desc="Running simulation")
    controller_errors = deque()
    setpoint = env.action_space.sample()
    while not (trunc or term):
        obs, _, term, trunc, info = env.step(setpoint)
        pbar.update(1)

