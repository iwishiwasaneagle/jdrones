import sys
import time
from collections import deque
from itertools import count
from typing import Callable
from typing import Deque
from typing import Optional
from typing import Tuple

import gymnasium
import numpy as np
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from gymnasium.wrappers import TimeLimit
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.velocityheading import VelHeadAltDroneEnv
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

    cost_func: Callable[["PIDTrajectoryDroneEnv"], float]
    env: VelHeadAltDroneEnv

    observations: Deque[State]

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        simulation_type: SimulationType = SimulationType.DIRECT,
        dt: float = 1 / 240,
        cost_func: Callable[["PIDTrajectoryDroneEnv"], float] = None,
    ):
        self.env = VelHeadAltDroneEnv(
            model=model,
            initial_state=initial_state,
            simulation_type=simulation_type,
            dt=dt,
        )
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
        self.observations.clear()
        super().reset(seed=seed, options=options)
        return self.env.reset(seed=seed)

    def step(self, action: PositionAction) -> Tuple[States, float, bool, bool, dict]:

        self.observations.clear()

        dist = euclidean_distance(*self.env.state.pos[:2], *action[:2])

        timer = time.time()
        term, trunc = False, False
        for i in count():
            cur_pos = self.env.state.pos
            new_vel_mag = self.env.model.max_vel_ms * clip_scalar(dist, 0.1, 0.4)

            new_action: VelHeadAltAction = np.array(
                (new_vel_mag, 0, yaw(*cur_pos[:2], *action[:2]), action[2])
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


def pid_main():
    dt = 1 / 240
    model = DronePlus
    logger.debug(model)

    initial_state = State()
    initial_state.pos = [0, 0, 2]

    env = PIDTrajectoryDroneEnv(
        model=model,
        initial_state=initial_state,
        simulation_type=SimulationType.DIRECT,
        dt=dt,
    )
    env = TimeLimit(env, max_episode_steps=5)

    env.reset()

    setpoints = deque()
    rewards = deque()
    observations = deque()
    while True:
        setpoint: PositionAction = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(setpoint)
        observations.append(np.array(obs).copy())
        setpoints.append(setpoint)
        rewards.append(reward)
        if trunc or term:
            print(f"{trunc=} {term=} {info=}")
            break


if __name__ == "__main__":
    pid_main()
