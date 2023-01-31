from typing import Tuple

import numpy as np
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from gymnasium.wrappers import TimeLimit
from jdrones.envs.velocityheading import VelHeadAltDroneEnv
from jdrones.types import Action
from jdrones.types import Observation


class PositionDroneEnv(VelHeadAltDroneEnv):
    @staticmethod
    def _calc_target_yaw(curpos, tgtpos):
        yaw = np.arctan2(tgtpos[1] - curpos[1], tgtpos[0] - curpos[0])
        return yaw

    @staticmethod
    def _calc_dist_to_target(curpos, tgtpos):
        return np.linalg.norm(tgtpos - curpos)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        x, y, z, vx_b = action
        # Yaw to target point
        yaw = self._calc_target_yaw(self.state.pos, (x, y, z))

        obs, _, term, trunc, info = super().step([vx_b, 0, yaw, z])
        reward = self.get_reward()
        return obs, reward, term, trunc, info

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array([(-10, 10), (-10, 10), (2, 2), (0.2, 0.2)])
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )


if __name__ == "__main__":
    import copy
    from collections import deque
    from itertools import count

    import sys

    from loguru import logger
    from tqdm.auto import tqdm

    from jdrones.envs.dronemodels import DronePlus
    from jdrones.types import State, SimulationType
    import matplotlib.pyplot as plt

    sys.setrecursionlimit(100000)

    T = 50
    dt = 1 / 240
    model = DronePlus
    logger.debug(model)

    initial_state = State()
    initial_state.pos = [0, 0, 2]
    initial_state.vel = [0, 0, 0]

    env = PositionDroneEnv(
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
    observations = deque()
    setpoints = deque()
    controller_errors = deque()
    rewards = deque()

    setpoint = [1, 1, 2, 0.1]
    print(f"\nCurrent goal is {setpoint}")
    setpoints.append(setpoint)
    while not (trunc or term):
        obs, reward, term, trunc, info = env.step((setpoint))
        pbar.update(1)
        observations.append(copy.copy(obs))
        rewards.append(reward)
        controller_errors.append(info["control"]["errors"])

    fig, ax = plt.subplots(2, 1)

    data = np.array(observations)
    ax[0].plot(data[:, 0], data[:, 1])
    ax[1].plot(data[:, 0], label="x")
    ax[1].plot(data[:, 1], label="y")
    ax[1].plot(data[:, 2], label="z")
    ax[1].legend()

    plt.show()
