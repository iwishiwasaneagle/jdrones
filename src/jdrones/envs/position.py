from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.vector.utils import spaces
from gymnasium.wrappers import TimeLimit

from jdrones.controllers import PID_antiwindup, PID
from jdrones.envs.velocityheading import VelHeadAltDroneEnv
from jdrones.types import (
    VelHeadAltAction,
    PositionVelAction, Action, Observation,
)


class PositionDroneEnv(VelHeadAltDroneEnv):
    @staticmethod
    def _calc_target_yaw(curpos, tgtpos):
        yaw = np.arctan2(tgtpos[1] - curpos[1], tgtpos[0] - curpos[0])
        return yaw

    @staticmethod
    def _calc_dist_to_target(curpos, tgtpos):
        return np.linalg.norm(tgtpos - curpos)

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, bool, dict]:
        action = PositionVelAction(action)
        self._tgt_pos = action[:3]
        # Yaw to target point
        yaw = self._calc_target_yaw(self.state.pos, action)
        vx_b = action.vx_b

        obs, _, term, trunc, info = super().step(
            [vx_b, 0, yaw, action.z]
        )
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
    from jdrones.plotting import plot_observations, plot_controller_errors, plot_xy
    from jdrones.types import State, SimulationType

    sys.setrecursionlimit(100000)

    T = 200
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

    setpoint = [*(env.state.pos + [0,1,0]),0.1]
    print(f"\nCurrent goal is {setpoint}")
    setpoints.append(setpoint)
    while not (trunc or term):
        obs, reward, term, trunc, info = env.step(PositionVelAction(setpoint))
        if (1/reward) < 0.1:
            try:
                setpoint = [*(env.state.pos + [1,0,0]),0.1]
                setpoints.append(setpoint)
                print(f"\nNext goal is {setpoint}")
            except StopIteration:
                print(f"\nFinal goal reached")
                break
        pbar.update(1)
        observations.append(copy.copy(obs))
        rewards.append(reward)
        controller_errors.append(info["control"]["errors"])

    plot_observations(observations, dt)
    plot_controller_errors(controller_errors, dt)
    plot_xy(observations, setpoints)

    fig, ax = plt.subplots()

    ax.plot(np.array(rewards))

    plt.show()
