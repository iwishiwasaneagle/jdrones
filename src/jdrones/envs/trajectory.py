import sys
import time
from collections import deque
from itertools import count
from typing import Optional
from typing import Tuple

import numpy as np
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from gymnasium.wrappers import TimeLimit
from jdrones.envs.dronemodels import DronePlus
from jdrones.envs.position import PositionDroneEnv
from jdrones.maths import clip_scalar
from jdrones.types import Action
from jdrones.types import Observation
from jdrones.types import PositionAction
from jdrones.types import State
from jdrones.types import VEC3
from jdrones.types import VEC4
from loguru import logger

sys.setrecursionlimit(100000)


class Trajectory:
    def __init__(
        self,
        start_pos: VEC3,
        dest_pos: VEC3,
        T: float = 5,
        start_vel: VEC3 = (0, 0, 0),
        dest_vel: VEC3 = (0, 0, 0),
        start_acc: VEC3 = (0, 0, 0),
        dest_acc: VEC3 = (0, 0, 0),
    ):

        self.start_pos = start_pos
        self.dest_pos = dest_pos

        self.start_vel = start_vel
        self.dest_vel = dest_vel

        self.start_acc = start_acc
        self.dest_acc = dest_acc

        self.T = T

        self._solve()

    def _solve(self) -> None:
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],  # f(t=0)
                [
                    self.T**5,
                    self.T**4,
                    self.T**3,
                    self.T**2,
                    self.T,
                    1,
                ],  # f(t=T)
                [0, 0, 0, 0, 1, 0],  # f'(t=0)
                [
                    5 * self.T**4,
                    4 * self.T**3,
                    3 * self.T**2,
                    2 * self.T,
                    1,
                    0,
                ],  # f'(t=T)
                [0, 0, 0, 2, 0, 0],  # f''(t=0)
                [20 * self.T**3, 12 * self.T**2, 6 * self.T, 2, 0, 0],  # f''(t=T)
            ]
        )

        b = np.row_stack(
            [
                self.start_pos,
                self.dest_pos,
                self.start_vel,
                self.dest_vel,
                self.start_acc,
                self.dest_acc,
            ]
        )

        self.coeffs = dict(
            x=np.linalg.solve(A, b[:, 0]),
            y=np.linalg.solve(A, b[:, 1]),
            z=np.linalg.solve(A, b[:, 2]),
        )

    def acceleration(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: 20 * c[0] * t**3
            + 12 * c[1] * t**2
            + 6 * c[2] * t
            + 2 * c[3]
        )
        xdd = calc(self.coeffs["x"], t)
        ydd = calc(self.coeffs["y"], t)
        zdd = calc(self.coeffs["z"], t)
        ret = (xdd, ydd, zdd)
        return ret

    def velocity(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: 5 * c[0] * t**4
            + 4 * c[1] * t**3
            + 3 * c[2] * t**2
            + 2 * c[3] * t
            + c[4]
        )
        xd = calc(self.coeffs["x"], t)
        yd = calc(self.coeffs["y"], t)
        zd = calc(self.coeffs["z"], t)
        ret = (xd, yd, zd)
        return ret

    def position(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: c[0] * t**5
            + c[1] * t**4
            + c[2] * t**3
            + c[3] * t**2
            + c[4] * t
            + c[5]
        )
        x = calc(self.coeffs["x"], t)
        y = calc(self.coeffs["y"], t)
        z = calc(self.coeffs["z"], t)
        ret = (x, y, z)
        return ret


class TrajectoryPositionDroneEnv(PositionDroneEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = deque()
        self.forces = deque()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        self.observations.clear()
        self.forces.clear()
        return super().reset(seed=seed, options=options)

    def calculate_propulsive_forces(self, action: VEC4) -> VEC4:
        forces = super().calculate_propulsive_forces(action)
        self.forces.append(forces)
        return forces

    @staticmethod
    def _calc_target_yaw(curpos, tgtpos):
        yaw = np.arctan2(tgtpos[1] - curpos[1], tgtpos[0] - curpos[0])
        return yaw

    @staticmethod
    def _calc_dist_to_target(curpos, tgtpos):
        return np.linalg.norm(tgtpos - curpos)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        x, y, z = action
        dist = self._calc_dist_to_target(self.state.pos, (x, y, z))
        self.observations.clear()

        t = time.time()
        for i in count():
            new_vel = self.model.max_vel_ms * clip_scalar(dist, 0.1, 0.4)
            obs, _, term, trunc, _ = super().step((x, y, z, new_vel))
            self.observations.append(obs)
            dist = self._calc_dist_to_target(self.state.pos, action)
            if dist <= 0.1:
                break
            if dist >= 20:
                term = True
            if term or trunc:
                break
        total_t = time.time() - t
        logger.debug(f"Completed {i} steps in {total_t:.2f}s ({i/total_t:.2f} it/s)")
        reward = self.trajectory_reward()
        self.forces.clear()
        info = self.get_info()
        return self.observations, reward, term, trunc, info

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array([(-10, 10), (-10, 10), (9, 10)])
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )

    def trajectory_reward(self):
        return np.asarray(self.forces).sum() / self.dt


def main():
    dt = 1 / 240
    model = DronePlus
    logger.debug(model)

    initial_state = State()
    initial_state.pos = [0, 0, 10]

    env = TrajectoryPositionDroneEnv(
        model=model,
        initial_state=initial_state,
        dt=dt,
    )
    env = TimeLimit(env, max_episode_steps=20)

    env.reset()

    setpoints = deque()
    rewards = deque()
    observations = deque()
    while True:
        setpoint = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(PositionAction(setpoint))
        observations.append(np.array(obs).copy())
        setpoints.append(setpoint)
        rewards.append(reward)

        if trunc or term:
            print(f"{trunc=} {term=}")
            break

    logger.info("Fin.")


if __name__ == "__main__":
    main()