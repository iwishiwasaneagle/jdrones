#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import functools
from typing import Optional
from typing import Tuple
from typing import Type

import gymnasium
import numpy as np
from drl_3d_wp.consts import ANG_VEL_LIM
from drl_3d_wp.consts import PITCH_LIM
from drl_3d_wp.consts import POS_LIM
from drl_3d_wp.consts import PROP_OMEGA_LIM
from drl_3d_wp.consts import ROLL_LIM
from drl_3d_wp.consts import TGT_SUB_LIM
from drl_3d_wp.consts import VEL_LIM
from drl_3d_wp.consts import YAW_LIM
from drl_3d_wp.state import State
from jdrones.data_models import State as JState
from jdrones.envs import BaseControlledEnv
from jdrones.envs import LQRDroneEnv
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.wrappers import EnergyCalculationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


class BaseEnv(gymnasium.Env):
    NORM_LIMITS = np.array(
        [
            POS_LIM,
            POS_LIM,
            POS_LIM,
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            ROLL_LIM,
            PITCH_LIM,
            YAW_LIM,
            VEL_LIM,
            VEL_LIM,
            VEL_LIM,
            ANG_VEL_LIM,
            ANG_VEL_LIM,
            ANG_VEL_LIM,
            PROP_OMEGA_LIM,
            PROP_OMEGA_LIM,
            PROP_OMEGA_LIM,
            PROP_OMEGA_LIM,
            POS_LIM,
            POS_LIM,
            POS_LIM,
            (-20, 20),
            (-20, 20),
            (-20, 20),
            (-100, 100),
            (-100, 100),
            (-100, 100),
            POS_LIM,
            POS_LIM,
            POS_LIM,
        ]
    )

    def __init__(
        self,
        dt,
        env_cls: Type[BaseDroneEnv | BaseControlledEnv] = NonlinearDynamicModelDroneEnv,
        env_cls_kwargs: dict = None,
    ):
        super().__init__()
        self.dt = dt
        if env_cls_kwargs is None:
            env_cls_kwargs = {}
        self.env = EnergyCalculationWrapper(env_cls(dt=self.dt, **env_cls_kwargs))

        self.target = self.next_target = None

    @property
    def state(self):
        return self.env.state

    def check_is_oob(self):
        pos = self.env.unwrapped.state.pos
        return bool(np.any((POS_LIM[0] > pos) | (pos > POS_LIM[1])))

    def check_is_unstable(self):
        r, p, _ = self.env.unwrapped.state.rpy
        _, _, dy = self.env.unwrapped.state.ang_vel

        roll_within_lims = (ROLL_LIM[0] < r) & (r < ROLL_LIM[1])
        pitch_within_lims = (PITCH_LIM[0] < p) & (p < PITCH_LIM[1])
        yaw_rate_within_lims = (ANG_VEL_LIM[0] < dy) & (dy < ANG_VEL_LIM[1])
        return bool(~(roll_within_lims & pitch_within_lims & yaw_rate_within_lims))

    def create_target(self):
        return np.array(
            [
                *np.random.uniform(
                    *TGT_SUB_LIM,
                    2,
                ),
                self.env.unwrapped.initial_state.pos[2],
            ]
        )

    def reset_target(self):
        if self.next_target is None:
            self.target = self.create_target()
        else:
            self.target = self.next_target
        self.next_target = self.create_target()
        self.info["target"] = self.target
        self.target_counter = 0
        self.target_error = self.integral_target_error = np.zeros_like(self.target)


class DRL_WP_Env_LQR(BaseEnv):
    Q = np.diag(
        [
            4.6176861297742446e-08,
            8.20079210554526e-09,
            0.002015319613213222,
            0.0019725098771573558,
            0.0019729667280019856,
            0.0017448932320868343,
            0.00038948848401190175,
            6.60213968465527e-08,
            0.0013440112213180333,
            0.0002558222985053156,
            7.213974171712665e-08,
            2.2824520808018255e-07,
        ]
    )

    R = np.diag(
        [
            0.004613030294333662,
            0.005277238690122938,
            0.003479685903363842,
            0.0004249097075205608,
        ]
    )

    def __init__(
        self, *args, T: float = 10, c: float = 50, sim_T: float = 0.1, **kwargs
    ):
        env_cls_kwargs = kwargs.pop("env_cls_kwargs", {})
        super().__init__(
            *args,
            env_cls=LQRDroneEnv,
            env_cls_kwargs=dict(Q=self.Q, R=self.R) | env_cls_kwargs,
            **kwargs,
        )

        self.T = T
        self.sim_T = sim_T
        self.c = c
        self.K = self.dt / self.sim_T

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(22,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def get_observation(self):
        state = State()
        state[:20] = self.env.unwrapped.state
        state.target = self.target
        state.next_target = self.next_target
        state.target_error = self.target_error
        state.rpy[2] %= 2 * np.pi

        normed_state: State = state.normed(self.NORM_LIMITS)
        return np.concatenate(
            [
                normed_state.pos,  # x y zte
                normed_state.vel,  # vx vy vz
                normed_state.target,  # tx ty tz
                normed_state.next_target,  # tx ty tz
                normed_state.rpy,  # roll pitch yaw
                normed_state.ang_vel,  # p q r
                normed_state.prop_omega,  # p1 p2 p3 p4,
            ]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        _, info = self.env.reset()
        self.previous_prop_omega = 0
        self.time = 0
        self.info = {"is_success": False, "targets": 0}
        self.reset_target()
        return self.get_observation(), info

    @staticmethod
    def calc_omni_velocity(a, b, max_velocity):
        """
        This function takes policy outputs (a, b) and maximum velocity
        and returns linear velocities (vx, vy) for omni-directional movement.

        Args:
            a: Scaled output from the policy (-1 to 1).
            b: Scaled output from the policy (-1 to 1).
            max_velocity: Maximum linear velocity of the agent.

        Returns:
            vx: Linear velocity in x-direction.
            vy: Linear velocity in y-direction.
        """
        magnitude = np.sqrt(a**2 + b**2)
        if magnitude <= 1e-6:
            return 0.0, 0.0

        unit_a = a / magnitude
        unit_b = b / magnitude

        magnitude_clipped = np.core.umath.minimum(magnitude, 1)

        vx = magnitude_clipped * max_velocity * unit_a
        vy = magnitude_clipped * max_velocity * unit_b

        return vx, vy

    def step(self, action) -> Tuple[State, float, bool, bool, dict]:
        trunc = False
        term = False

        x = State()
        x.pos = [0, 0, self.target[2]]
        x.vel = [*self.calc_omni_velocity(*action, 1.0), 0]

        net_control_action = 0
        net_dcontrol_action = 0
        net_energy = 0
        reward = 0

        states = []
        for _ in range(int(1 / self.K)):
            obs, _, _, _, info = self.env.step(x.to_x())
            self.time += self.dt

            energy = info["energy"]
            net_energy += energy

            prop_omega = obs.prop_omega
            control_action = np.linalg.norm(obs.prop_omega)
            dcontrol_action = np.linalg.norm(self.previous_prop_omega - prop_omega)
            self.previous_prop_omega = np.copy(prop_omega)
            net_control_action += control_action
            net_dcontrol_action += dcontrol_action

            self.target_error = self.target - obs.pos
            distance_from_tgt = np.linalg.norm(self.target_error)
            self.info["distance_from_target"] = distance_from_tgt

            states.append(np.copy(self.env.unwrapped.state))

            reward += self.K * (
                -1 * distance_from_tgt
                # + 0.000001 * energy
                - 0.00001 * control_action
                # + 0 * dcontrol_action
            )

            if distance_from_tgt < 1:
                reward = self.c
                self.info["is_success"] = True
                self.info["targets"] += 1
                self.reset_target()

            if self.time > self.T:
                trunc = True
                self.info["TimeLimit.truncated"] = True
                break

            is_oob = self.check_is_oob()
            self.info["is_oob"] = is_oob
            is_unstable = self.check_is_unstable()
            self.info["is_unstable"] = is_unstable
            if is_oob or is_unstable:
                self.info["is_success"] = False
                trunc = True
                reward = -self.c
                break

        self.info["action"] = action
        self.info["state"] = np.vstack(states)
        info["energy"] = net_energy
        self.info["control_action"] = net_control_action
        self.info["dcontrol_action"] = net_dcontrol_action

        return self.get_observation(), float(reward), term, trunc, self.info | info


class Multi_DRL_WP_Env_LQR(gymnasium.Env):

    @staticmethod
    def make_sub_env(
        *, angle: float, T: float, sim_T: float = 0.1, dt: float, c: float
    ) -> gymnasium.Env:
        state = JState()
        state.pos = [2.5 * np.cos(angle), 2.5 * np.sin(angle), 0]
        return DRL_WP_Env_LQR(
            T=T, dt=dt, c=c, sim_T=sim_T, env_cls_kwargs=dict(initial_state=state)
        )

    def __init__(self, *, T, dt, N_envs: int = 2, c: float = 50, sim_T: float = 0.1):
        super().__init__()

        self.T = T
        self.sim_T = sim_T
        self.dt = dt
        self.c = c

        self.envs = DummyVecEnv(
            [
                functools.partial(
                    self.make_sub_env,
                    angle=f,
                    T=self.T,
                    dt=self.dt,
                    c=self.c,
                    sim_T=self.sim_T,
                )
                for f in np.linspace(0, 2 * np.pi, N_envs + 1)[1:]
            ]
        )

        obs = self.envs.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=np.hstack([[[0]], np.tile(obs.low, (1, N_envs))]),
            high=np.hstack([[[self.T]], np.tile(obs.high, (1, N_envs))]),
            shape=(1, 1 + N_envs * obs.shape[0]),
        )
        act = self.envs.action_space
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(act.low, (N_envs, 1)),
            high=np.tile(act.high, (N_envs, 1)),
            shape=(N_envs, *act.shape),
        )

    @staticmethod
    def merge_infos(*infos):
        merged_info = {f"env_{i}": info for i, info in enumerate(infos)}

        success = all(f.get("is_success") for f in merged_info.values())
        oob = any(f.get("is_oob") for f in merged_info.values())
        unstable = any(f.get("is_unstable") for f in merged_info.values())
        timelimit = any(f.get("TimeLimit.truncated") for f in merged_info.values())

        merged_info["is_oob"] = oob
        merged_info["is_success"] = success
        merged_info["is_unstable"] = unstable
        merged_info["TimeLimit.truncated"] = timelimit

        return merged_info

    @staticmethod
    def merge_observations(*observations):
        return np.concatenate(observations)

    def get_observation(self, *observations):
        return np.concatenate([[self.time], self.merge_observations(*observations)])[np.newaxis, :]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        obs = self.envs.reset()
        self.time = 0

        return self.get_observation(*obs), {}

    def step(self, action):
        obs, rew, dones, info = self.envs.step(action)
        self.time += self.sim_T / self.dt

        obs = self.get_observation(*obs)
        done = np.any(dones)

        if self.envs.num_envs > 1:
            pos = np.array([f.pos for f in self.envs.get_attr("state")])
            dists = np.linalg.norm(
                pos[np.newaxis, :, :] - pos[:, np.newaxis, :], axis=2
            )[~np.eye(pos.shape[0], dtype=bool)]

            min_distance_between_envs = dists.min()

            collision = min_distance_between_envs < 0.5
            if collision:
                rew = rew * 0 - self.c
                done = True
        else:
            collision = False

        for i in range(len(info)):
            info[i]["collision"] = collision

        lower, upper = -self.c, self.c
        reward = ((rew - lower) / (upper - lower) - 0.5) * 2
        reward = np.sum(reward)

        info = self.merge_infos(*info)
        return obs, reward, done, done, info
