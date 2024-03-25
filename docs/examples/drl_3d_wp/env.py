#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
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
from jdrones.envs import BaseControlledEnv
from jdrones.envs import LQRDroneEnv
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.types import PropellerAction
from jdrones.wrappers import EnergyCalculationWrapper


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
        ]
    )

    def __init__(
        self,
        dt,
        env_cls: Type[BaseDroneEnv | BaseControlledEnv] = NonlinearDynamicModelDroneEnv,
    ):
        super().__init__()
        self.dt = dt

        self.env = EnergyCalculationWrapper(env_cls(dt=self.dt))

    def check_is_oob(self):
        pos = self.env.state.pos
        return bool(np.any((POS_LIM[0] > pos) | (pos > POS_LIM[1])))

    def check_is_unstable(self):
        r, p, _ = self.env.state.rpy
        _, _, dy = self.env.state.ang_vel

        roll_within_lims = (ROLL_LIM[0] < r) & (r < ROLL_LIM[1])
        pitch_within_lims = (PITCH_LIM[0] < p) & (p < PITCH_LIM[1])
        yaw_rate_within_lims = (ANG_VEL_LIM[0] < dy) & (dy < ANG_VEL_LIM[1])
        return bool(~(roll_within_lims & pitch_within_lims & yaw_rate_within_lims))

    def reset_target(self):
        self.target = np.random.uniform(
            *TGT_SUB_LIM,
            3,
        )
        self.info["target"] = self.target
        self.target_counter = 0


class DRL_WP_Env(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(State.k,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

    def get_observation(self):
        state = State()
        state[:20] = self.env.unwrapped.state
        state.target = self.target
        state.target_error = self.target_error
        state.target_error_integral = self.integral_target_error
        state.rpy[2] %= 2 * np.pi

        normed_state = state.normed(self.NORM_LIMITS)
        return normed_state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        _, info = self.env.reset()
        self.previous_prop_omega = 0
        self.target_error = self.integral_target_error = np.zeros_like(self.target)
        self.info = {"is_success": False, "targets": 0}
        self.reset_target()
        return self.get_observation(), info

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        trunc = False
        term = False
        denormed_action = np.interp(action, (-1, 1), PROP_OMEGA_LIM)
        obs, _, _, _, info = self.env.step(denormed_action)
        self.info["action"] = action

        prop_omega = obs.prop_omega
        control_action = np.linalg.norm(obs.prop_omega)
        dcontrol_action = np.linalg.norm(self.previous_prop_omega - prop_omega)
        self.previous_prop_omega = np.copy(prop_omega)
        self.info["control_action"] = control_action
        self.info["dcontrol_action"] = dcontrol_action

        self.target_error = self.target - obs.pos
        self.integral_target_error = (
            self.integral_target_error * 0.9 + self.target_error
        )

        distance_from_tgt = np.linalg.norm(self.target_error)
        self.info["distance_from_target"] = distance_from_tgt

        reward = (
            0  # alive bonus
            + -1 * distance_from_tgt
            + 0 * info["energy"]
            + 0 * control_action
            + 0 * dcontrol_action
            + 0 * np.linalg.norm(self.integral_target_error)
        )

        if distance_from_tgt < 1.5:
            reward += 50
            self.info["is_success"] = True
            self.info["targets"] += 1
            self.reset_target()

        is_oob = self.check_is_oob()
        self.info["is_oob"] = is_oob
        is_unstable = self.check_is_unstable()
        self.info["is_unstable"] = is_unstable
        if is_oob or is_unstable:
            self.info["is_success"] = False
            trunc = True
            reward -= 50
        c = 50 + np.sqrt(3 * 20 * 20)
        lower, upper = -c, c
        reward = ((reward - lower) / (upper - lower) - 0.5) * 2
        self.info["state"] = self.env.unwrapped.state

        return self.get_observation(), float(reward), term, trunc, self.info | info


class DRL_WP_Env_LQR(BaseEnv):

    def __init__(self, *args, T: float = 10, **kwargs):
        super().__init__(*args, env_cls=LQRDroneEnv, **kwargs)

        self.T = T

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )

    def get_observation(self):
        state = State()
        state[:20] = self.env.unwrapped.state
        state.target = self.target
        state.target_error = self.target_error
        state.rpy[2] %= 2 * np.pi

        normed_state = state.normed(self.NORM_LIMITS)
        return np.concatenate(
            [normed_state.pos, normed_state.vel, normed_state.target, [self.time]]
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
        self.target_error = 0
        self.time = 0
        self.info = {"is_success": False, "targets": 0}
        self.reset_target()
        return self.get_observation(), info

    def step(self, action) -> Tuple[State, float, bool, bool, dict]:
        trunc = False
        term = False
        position_action = action[:3] * POS_LIM[1]
        velocity_action = action[3:6] * VEL_LIM[1]
        min_action_t, max_action_t = 0.1, 1
        time_action = np.interp(action[6], (-1, 1), (min_action_t, max_action_t))
        x = State()
        x.pos = position_action
        x.vel = velocity_action

        net_control_action = 0
        net_dcontrol_action = 0
        net_energy = 0
        reward = 0

        states = []
        for _ in range(max(1, int(time_action / self.dt))):
            obs, _, _, _, info = self.env.step(x.to_x())
            self.time += self.dt

            net_energy += info["energy"]

            prop_omega = obs.prop_omega
            control_action = np.linalg.norm(obs.prop_omega)
            dcontrol_action = np.linalg.norm(self.previous_prop_omega - prop_omega)
            self.previous_prop_omega = np.copy(prop_omega)
            net_control_action += control_action
            net_dcontrol_action += dcontrol_action

            self.target_error = self.target - obs.pos
            distance_from_tgt = np.linalg.norm(self.target_error)
            self.info["distance_from_target"] = distance_from_tgt

            states.append(np.copy(self.env.state))

            reward += (
                0  # alive bonus
                + -(max_action_t / self.dt) * distance_from_tgt
                + 0 * net_energy
                + 0 * net_control_action
                + 0 * net_dcontrol_action
            )

            if distance_from_tgt < 0.5:
                reward += 50
                self.info["is_success"] = True
                self.info["targets"] += 1
                self.reset_target()
                break

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
                reward -= 50
                break

        self.info["action"] = action
        self.info["state"] = np.vstack(states)
        info["energy"] = net_energy
        self.info["control_action"] = net_control_action
        self.info["dcontrol_action"] = net_dcontrol_action

        c = 50 + max_action_t * np.sqrt(3 * 20 * 20)
        lower, upper = -c, c
        reward = ((reward - lower) / (upper - lower) - 0.5) * 2

        return self.get_observation(), float(reward), term, trunc, self.info | info
