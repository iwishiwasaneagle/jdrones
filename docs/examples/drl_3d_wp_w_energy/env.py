#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Optional
from typing import Tuple

import gymnasium
import numpy as np
from drl_3d_wp_w_energy.consts import ANG_VEL_LIM
from drl_3d_wp_w_energy.consts import PITCH_LIM
from drl_3d_wp_w_energy.consts import POS_LIM
from drl_3d_wp_w_energy.consts import PROP_OMEGA_LIM
from drl_3d_wp_w_energy.consts import ROLL_LIM
from drl_3d_wp_w_energy.consts import TGT_SUB_LIM
from drl_3d_wp_w_energy.consts import VEL_LIM
from drl_3d_wp_w_energy.consts import YAW_LIM
from drl_3d_wp_w_energy.state import State
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.types import PropellerAction
from jdrones.wrappers import EnergyCalculationWrapper


class HoverEnv(gymnasium.Env):
    LIMITS = np.array(
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

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

        self.env = EnergyCalculationWrapper(NonlinearDynamicModelDroneEnv(dt=self.dt))

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(State.k,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.reset_target()

    def get_observation(self):
        state = State()
        state[:20] = self.env.unwrapped.state
        state.target = self.target
        state.target_error = self.target_error
        state.target_error_integral = self.integral_target_error
        state.rpy[2] %= 2 * np.pi

        normed_state = state.normed(self.LIMITS)
        return normed_state

    def reset_target(self):
        self.target = np.random.uniform(
            *TGT_SUB_LIM,
            3,
        )
        self.target_counter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        _, info = self.env.reset()
        self.reset_target()
        self.previous_prop_omega = 0
        self.target_error = self.integral_target_error = np.zeros_like(self.target)
        self.info = {"is_success": False, "targets": 0}
        return self.get_observation(), info

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

        return self.get_observation(), float(reward), term, trunc, self.info | info
