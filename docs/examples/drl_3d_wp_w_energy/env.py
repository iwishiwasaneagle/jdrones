#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Optional
from typing import Tuple

import gymnasium
import numpy as np
from jdrones.data_models import State as _State
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.types import PropellerAction
from jdrones.wrappers import EnergyCalculationWrapper

from .consts import ANG_VEL_LIM
from .consts import POS_LIM
from .consts import PROP_OMEGA_LIM
from .consts import RPY_LIM
from .consts import TGT_SUB_LIM
from .consts import VEL_LIM
from .state import State


class HoverEnv(gymnasium.Env):
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

        normed_state = state.normed(
            [
                POS_LIM,
                POS_LIM,
                POS_LIM,
                (-1, 1),
                (-1, 1),
                (-1, 1),
                (-1, 1),
                RPY_LIM,
                RPY_LIM,
                RPY_LIM,
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

        reset_state = _State()
        reset_state.pos = np.random.uniform(*TGT_SUB_LIM, 3)
        reset_state.vel = np.random.uniform(-1, 1, 3)
        reset_state.rpy = np.random.uniform(-0.2, 0.2, 3)
        reset_state.ang_vel = np.random.uniform(-0.1, 0.1, 3)

        _, info = self.env.reset(options=dict(reset_state=reset_state))
        self.reset_target()
        self.previous_prop_omega = 0
        self.target_error = self.integral_target_error = np.zeros_like(self.target)
        self.info = {"is_success": False, "targets": 0}
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
            4  # alive bonus
            + -5e-1 * distance_from_tgt
            + 0 * info["energy"]
            + 0 * control_action
            + 0 * dcontrol_action
            + 0 * np.linalg.norm(self.integral_target_error)
        )
        if distance_from_tgt < 1.5:
            self.target_counter += 1
            reward += 1
            if self.target_counter > int(1 / self.env.dt):
                self.info["is_success"] = True
                self.info["targets"] += 1
                self.reset_target()
        elif np.any((POS_LIM[0] > obs[:3]) | (POS_LIM[1] < obs[:3])):
            self.info["is_success"] = False
            trunc = True
            reward -= 10

        return self.get_observation(), float(reward), term, trunc, self.info | info
