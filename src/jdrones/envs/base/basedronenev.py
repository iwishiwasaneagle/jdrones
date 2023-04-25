#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from copy import copy
from typing import Any
from typing import Optional
from typing import Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs.dronemodels import DronePlus
from jdrones.transforms import euler_to_quat
from jdrones.types import DType


class BaseDroneEnv(gymnasium.Env):
    state: State
    """Current drone state"""

    initial_state: State
    """Initial drone state. Used for resettign the simulation"""

    model: URDFModel
    """Model parameters"""

    info: dict[str, Any]
    """Information dictionary to return"""

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
    ):
        if initial_state is None:
            initial_state = State()
        self.initial_state = initial_state
        self.state = copy(self.initial_state)
        self.model = model
        self.dt = dt
        self.info = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        self.info = {}
        self.state = copy(self.initial_state)
        self.state.quat = euler_to_quat(self.state.rpy)
        return self.state, self.info

    @property
    def action_space(self):
        act_bounds = np.array(
            [
                (0.0, 1e6),  # R1
                (0.0, 1e6),  # R2
                (0.0, 1e6),  # R3
                (0.0, 1e6),  # R4
            ],
            dtype=DType,
        )
        return spaces.Box(low=act_bounds[:, 0], high=act_bounds[:, 1], dtype=DType)

    @property
    def observation_space(self):
        obs_bounds = np.array(
            [
                # XYZ
                # Position
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # Q 1-4
                # Quarternion rotation
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                # RPY
                # Roll pitch yaw rotation
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                # V XYZ
                # Velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # V RPY
                # Angular velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # P 0-4
                # Propeller speed
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
            ],
            dtype=DType,
        )
        return spaces.Box(low=obs_bounds[:, 0], high=obs_bounds[:, 1], dtype=DType)
