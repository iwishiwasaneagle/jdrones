#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import abc
from typing import Dict
from typing import Optional
from typing import Tuple

import gymnasium
from gymnasium.core import ActType
from jdrones.controllers import Controller
from jdrones.controllers import PID
from jdrones.data_models import State
from jdrones.envs.base.basedronenev import BaseDroneEnv


class BaseControlledEnv(gymnasium.Env, abc.ABC):
    env: BaseDroneEnv

    controllers: Dict[str, PID]

    def __init__(self, env: BaseDroneEnv, dt: float = 1 / 240):
        self.env = env
        self.dt = dt
        self.controllers = self._init_controllers()

        self.observation_space = self.env.observation_space

    @property
    @abc.abstractmethod
    def action_space(self) -> ActType:
        """
        Returns the action space required by gymnasium

        Returns
        -------
        gymnasium.core.ActType
            Action type describing the action space
        """
        pass

    @property
    def state(self):
        return self.env.state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        for ctrl in self.controllers.values():
            ctrl.reset()
        return self.env.reset()

    @abc.abstractmethod
    def _init_controllers(self) -> dict[str, Controller]:
        pass
