from typing import Any
from typing import SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import WrapperActType
from gymnasium.core import WrapperObsType
from jdrones.energy_model import BaseEnergyModel
from jdrones.energy_model import StaticPropellerVariableVelocityEnergyModel
from jdrones.envs import BaseControlledEnv
from jdrones.envs.base.basedronenev import BaseDroneEnv


class EnergyCalculationWrapper(gymnasium.Wrapper):
    """
    Wrap a drone env to get energy consumption calculations in the returned `info`.
    """

    def __init__(
        self,
        env: BaseDroneEnv | BaseControlledEnv,
        energy_model: type[
            BaseEnergyModel
        ] = StaticPropellerVariableVelocityEnergyModel,
    ):
        super().__init__(env)

        if not isinstance(self.unwrapped, (BaseDroneEnv, BaseControlledEnv)):
            raise ValueError(f"Type {type(self.unwrapped)} is not allowed.")

        if isinstance(self.unwrapped, BaseControlledEnv):
            model = self.unwrapped.env.model
        else:
            model = self.unwrapped.model

        self.energy_calculation = energy_model(env.dt, model)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, term, trunc, info = super().step(action)

        if isinstance(self.unwrapped, BaseControlledEnv):
            vel = self.unwrapped.env.state.vel
        else:
            vel = self.unwrapped.state.vel

        speed = np.linalg.norm(vel)
        info["energy"] = self.energy_calculation.energy(speed)

        return obs, reward, term, trunc, info
