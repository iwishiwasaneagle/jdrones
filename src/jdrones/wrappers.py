from typing import Any
from typing import SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import WrapperActType
from gymnasium.core import WrapperObsType
from jdrones.energy_model import BaseEnergyModel
from jdrones.energy_model import StaticPropellerVariableVelocityEnergyModel
from jdrones.envs.base.basedronenev import BaseDroneEnv


class EnergyCalculationWrapper(gymnasium.Wrapper):
    """
    Wrap a drone env to get energy consumption calculations in the returned `info`.
    """

    def __init__(
        self,
        env: BaseDroneEnv,
        energy_model: type[
            BaseEnergyModel
        ] = StaticPropellerVariableVelocityEnergyModel,
    ):
        super().__init__(env)
        if hasattr(self.env, "model"):
            model = self.env.model
        elif hasattr(self.env, "env"):
            model = self.env.env.model
        else:
            raise ValueError(
                "Could not find model information within the wrapped " "environment"
            )
        self.energy_calculation = energy_model(env.dt, model)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, term, trunc, info = super().step(action)

        vel = self.env.unwrapped.state.vel
        speed = np.linalg.norm(vel)
        info["energy"] = self.energy_calculation.energy(speed)

        return obs, reward, term, trunc, info
