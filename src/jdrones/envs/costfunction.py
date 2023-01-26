from typing import Callable, Tuple

from gymnasium.core import ObsType

from jdrones.envs.trajectory import TrajectoryPositionDroneEnv
from jdrones.types import PositionAction


class CustomCostFunctionTrajectoryDroneEnv(TrajectoryPositionDroneEnv):
    def __init__(
        self,
        cost_func: Callable[["CustomCostFunctionTrajectoryDroneEnv"], float],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cost_func = cost_func

    def step(self, action: PositionAction) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, _, term, trunc, info = super().step(action)
        return obs, self.cost_func(self), term, trunc, info
