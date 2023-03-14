#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from gymnasium.envs.registration import register

register(
    "PB3NonLinearDrone-v0",
    entry_point="jdrones.envs:PyBulletDroneEnv",
)
register(
    "NonLinearDrone-v0",
    entry_point="jdrones.envs:NonlinearDynamicModelDroneEnv",
)
register(
    "LinearDrone-v0",
    entry_point="jdrones.envs:LinearDynamicModelDroneEnv",
)
register(
    "LQRDrone-v0",
    entry_point="jdrones.envs:LQRDroneEnv",
)
register(
    "LQRPositionDrone-v0",
    entry_point="jdrones.envs:LQRPositionDroneEnv",
)
register(
    "LQRPolyPositionDrone-v0",
    entry_point="jdrones.envs:LQRPolyPositionDroneEnv",
)

__version__ = "unknown"
