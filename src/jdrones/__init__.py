#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from gymnasium.envs.registration import register


register("QuadPyBulletDroneEnv-v0", entry_point="jdrones.envs:QuadPyBulletDroneEnv")
register(
    "QuadNonLinearDynamicModelDroneEnv-v0",
    entry_point="jdrones.envs:QuadNonlinearDynamicModelDroneEnv",
)
register(
    "QuadLinearDynamicModelDroneEnv-v0",
    entry_point="jdrones.envs:QuadLinearDynamicModelDroneEnv",
)
register(
    "XWingNonlinearDynamicModelDroneEnv-v0",
    entry_point="jdrones.envs:XWingNonlinearDynamicModelDroneEnv",
)
register(
    "LQRDroneEnv-v0",
    entry_point="jdrones.envs:LQRDroneEnv",
)
register(
    "LQRPositionDroneEnv-v0",
    entry_point="jdrones.envs:LQRPositionDroneEnv",
)
register(
    "PolyPositionDroneEnv-v0",
    entry_point="jdrones.envs:PolyPositionDroneEnv",
)

__version__ = "unknown"
