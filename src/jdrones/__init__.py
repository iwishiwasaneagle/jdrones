#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from gymnasium.envs.registration import register


register("PyBulletDroneEnv-v0", entry_point="jdrones.envs:PyBulletDroneEnv")
register(
    "NonLinearDynamicModelDroneEnv-v0",
    entry_point="jdrones.envs:NonlinearDynamicModelDroneEnv",
)
register(
    "LinearDynamicModelDroneEnv-v0",
    entry_point="jdrones.envs:LinearDynamicModelDroneEnv",
)
register(
    "LQRDroneEnv-v0",
    entry_point="jdrones.envs:LQRDroneEnv",
)
register(
    "FirstOrderPolyPositionDroneEnv-v0",
    entry_point="jdrones.envs:FirstOrderPolyPositionDroneEnv",
)
register(
    "FifthOrderPolyPositionDroneEnv-v0",
    entry_point="jdrones.envs:FifthOrderPolyPositionDroneEnv",
)
register(
    "FifthOrderPolyPositionWithLookAheadDroneEnv-v0",
    entry_point="jdrones.envs:FifthOrderPolyPositionWithLookAheadDroneEnv",
)


__version__ = "unknown"
