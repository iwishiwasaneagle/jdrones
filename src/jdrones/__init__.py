#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import warnings

from gymnasium.envs.registration import register

warnings.warn(
    """
Both NL and L are currently using a LHR coordinate system,
whilst the PB3 model is using a RHR system. This WILL cause
issues in the future and MUST changed at some point.
"""
)

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
    "PositionDroneEnv-v0",
    entry_point="jdrones.envs:PositionDroneEnv",
)

__version__ = "unknown"
