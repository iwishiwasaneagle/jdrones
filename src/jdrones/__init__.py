#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from gymnasium.envs.registration import register

from ._version import __version__
from ._version import __version_tuple__


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
    "OptimalFifthOrderPolyPositionDroneEnv-v0",
    entry_point="jdrones.envs:OptimalFifthOrderPolyPositionDroneEnv",
)
register(
    "FifthOrderPolyPositionWithLookAheadDroneEnv-v0",
    entry_point="jdrones.envs:FifthOrderPolyPositionWithLookAheadDroneEnv",
)
register(
    "OptimalFifthOrderPolyPositionWithLookAheadDroneEnv-v0",
    entry_point="jdrones.envs:OptimalFifthOrderPolyPositionWithLookAheadDroneEnv",
)
