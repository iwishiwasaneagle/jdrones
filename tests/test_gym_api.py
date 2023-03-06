#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env


@pytest.mark.integration
@pytest.mark.parametrize(
    "env,kwargs",
    [
        ("QuadPyBulletDroneEnv-v0", {}),
        ("QuadNonLinearDynamicModelDroneEnv-v0", {}),
        ("QuadLinearDynamicModelDroneEnv-v0", {}),
        ("XWingNonlinearDynamicModelDroneEnv-v0", {}),
        ("LQRDroneEnv-v0", {}),
        ("LQRPositionDroneEnv-v0", {}),
        ("PolyPositionDroneEnv-v0", {}),
    ],
)
def test_make(env, kwargs):
    assert isinstance(gymnasium.make(env, **kwargs), gymnasium.Env)


@pytest.mark.integration
def test_QuadPB3DroneEnv(quadpbdroneenv):
    check_env(quadpbdroneenv)


@pytest.mark.integration
def test_QuadLinearDynamicsDroneEnv(quadlineardroneenv):
    check_env(quadlineardroneenv)


@pytest.mark.integration
def test_QuadNonLinearDynamicsDroneEnv(quadnonlineardroneenv):
    check_env(quadnonlineardroneenv)


@pytest.mark.integration
def test_XWingNonLinearDynamicsDroneEnv(xwingnonlineardroneenv):
    check_env(xwingnonlineardroneenv)


@pytest.mark.integration
def test_LQRDroneEnv(lqrdroneenv):
    check_env(lqrdroneenv)


@pytest.mark.integration
def test_LQRPositionDroneEnv(lqrpositiondroneenv):
    check_env(lqrpositiondroneenv)


@pytest.mark.integration
def test_PolyPositionDroneEnv(polypositiondroneenv):
    check_env(polypositiondroneenv)
