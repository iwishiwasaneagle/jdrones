#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env


@pytest.mark.integration
@pytest.mark.parametrize(
    "env,kwargs",
    [
        ("PyBulletDroneEnv-v0", {}),
        ("NonLinearDynamicModelDroneEnv-v0", {}),
        ("LinearDynamicModelDroneEnv-v0", {}),
        ("LQRDroneEnv-v0", {}),
        ("PositionDroneEnv-v0", {}),
    ],
)
def test_make(env, kwargs):
    assert isinstance(gymnasium.make(env, **kwargs), gymnasium.Env)


@pytest.mark.integration
def test_PB3DroneEnv(pbdroneenv):
    check_env(pbdroneenv)


@pytest.mark.integration
def test_LinearDynamicsDroneEnv(lineardroneenv):
    check_env(lineardroneenv)


@pytest.mark.integration
def test_NonLinearDynamicsDroneEnv(nonlineardroneenv):
    check_env(nonlineardroneenv)


@pytest.mark.integration
def test_LQRDroneEnv(lqrdroneenv):
    check_env(lqrdroneenv)


@pytest.mark.integration
def test_PositionDroneEnv(positiondroneenv):
    check_env(positiondroneenv)
