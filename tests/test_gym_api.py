#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env


@pytest.mark.integration
@pytest.mark.parametrize(
    "env",
    filter(
        lambda item: "jdrones" in str(item.entry_point),
        tuple(gymnasium.registry.values()),
    ),
)
def test_make(env):
    assert isinstance(gymnasium.make(env), gymnasium.Env)


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
def test_LQRPositionDroneEnv(lqrpositiondroneenv):
    check_env(lqrpositiondroneenv)


@pytest.mark.integration
def test_PolyPositionDroneEnv(polypositiondroneenv):
    check_env(polypositiondroneenv)
