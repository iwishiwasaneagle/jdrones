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
def test_LinearDynamicsDroneEnv(lineardroneenv):
    check_env(lineardroneenv)


@pytest.mark.integration
def test_NonLinearDynamicsDroneEnv(nonlineardroneenv):
    check_env(nonlineardroneenv)


@pytest.mark.integration
def test_LQRDroneEnv(lqrdroneenv):
    check_env(lqrdroneenv)


@pytest.mark.integration
def test_FirstOrderPolyPositionDroneEnv(firstorderploypositiondroneenv):
    check_env(firstorderploypositiondroneenv)


@pytest.mark.integration
def test_FifthOrderPolyPositionDroneEnv(fifthorderpolypositiondroneenv):
    check_env(fifthorderpolypositiondroneenv)


@pytest.mark.integration
def test_OptimalFifthOrderPolyPositionDroneEnv(optimalfifthorderpolypositiondroneenv):
    check_env(optimalfifthorderpolypositiondroneenv)
