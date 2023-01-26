import pytest
from gymnasium.utils.env_checker import check_env
import gymnasium
import jdrones

@pytest.mark.integration_test
@pytest.mark.parametrize(
    "env,kwargs",
    [
        ("DroneEnv-v0", {}),
        ("AttitudeAltitudeDroneEnv-v0", {}),
        ("VelHeadAltDroneEnv-v0", {}),
        ("PositionDroneEnv-v0", {}),
        ("TrajectoryPositionDroneEnv-v0", {}),
        ("CustomCostFunctionTrajectoryDroneEnv-v0", {"cost_func": lambda s: 0}),
    ],
)
def test_make(env, kwargs):
    assert isinstance(gymnasium.make(env, **kwargs), gymnasium.Env)


@pytest.mark.integration_test
def test_DroneEnv(droneenv):
    check_env(droneenv)


@pytest.mark.integration_test
def test_AttitudeAltitudeDroneEnv(attaltdroneenv):
    check_env(attaltdroneenv)


@pytest.mark.integration_test
def test_VelHeadAltDroneEnv(velheadaltdroneenv):
    check_env(velheadaltdroneenv)


@pytest.mark.integration_test
def test_PositionDroneEnv(posdroneenv):
    check_env(posdroneenv)


@pytest.mark.slow_integration_test
def test_TrajectoryPositionDroneEnv(trajposdroneenv):
    check_env(trajposdroneenv)
