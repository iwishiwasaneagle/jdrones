import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env


@pytest.mark.integration
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


@pytest.mark.integration
def test_DroneEnv(droneenv):
    check_env(droneenv)


@pytest.mark.integration
def test_AttitudeAltitudeDroneEnv(attaltdroneenv):
    check_env(attaltdroneenv)


@pytest.mark.integration
def test_VelHeadAltDroneEnv(velheadaltdroneenv):
    check_env(velheadaltdroneenv)


@pytest.mark.integration
def test_PositionDroneEnv(posdroneenv):
    check_env(posdroneenv)


@pytest.mark.integration
def test_TrajectoryPositionDroneEnv(trajposdroneenv):
    check_env(trajposdroneenv)
