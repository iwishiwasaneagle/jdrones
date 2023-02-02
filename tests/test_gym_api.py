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
        ("PIDTrajectoryDroneEnv-v0", {}),
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
def test_PIDTrajectoryDroneEnv(pidtrajposdroneenv):
    check_env(pidtrajposdroneenv)
