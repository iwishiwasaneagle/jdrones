#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pathlib

import numpy as np
import pytest
from jdrones.envs.attitude import AttitudeAltitudeDroneEnv
from jdrones.envs.drone import DroneEnv
from jdrones.envs.trajectory import PIDTrajectoryDroneEnv
from jdrones.envs.velocityheading import VelHeadAltDroneEnv
from jdrones.transforms import euler_to_quat
from jdrones.types import SimulationType
from jdrones.types import State
from jdrones.types import URDFModel


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests and only run when "
        "--with-integration or --only-integration is passed",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--with-integration", action="store_true", help="Run integration tests"
    )
    parser.addoption(
        "--only-integration", action="store_true", help="Only run integration tests"
    )


def pytest_collection_modifyitems(config, items):
    w_int = config.getoption("--with-integration")
    o_int = config.getoption("--only-integration")
    if w_int and o_int:
        raise Exception("Cannot have both -with-integration and --only-integration")
    elif w_int and not o_int:
        # --with-integration given in cli: do not skip any tests
        return
    elif not w_int and not o_int:
        # --with-integration NOT given in cli: skip integration tests
        skip_integration = pytest.mark.skip(
            reason="Integration tests are being skipped"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    elif not w_int and o_int:
        # --only-integration given in cli: skip all non-integration tests
        skip_non_integration = pytest.mark.skip(
            reason="Non-integration tests are being skipped"
        )
        for item in items:
            if "integration" not in item.keywords:
                item.add_marker(skip_non_integration)
    else:
        raise Exception("Unknown condition")


@pytest.fixture(params=[1 / 240])
def dt(request):
    return request.param


@pytest.fixture(params=[8008])
def seed(request):
    return request.param


@pytest.fixture(params=[9.81])
def g(request):
    return request.param


@pytest.fixture(params=[1.4])
def mass(request):
    return request.param


@pytest.fixture(params=[0.1])
def length(request):
    return request.param


@pytest.fixture(params=[(0.1, 0.1, 0.1)])
def Inertia(request):
    return request.param


@pytest.fixture(params=[0.1])
def k_T(request):
    return request.param


@pytest.fixture(params=[0.1])
def k_Q(request):
    return request.param


@pytest.fixture(params=[0.1])
def tau_T(request):
    return request.param


@pytest.fixture(params=[0.1])
def tau_Q(request):
    return request.param


@pytest.fixture(params=[(0.1, 0.1, 0.1)])
def drag_coeffs(request):
    return request.param


@pytest.fixture(params=["droneplus.urdf"])
def filepath(request):
    root = pathlib.Path("tests")
    base = pathlib.Path("assets")
    if root.exists():
        return root / base / request.param
    return base / request.param


@pytest.fixture
def mixing_matrix():
    def droneplus_mixing_matrix(length, k_f, k_t):
        h = k_f
        i = k_t
        j = length * h

        return np.array(
            [
                [0, 1 / (2 * j), i / 4, 1 / (4 * h)],
                [1 / (2 * j), 0, -i / 4, 1 / (4 * h)],
                [0, -1 / (2 * j), i / 4, 1 / (4 * h)],
                [-1 / (2 * j), 0, -i / 4, 1 / (4 * h)],
            ]
        )

    return droneplus_mixing_matrix


@pytest.fixture(params=[(0, 0, 0.1)])
def pos(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def rpy(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def velocity(request):
    return request.param


@pytest.fixture
def state(pos, velocity, rpy):
    s = State()
    s.pos = pos
    s.vel = velocity
    s.rpy = rpy
    s.quat = euler_to_quat(rpy)
    return s


@pytest.fixture(params=[(0, 0, 0, 0)])
def vec_omega(request):
    return request.param


@pytest.fixture
def action(vec_omega):
    return np.array(vec_omega)


@pytest.fixture(params=[SimulationType.DIRECT])
def simulation_type(request):
    return request.param


@pytest.fixture
def urdfmodel(
    g,
    length,
    Inertia,
    k_T,
    k_Q,
    tau_T,
    tau_Q,
    drag_coeffs,
    mass,
    filepath,
    mixing_matrix,
):
    return URDFModel(
        g=g,
        l=length,
        I=Inertia,
        k_T=k_T,
        k_Q=k_Q,
        tau_T=tau_T,
        tau_Q=tau_Q,
        drag_coeffs=drag_coeffs,
        mass=mass,
        filepath=str(filepath),
        mixing_matrix=mixing_matrix,
        max_vel_ms=1,
    )


@pytest.fixture
def env_default_kwargs(urdfmodel, dt, state, simulation_type):
    return dict(
        model=urdfmodel, initial_state=state, dt=dt, simulation_type=simulation_type
    )


@pytest.fixture
def droneenv(env_default_kwargs):
    d = DroneEnv(**env_default_kwargs)
    yield d
    d.close()


@pytest.fixture
def attaltdroneenv(env_default_kwargs):
    a = AttitudeAltitudeDroneEnv(**env_default_kwargs)
    yield a
    a.close()


@pytest.fixture
def velheadaltdroneenv(env_default_kwargs):
    a = VelHeadAltDroneEnv(**env_default_kwargs)
    yield a
    a.close()


@pytest.fixture
def pidtrajposdroneenv(env_default_kwargs):
    a = PIDTrajectoryDroneEnv(**env_default_kwargs)
    yield a
    a.close()
