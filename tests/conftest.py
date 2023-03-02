#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pathlib

import numpy as np
import pytest
from gymnasium import spaces
from jdrones.data_models import SimulationType
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs import LinearDynamicModelDroneEnv
from jdrones.envs import LQRDroneEnv
from jdrones.envs import LQRPositionDroneEnv
from jdrones.envs import NonlinearDynamicModelDroneEnv
from jdrones.envs import PolyPositionDroneEnv
from jdrones.envs import PyBulletDroneEnv
from jdrones.envs.dronemodels import droneplus_mixing_matrix
from jdrones.envs.position import BasePositionDroneEnv
from jdrones.transforms import euler_to_quat


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests and only run when "
        "--with-integration or --only-integration is passed",
    )
    config.addinivalue_line(
        "markers",
        "stress: mark tests as stress tests and only run when "
        "--with-stress-tests or --only-stress-tests is passed",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--with-integration", action="store_true", help="Run integration tests"
    )
    parser.addoption(
        "--only-integration", action="store_true", help="Only run integration tests"
    )
    parser.addoption(
        "--with-stress-tests", action="store_true", help="Run stress tests"
    )
    parser.addoption(
        "--only-stress-tests", action="store_true", help="Only run stress tests"
    )


def pytest_collection_modifyitems(config, items):
    w_int = config.getoption("--with-integration")
    o_int = config.getoption("--only-integration")
    w_stress = config.getoption("--with-stress-tests")
    o_stress = config.getoption("--only-stress-tests")

    skip_integration = pytest.mark.skip(reason="Integration tests are being skipped")
    skip_non_integration = pytest.mark.skip(
        reason="Non-integration tests are being skipped"
    )
    skip_stress = pytest.mark.skip(reason="Stress tests are being skipped")
    skip_non_stress = pytest.mark.skip(reason="Non-stress tests are being skipped")

    if w_int and w_stress:
        return

    if o_stress and o_int:
        raise Exception("Cannot have both --only-integration and --only-stress-tests")

    for item in items:
        int_in_kw = "integration" in item.keywords
        stress_in_kw = "stress" in item.keywords
        apply = []

        if o_int:
            if not int_in_kw:
                apply.append(skip_non_integration)
        elif o_stress:
            if not stress_in_kw:
                apply.append(skip_non_stress)
        elif w_int:
            if stress_in_kw:
                apply.append(skip_stress)
        elif w_stress:
            if int_in_kw:
                apply.append(skip_integration)
        else:
            if int_in_kw:
                apply.append(skip_integration)
            if stress_in_kw:
                apply.append(skip_stress)

        for marker in apply:
            item.add_marker(marker)


@pytest.fixture
def np_ndarray_factory():
    def fn(x):
        return np.asarray(x)

    return fn


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
    root = pathlib.Path("src")
    base = pathlib.Path("jdrones/envs")
    if root.exists():
        return root / base / request.param
    return base / request.param


@pytest.fixture
def mixing_matrix():
    return droneplus_mixing_matrix


@pytest.fixture(params=[(0, 0, 1)])
def pos(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def rpy(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def velocity(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def angular_velocity(request):
    return request.param


@pytest.fixture
def state(pos, velocity, angular_velocity, rpy):
    s = State()
    s.pos = pos
    s.vel = velocity
    s.rpy = rpy
    s.ang_vel = angular_velocity
    s.quat = euler_to_quat(rpy)
    return s


@pytest.fixture(params=[(1, 1, 1, 1)])
def vec_omega(request, equilibrium_prop_rpm, np_ndarray_factory):
    return np_ndarray_factory(request.param) * equilibrium_prop_rpm


@pytest.fixture
def action(vec_omega, np_ndarray_factory):
    return np_ndarray_factory(vec_omega)


@pytest.fixture(params=[SimulationType.DIRECT])
def simulation_type(request):
    return request.param


@pytest.fixture(params=[None])
def equilibrium_prop_rpm(request, k_T, mass, g):
    if request.param is None:
        return np.sqrt(mass * g / (4 * k_T))
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
def env_default_kwargs(urdfmodel, dt, state):
    return dict(model=urdfmodel, initial_state=state, dt=dt)


@pytest.fixture
def pbdroneenv(env_default_kwargs, simulation_type):
    d = PyBulletDroneEnv(**env_default_kwargs, simulation_type=simulation_type)
    yield d
    d.close()


@pytest.fixture
def nonlineardroneenv(env_default_kwargs):
    d = NonlinearDynamicModelDroneEnv(**env_default_kwargs)
    yield d
    d.close()


@pytest.fixture
def lineardroneenv(env_default_kwargs):
    d = LinearDynamicModelDroneEnv(**env_default_kwargs)
    yield d
    d.close()


@pytest.fixture
def lqrdroneenv(env_default_kwargs):
    d = LQRDroneEnv(**env_default_kwargs)
    yield d
    d.close()


@pytest.fixture(params=[[[-0.1, 0.1], [-0.1, 0.1], [0, 0.2]]])
def position_drone_action_space(request):
    a = np.array(request.param)
    return spaces.Box(low=a[:, 0], high=a[:, 1])


def custom_position_action_space_wrapper(action_space, obj: type[BasePositionDroneEnv]):
    class _A(obj):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_space = action_space

    return _A


@pytest.fixture
def lqrpositiondroneenv(position_drone_action_space, env_default_kwargs):
    d = custom_position_action_space_wrapper(
        position_drone_action_space, LQRPositionDroneEnv
    )(**env_default_kwargs)
    yield d
    d.close()


@pytest.fixture
def polypositiondroneenv(position_drone_action_space, env_default_kwargs):
    d = custom_position_action_space_wrapper(
        position_drone_action_space, PolyPositionDroneEnv
    )(**env_default_kwargs)
    yield d
    d.close()
