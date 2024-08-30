#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.trajectory import FifthOrderPolynomialTrajectory
from jdrones.trajectory import FirstOrderPolynomialTrajectory
from jdrones.trajectory import OptimalFifthOrderPolynomialTrajectory


@pytest.fixture
def vec3_factory():
    def fn(x, y, z):
        return np.array([x, y, z], dtype=np.float64)

    return fn


@pytest.fixture(params=[(0, 0, 0)])
def start_pos(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[(0, 0, 0)])
def start_vel(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[(0, 0, 0)])
def start_acc(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[(1, 1, 1)])
def dest_pos(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[(0, 0, 0)])
def dest_vel(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[(0, 0, 0)])
def dest_acc(request, vec3_factory):
    return vec3_factory(*request.param)


@pytest.fixture(params=[5, 10])
def T(request):
    return request.param


@pytest.fixture(autouse=True)
def mark_skip(start_pos, start_vel, start_acc, dest_pos, dest_vel, dest_acc):
    start = np.array([start_pos, start_vel, start_acc])
    dest = np.array([dest_pos, dest_vel, dest_acc])

    if np.allclose(start, dest):
        pytest.skip("Start and dest poses are identical")


@pytest.fixture()
def fifth_o_trajectory(
    start_pos, start_vel, start_acc, dest_pos, dest_vel, dest_acc, T
):
    return FifthOrderPolynomialTrajectory(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        dest_pos=dest_pos,
        dest_vel=dest_vel,
        dest_acc=dest_acc,
        T=T,
    )


@pytest.fixture()
def optimal_fifth_o_trajectory(
    start_pos, start_vel, start_acc, dest_pos, dest_vel, dest_acc
):
    traj = OptimalFifthOrderPolynomialTrajectory(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        dest_pos=dest_pos,
        dest_vel=dest_vel,
        dest_acc=dest_acc,
        adaptive_acceleration=True,
        _solve=True,
    )
    yield traj


@pytest.fixture()
def first_o_trajectory(start_pos, dest_pos, T):
    return FirstOrderPolynomialTrajectory(
        start_pos=start_pos,
        dest_pos=dest_pos,
        T=T,
    )


DATA = [
    [
        (0, 0, 0),
        (0, 0, 0),
    ],
    [
        (0, 0, 0),
        (1, 0, 0),
    ],
    [
        (-3, -2, -1),
        (1, 2, 3),
    ],
    [(10, -20, 34), (6, 89, -20)],
]

ACC = pytest.mark.parametrize(
    "start_acc,dest_acc",
    DATA,
    indirect=True,
)
VEL = pytest.mark.parametrize(
    "start_vel,dest_vel",
    DATA,
    indirect=True,
)
POS = pytest.mark.parametrize(
    "start_pos,dest_pos",
    DATA,
    indirect=True,
)


@ACC
@VEL
@POS
def test_fifth_o_traj_bounds(
    fifth_o_trajectory, T, start_pos, dest_pos, start_vel, dest_vel, start_acc, dest_acc
):
    act_start_pos = fifth_o_trajectory.position(0)
    act_dest_pos = fifth_o_trajectory.position(T)
    assert np.allclose(act_start_pos, start_pos)
    assert np.allclose(act_dest_pos, dest_pos)

    act_start_vel = fifth_o_trajectory.velocity(0)
    act_dest_vel = fifth_o_trajectory.velocity(T)
    assert np.allclose(act_start_vel, start_vel)
    assert np.allclose(act_dest_vel, dest_vel)

    act_start_acc = fifth_o_trajectory.acceleration(0)
    act_dest_acc = fifth_o_trajectory.acceleration(T)
    assert np.allclose(act_start_acc, start_acc)
    assert np.allclose(act_dest_acc, dest_acc)


@ACC
@VEL
@POS
def test_optimal_fifth_o_traj_bounds(
    optimal_fifth_o_trajectory: OptimalFifthOrderPolynomialTrajectory,
    start_pos,
    dest_pos,
    start_vel,
    dest_vel,
    start_acc,
    dest_acc,
):
    t = optimal_fifth_o_trajectory.T
    act_start_pos = optimal_fifth_o_trajectory.position(0)
    act_dest_pos = optimal_fifth_o_trajectory.position(t)
    assert np.allclose(act_start_pos, start_pos)
    assert np.allclose(act_dest_pos, dest_pos)

    act_start_vel = optimal_fifth_o_trajectory.velocity(0)
    act_dest_vel = optimal_fifth_o_trajectory.velocity(t)
    assert np.allclose(act_start_vel, start_vel)
    assert np.allclose(act_dest_vel, dest_vel)

    act_start_acc = optimal_fifth_o_trajectory.acceleration(0)
    act_dest_acc = optimal_fifth_o_trajectory.acceleration(t)
    assert np.allclose(act_start_acc, start_acc)
    assert np.allclose(act_dest_acc, dest_acc)

    times = np.linspace(0, t)
    acc = np.array([optimal_fifth_o_trajectory.acceleration(ti) for ti in times])
    max_acc = np.abs(acc).max()
    assert max_acc <= optimal_fifth_o_trajectory.max_acceleration


@POS
def test_first_o_traj_bounds(first_o_trajectory, T, start_pos, dest_pos):
    act_start_pos = first_o_trajectory.position(0)
    first_o_trajectory.position(T / 2)
    first_o_trajectory.position(T * 2)
    act_dest_pos = first_o_trajectory.position(T)
    assert np.allclose(act_start_pos, start_pos)
    assert np.allclose(act_dest_pos, dest_pos)
