#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.trajectory import FifthOrderPolynomialTrajectory
from jdrones.trajectory import FirstOrderPolynomialTrajectory


@pytest.fixture
def vec3_factory():
    def fn(x, y, z):
        return x, y, z

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


@pytest.fixture(params=[5])
def T(request):
    return request.param


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
def first_o_trajectory(start_pos, dest_pos, T):
    return FirstOrderPolynomialTrajectory(
        start_pos=start_pos,
        dest_pos=dest_pos,
        T=T,
    )


@pytest.mark.parametrize(
    "start_pos,dest_pos,exp",
    [
        [
            (0, 0, 0),
            (1, 0, 0),
            dict(
                x=[
                    0.00192,
                    -0.024,
                    0.08,
                    0,
                    0,
                    0,
                ],
                y=np.zeros(6),
                z=np.zeros(6),
            ),
        ],
        [
            (0, 0, 0),
            (0, 1, 0),
            dict(
                y=[
                    0.00192,
                    -0.024,
                    0.08,
                    0,
                    0,
                    0,
                ],
                x=np.zeros(6),
                z=np.zeros(6),
            ),
        ],
        [
            (0, 0, 0),
            (0, 0, 1),
            dict(
                z=[
                    0.00192,
                    -0.024,
                    0.08,
                    0,
                    0,
                    0,
                ],
                y=np.zeros(6),
                x=np.zeros(6),
            ),
        ],
    ],
    indirect=["start_pos", "dest_pos"],
)
def test_fifth_o_traj_coeffs_only_pos(fifth_o_trajectory, exp):
    act = fifth_o_trajectory.coeffs

    assert all(np.allclose(exp[f], act[f]) for f in ("x", "y", "z"))


@pytest.mark.parametrize(
    "start_acc,dest_acc",
    [
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
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "start_vel,dest_vel",
    [
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
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "start_pos,dest_pos",
    [
        [
            (0, 0, 0),
            (0, 0, 0),
        ],
        [
            (0, 0, 0),
            (1, 0, 0),
        ],
        [
            (0, 0, 0),
            (0, 1, 0),
        ],
        [
            (0, 0, 0),
            (0, 0, 1),
        ],
        [
            (-3, -2, -1),
            (1, 2, 3),
        ],
    ],
    indirect=True,
)
@pytest.mark.parametrize("T", [5, 10], indirect=True)
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


@pytest.mark.parametrize(
    "start_pos,dest_pos,exp",
    [
        [
            (0, 0, 0),
            (1, 0, 0),
            dict(
                x=[0.2, 0],
                y=np.zeros(2),
                z=np.zeros(2),
            ),
        ],
        [
            (0, 0, 0),
            (0, 1, 0),
            dict(
                y=[0.2, 0],
                x=np.zeros(2),
                z=np.zeros(2),
            ),
        ],
        [
            (0, 0, 0),
            (0, 0, 1),
            dict(
                z=[0.2, 0],
                y=np.zeros(2),
                x=np.zeros(2),
            ),
        ],
    ],
    indirect=["start_pos", "dest_pos"],
)
def test_first_o_traj_coeffs_only_pos(first_o_trajectory, exp):
    act = first_o_trajectory.coeffs

    assert all(np.allclose(exp[f], act[f]) for f in ("x", "y", "z"))


@pytest.mark.parametrize(
    "start_pos,dest_pos",
    [
        [
            (0, 0, 0),
            (0, 0, 0),
        ],
        [
            (0, 0, 0),
            (1, 0, 0),
        ],
        [
            (0, 0, 0),
            (0, 1, 0),
        ],
        [
            (0, 0, 0),
            (0, 0, 1),
        ],
        [
            (-3, -2, -1),
            (1, 2, 3),
        ],
        np.random.random((2, 3)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("T", [5, 10], indirect=True)
def test_first_o_traj_bounds(first_o_trajectory, T, start_pos, dest_pos):
    act_start_pos = first_o_trajectory.position(0)
    first_o_trajectory.position(T / 2)
    first_o_trajectory.position(T * 2)
    act_dest_pos = first_o_trajectory.position(T)
    assert np.allclose(act_start_pos, start_pos)
    assert np.allclose(act_dest_pos, dest_pos)
