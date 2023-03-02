#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.trajectory import QuinticPolynomialTrajectory


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
def trajectory(start_pos, start_vel, start_acc, dest_pos, dest_vel, dest_acc, T):
    return QuinticPolynomialTrajectory(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        dest_pos=dest_pos,
        dest_vel=dest_vel,
        dest_acc=dest_acc,
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
def test_traj_coeffs_only_pos(trajectory, exp):
    act = trajectory.coeffs

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
            (1, 0, 0),
        ],
        [
            (-3, -2, -1),
            (1, 2, 3),
        ],
    ],
    indirect=True,
)
@pytest.mark.parametrize("T", [5, 10], indirect=True)
def test_traj_bounds(
    trajectory, T, start_pos, dest_pos, start_vel, dest_vel, start_acc, dest_acc
):
    act_start_pos = trajectory.position(0)
    act_dest_pos = trajectory.position(T)
    assert np.allclose(act_start_pos, start_pos)
    assert np.allclose(act_dest_pos, dest_pos)

    act_start_vel = trajectory.velocity(0)
    act_dest_vel = trajectory.velocity(T)
    assert np.allclose(act_start_vel, start_vel)
    assert np.allclose(act_dest_vel, dest_vel)

    act_start_acc = trajectory.acceleration(0)
    act_dest_acc = trajectory.acceleration(T)
    assert np.allclose(act_start_acc, start_acc)
    assert np.allclose(act_dest_acc, dest_acc)
