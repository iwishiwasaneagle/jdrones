#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.data_models import State
from jdrones.data_models import STATE_ENUM
from scipy.spatial.transform import Rotation as R


@pytest.fixture(params=[(0.0,) * 20])
def state(request):
    return State(request.param)


def test_state_from_empty():
    assert np.allclose(State(), (0,) * 20)


@pytest.mark.parametrize("n", [19, 21, 1])
def test_state_incorrect_shape(n):
    with pytest.raises(ValueError):
        State((0,) * n)


@pytest.mark.parametrize("attr", ["pos", "rpy", "vel", "ang_vel"])
@pytest.mark.parametrize("value", [(1, 2, 3)])
def test_state_3vec(state, attr, value):
    assert np.allclose(getattr(state, attr), (0, 0, 0))

    setattr(state, attr, value)

    assert np.allclose(getattr(state, attr), value)


@pytest.mark.parametrize("attr", ["quat", "prop_omega"])
@pytest.mark.parametrize("value", [(1, 2, 3, 4)])
def test_state_4vec(state, attr, value):
    assert np.allclose(getattr(state, attr), (0, 0, 0, 0))

    setattr(state, attr, value)

    assert np.allclose(getattr(state, attr), value)


@pytest.mark.parametrize("s", np.random.random((5, 20)))
def test_state_to_x_to_state(s):
    state = State(s)
    act = State.from_x(state.to_x())
    for attr in ("pos", "vel", "rpy", "ang_vel"):
        assert np.allclose(getattr(state, attr), getattr(act, attr))
    assert np.allclose(act.quat, np.zeros(4))
    assert np.allclose(act.prop_omega, np.zeros(4))


def test_state_to_x():
    state = State(
        [
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    state_to_x = state.to_x()
    assert np.allclose(state_to_x[:3], (1, 2, 3))
    assert np.allclose(state_to_x[3:6], (7, 8, 9))
    assert np.allclose(state_to_x[6:9], (4, 5, 6))
    assert np.allclose(state_to_x[9:], (10, 11, 12))


def test_x_to_state():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    x_to_state = State.from_x(x)
    assert np.allclose(x_to_state.pos, (1, 2, 3))
    assert np.allclose(x_to_state.vel, (4, 5, 6))
    assert np.allclose(x_to_state.rpy, (7, 8, 9))
    assert np.allclose(x_to_state.ang_vel, (10, 11, 12))


@pytest.mark.parametrize(
    "quat,act,exp",
    [
        [
            (0.0, 0.0, 0.0, 1.0),
            *(
                [
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                ],
            )
            * 2,
        ],
        [
            (1.0, 0.0, 0.0, 0.0),
            [
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            [
                1.0,
                -2.0,
                -3.0,
                1.0,
                0.0,
                0.0,
                0.0,
                np.pi,
                0.0,
                0.0,
                4.0,
                -5.0,
                -6.0,
                7.0,
                -8.0,
                -9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
        ],
        [
            (0.0, 1.0, 0.0, 0.0),
            [
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            [
                -1.0,
                2.0,
                -3.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                np.pi,
                0.0,
                -4.0,
                5.0,
                -6.0,
                -7.0,
                8.0,
                -9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
        ],
        [
            (0.0, 0.0, 1.0, 0.0),
            [
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            [
                -1.0,
                -2.0,
                3.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                np.pi,
                -4.0,
                -5.0,
                6.0,
                -7.0,
                -8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
        ],
    ],
)
def test_state_quat_rotation(quat, act, exp):
    exp = State(exp)
    rotated = State(act).quat_rotation(quat)
    assert np.allclose(rotated.pos, exp.pos)
    assert np.allclose(rotated.quat, exp.quat)
    assert np.allclose(
        R.from_euler("ZYX", rotated.rpy).as_quat(),
        R.from_euler("ZYX", exp.rpy).as_quat(),
    )
    assert np.allclose(rotated.ang_vel, exp.ang_vel)
    assert np.allclose(rotated.vel, exp.vel)
    assert np.allclose(rotated.prop_omega, exp.prop_omega)


def test_as_str_list():
    assert tuple(STATE_ENUM.as_list()) == (
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "phi",
        "theta",
        "psi",
        "vx",
        "vy",
        "vz",
        "p",
        "q",
        "r",
        "P0",
        "P1",
        "P2",
        "P3",
    )


def test_as_str_list_fail():
    assert (
        tuple(STATE_ENUM.as_list())
        != (
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "phi",
            "theta",
            "psi",
            "vx",
            "vy",
            "vz",
            "p",
            "q",
            "r",
            "P0",
            "P1",
            "P2",
            "P3",
        )[::-1]
    )
