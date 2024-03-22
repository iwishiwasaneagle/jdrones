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


def test_URDFModel_hashing(urdfmodel):
    assert hash(urdfmodel) is not None
    assert hash(urdfmodel) == hash(urdfmodel)

    urdfmodel.g = 0
    hash1 = hash(urdfmodel)
    urdfmodel.g += 1
    hash2 = hash(urdfmodel)
    assert hash1 != hash2


@pytest.mark.parametrize(
    "state", [np.full(20, -5), np.full(20, 5), np.zeros(20)], indirect=True
)
def test_state_normed(state: State):
    normed_state = state.normed(np.column_stack([np.full(20, -5), np.full(20, 5)]))

    assert np.all((normed_state >= -1) & (normed_state <= 1))


@pytest.mark.parametrize(
    "state,normalization_lims",
    [(list(range(20)), [(-1 + i, 1 + i) for i in range(20)])],
    indirect=["state"],
)
def test_state_normed_varying_lims(state: State, normalization_lims):
    normed_state = state.normed(normalization_lims)
    assert np.allclose(normed_state, 0)
