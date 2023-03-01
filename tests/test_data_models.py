#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.data_models import State


@pytest.fixture(params=[(0,) * 20])
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
    state = State([1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0])
    state_to_x = state.to_x()
    assert np.allclose(state_to_x[:3], (1, 2, 3))
    assert np.allclose(state_to_x[3:6], (7, 8, 9))
    assert np.allclose(state_to_x[6:9], (4, 5, 6))
    assert np.allclose(state_to_x[9:], (10, 11, 12))


def test_x_to_state():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x_to_state = State.from_x(x)
    assert np.allclose(x_to_state.pos, (1, 2, 3))
    assert np.allclose(x_to_state.vel, (4, 5, 6))
    assert np.allclose(x_to_state.rpy, (7, 8, 9))
    assert np.allclose(x_to_state.ang_vel, (10, 11, 12))


@pytest.mark.parametrize(
    "quat,state,exp", [[(0, 0, 0, 1), np.arange(20), np.arange(20)]], indirect=["state"]
)
def test_state_apply_quat(quat, state, exp):
    assert np.allclose(state.apply_quat(quat), exp)
