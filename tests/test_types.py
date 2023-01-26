import numpy as np
import pytest

from jdrones.types import State, PropellerAction


@pytest.fixture(params=[(0,) * 20])
def state(request):
    return State(request.param)


@pytest.fixture(params=[(0,) * 4])
def action(request):
    return PropellerAction(request.param)


def test_state_from_empty():
    assert np.allclose(State(), (0,) * 20)


@pytest.mark.parametrize("n", [19, 21, 1])
def test_state_incorrect_shape(n):
    with pytest.raises(ValueError):
        State((0,) * n)


def test_action_from_empty():
    assert np.allclose(PropellerAction(), (0,) * 4)


@pytest.mark.parametrize("n", [3, 5])
def test_action_incorrect_shape(n):
    with pytest.raises(ValueError):
        PropellerAction((0,) * n)


@pytest.mark.parametrize("attr", ["pos", "rpy", "vel", "ang_vel"])
@pytest.mark.parametrize("value", [(1, 2, 3)])
def test_state_3vec(state, attr, value):
    assert np.allclose(getattr(state, attr), (0, 0, 0))

    setattr(state, attr, value)

    assert np.allclose(getattr(state, attr), value)


@pytest.mark.parametrize("attr", ["quat", "prop_omega"])
@pytest.mark.parametrize("value", [(1, 2, 3, 4)])
def test_state_3vec(state, attr, value):
    assert np.allclose(getattr(state, attr), (0, 0, 0, 0))

    setattr(state, attr, value)

    assert np.allclose(getattr(state, attr), value)
