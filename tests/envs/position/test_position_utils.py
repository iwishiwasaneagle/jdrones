import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jdrones.envs.position import PolynomialPositionBaseDronEnv

OneD = st.floats(allow_nan=False)
VEC3 = st.tuples(*(OneD,) * 3)
VEC6 = st.tuples(*(OneD,) * 6)


@st.composite
def var_length_2d_action(draw):
    size = draw(st.integers(min_value=1, max_value=100).filter(lambda v: v != 3))
    pos = st.lists(OneD, min_size=size, max_size=size)
    vel = st.lists(OneD, min_size=size, max_size=size)
    return draw(pos), draw(vel)


@given(action=VEC3)
def test__validate_action_input_vec3(action):
    action_as_state = PolynomialPositionBaseDronEnv._validate_action_input(action)
    assert np.allclose(action_as_state.pos, action)
    assert np.sum(action_as_state[3:]) == 0.0


@given(pos=VEC3, vel=VEC3)
def test__validate_action_input_vec3_vec3(pos, vel):
    action_as_state = PolynomialPositionBaseDronEnv._validate_action_input((pos, vel))
    assert np.allclose(action_as_state.pos, pos)
    assert np.allclose(action_as_state.vel, vel)

    action_as_state.pos = (0, 0, 0)
    action_as_state.vel = (0, 0, 0)

    assert np.sum(action_as_state) == 0.0


@given(action=VEC6)
def test__validate_action_input_vec6(action):
    pos, vel = action[:3], action[3:]
    action_as_state = PolynomialPositionBaseDronEnv._validate_action_input(action)
    assert np.allclose(action_as_state.pos, pos)
    assert np.allclose(action_as_state.vel, vel)

    action_as_state.pos = (0, 0, 0)
    action_as_state.vel = (0, 0, 0)

    assert np.sum(action_as_state) == 0.0


@given(
    action=st.one_of(
        st.lists(OneD).filter(lambda v: len(v) not in (3, 6)), var_length_2d_action()
    )
)
def test__validate_action_input_invalid(action):
    with pytest.raises(ValueError):
        PolynomialPositionBaseDronEnv._validate_action_input(action)
