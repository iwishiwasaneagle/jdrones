#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jdrones.maths import apply_rpy
from jdrones.maths import clip
from jdrones.maths import clip_scalar
from jdrones.maths import yaw


@given(
    value=st.floats(allow_nan=False),
    vmin=st.floats(max_value=0, allow_nan=False),
    vmax=st.floats(min_value=1, allow_nan=False),
)
def test_clip_scalar(value, vmin, vmax):
    clipped = clip_scalar(value, vmin, vmax)
    assert vmin <= clipped <= vmax


@given(
    value=st.lists(st.floats(allow_nan=False), min_size=1),
    vmin=st.floats(max_value=0, allow_nan=False),
    vmax=st.floats(min_value=1, allow_nan=False),
)
def test_clip_array(value, vmin, vmax):
    clipped = clip(value, vmin, vmax)
    assert np.all((vmin <= np.asarray(clipped)) <= vmax)


@pytest.mark.parametrize(
    "a,b,exp",
    [
        [(0, 0), (1, 0), 0],
        [(0, 0), (0, 1), np.pi / 2],
        [(0, 0), (0, -1), -np.pi / 2],
        [(0, 0), (-1, 0.0000001), np.pi],
        [(0, 0), (-1, -0.0000001), -np.pi],
        [(100, 50), (100, 70), np.pi / 2],
        [(1, 2), (2, 4), 1.1071487177940904],
    ],
)
def test_yaw(a, b, exp):
    act = yaw(*a, *b)
    assert np.isclose(act, exp)


@pytest.mark.parametrize(
    "value,rpy,exp",
    [
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(1, 0, 0), (0, 0, 0), (1, 0, 0)],
        [(0, 1, 0), (0, 0, 0), (0, 1, 0)],
        [(0, 0, 1), (0, 0, 0), (0, 0, 1)],
        [(1, 0, 0), (0, 0, np.pi / 2), (0, -1, 0)],
        [(0, 1, 0), (0, 0, np.pi / 2), (1, 0, 0)],
        [(0, 0, 1), (0, 0, np.pi / 2), (0, 0, 1)],
        [(1, 1, 0), (np.pi / 2, 0, 0), (1, 0, -1)],
    ],
)
def test_apply_rpy(value, rpy, exp):
    act = apply_rpy(value, rpy)
    assert np.allclose(act, exp)
