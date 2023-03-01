#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from jdrones.maths import apply_rpy
from jdrones.maths import clip
from jdrones.maths import clip_scalar
from jdrones.maths import quat_mul
from jdrones.maths import yaw
from scipy.spatial.transform import Rotation as R


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


NonNanNonInfFloat = st.floats(
    allow_nan=False,
    allow_subnormal=False,
    allow_infinity=False,
    min_value=-1,
    max_value=1,
)
QUAT = st.tuples(*(NonNanNonInfFloat,) * 4)


@given(
    a=QUAT,
    b=QUAT,
)
def test_quat_mul(a, b):
    assume(np.sum(a) != 0 and np.sum(b) != 0)

    ar = R.from_quat(a)
    br = R.from_quat(b)

    act = quat_mul(ar.as_quat(), br.as_quat())
    exp = (ar * br).as_quat()
    assert np.allclose(act, exp)
