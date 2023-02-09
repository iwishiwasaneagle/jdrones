#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
"""
Tests for transformation helper functions.

.. warning::
    https://stackoverflow.com/a/15050505
"""
import numpy as np
import pybullet
import pytest
from jdrones.transforms import euler_to_quat
from jdrones.transforms import euler_to_rotmat
from jdrones.transforms import quat_to_euler
from jdrones.transforms import quat_to_rotmat


QUAT_EULER = pytest.mark.parametrize(
    "quat,euler",
    [
        [(0.47942554, 0.0, 0.0, 0.87758256), (1, 0, 0)],
        [(0.0, 0.47942554, 0.0, 0.87758256), (0, 1, 0)],
        [(0.0, 0.0, 0.47942554, 0.87758256), (0, 0, 1)],
    ],
)

EULER_ROTMAT = pytest.mark.parametrize(
    "euler,rotmat",
    [
        [(0, 0, 0), np.eye(3)],
        [(np.pi, 0, 0), [[1, 0, 0], [0, -1, 0], [0, 0, -1]]],
        [(0, np.pi, 0), [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]],
        [(0, 0, np.pi), [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]],
    ],
)


@QUAT_EULER
def test_quat_euler_sanity_check(quat, euler):
    assert np.allclose(
        pybullet.getEulerFromQuaternion(pybullet.getQuaternionFromEuler(euler)), euler
    )
    assert np.allclose(
        pybullet.getQuaternionFromEuler(pybullet.getEulerFromQuaternion(quat)), quat
    )


@QUAT_EULER
def test_quat_to_euler(quat, euler):
    act_euler = quat_to_euler(quat)
    assert np.allclose(euler, act_euler)


@QUAT_EULER
def test_euler_to_quat(quat, euler):
    act_quat = euler_to_quat(euler)
    assert np.allclose(quat, act_quat) or np.allclose(-1 * quat, act_quat)


@EULER_ROTMAT
def test_euler_to_rotmat(euler, rotmat):
    act_rotmat = euler_to_rotmat(euler)
    assert np.allclose(rotmat, act_rotmat)


@EULER_ROTMAT
def test_quat_to_rotmat(euler, rotmat):
    quat = euler_to_quat(euler)
    act_rotmat = quat_to_rotmat(quat)
    assert np.allclose(rotmat, act_rotmat)
