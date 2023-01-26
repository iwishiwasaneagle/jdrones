import numpy as np
import pytest
from jdrones.transforms import (
    quat_to_euler,
    euler_to_quat,
    euler_to_rotmat,
    quat_to_rotmat,
)

QUAT_EULER = pytest.mark.parametrize(
    "quat,euler",
    [
        [(0, 0, 0, 1), (0, 0, 0)],
        [(0.5, 0.5, 0.5, 0.5), (1.5707963, 1.5707963, 0)],
        [
            (0.2581989, 0.5163978, 0.7745967, 0.2581989),
            (-1.1071487, 0.7297277, 2.9617391),
        ],
    ],
)

EULER_ROTMAT = pytest.mark.parametrize(
    "euler,rotmat",
    [
        [(0, 0, 0), np.eye(3)],
        [(np.pi / 2, np.pi / 2, 0), [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
        [
            (1, 0, 0),
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5403023, -0.8414710],
                [0.0, 0.8414710, 0.5403023],
            ],
        ],
        [
            (2.6235249, 0.7805774, -1.9971274),
            [
                [-0.2938183, 0.6469092, 0.7036898],
                [0.6469092, 0.6765454, -0.3518449],
                [-0.7036898, 0.3518449, -0.6172729],
            ],
        ],
        [
            (-1.2137551, 0.0894119, -2.3429888),
            [
                [-0.6949205, 0.7135210, 0.0892929],
                [-0.1920070, -0.3037851, 0.9331924],
                [0.6929781, 0.6313497, 0.3481075],
            ],
        ],
    ],
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

