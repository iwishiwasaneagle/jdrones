#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numba
import numpy as np
from jdrones.types import MAT3X3
from jdrones.types import VEC3
from jdrones.types import VEC4


@numba.njit
def quat_to_euler(quat: VEC4) -> VEC3:
    """
    PyBullet does a Body 3-2-1 sequence:

        1. Body Yaw (X)
        2. Body Pitch (Y)
        3. Body Roll (Z)

    ..warning::

        https://stackoverflow.com/a/15050505

    More details can be found here
    https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/examples/pybullet/pybullet.c#L10854

    Pure python equivalent:

    ..code-block:: python
        def euler_to_quat(roll, pitch, yaw):
            phi, the, psi = roll / 2, pitch / 2, yaw / 2
            q = np.array(
                (
                    sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi),
                    cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi),
                    cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi),
                    cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi),
                )
            )
            length = np.sqrt(np.sum(np.square(q)))
            return q / length

    ..warning::

        https://stackoverflow.com/a/15050505

    Parameters
    ----------
    quat : float,float,float,float
        Quaternion (x,y,z,w)

    Returns
    -------
    float,float,float
        Euler angle (roll,pitch,yaw)

    """
    x, y, z, w = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.sqrt(1 + 2 * (w * y - x * z))
    cosp = np.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


@numba.njit
def euler_to_quat(euler: VEC3) -> VEC4:
    """PyBullet does a Body 3-2-1 sequence:

        1. Body Yaw (X)
        2. Body Pitch (Y)
        3. Body Roll (Z)

    More details can be found here
    https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/examples/pybullet/pybullet.c#L10854

    ..warning::

        https://stackoverflow.com/a/15050505

    ..code-block:: python
        def quat_to_euler(x, y, z, w):
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = np.sqrt(1 + 2 * (w * y - x * z))
            cosp = np.sqrt(1 - 2 * (w * y - x * z))
            pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            return roll, pitch, yaw


    Parameters
    ----------
    euler : float,float,float
        Euler angles (roll pitch yaw)

    Returns
    -------
    float,float,float,float
        Quaternion (x,y,z,w)

    """
    roll, pitch, yaw = euler

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])


@numba.njit
def quat_to_rotmat(quat: VEC4) -> MAT3X3:
    x, y, z, w = quat
    x2, y2, z2, w2 = quat * quat
    xy, xz, xw, yz, yw, zw = (
        quat[0] * quat[1],
        quat[0] * quat[2],
        quat[0] * quat[3],
        quat[1] * quat[2],
        quat[1] * quat[3],
        quat[2] * quat[3],
    )

    rot_mat = np.array(
        [
            [1 - 2 * (y2 + z2), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (x2 + z2), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (x2 + y2)],
        ]
    )
    return rot_mat


@numba.njit
def euler_to_rotmat(euler: VEC3) -> MAT3X3:
    return quat_to_rotmat(euler_to_quat(euler))
