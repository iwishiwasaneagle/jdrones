#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import numpy.typing as npt
from jdrones.transforms import euler_to_rotmat
from jdrones.types import VEC3
from jdrones.types import VEC4


def clip_scalar(value: float, vmin: float, vmax: float) -> float:
    return vmin if value < vmin else vmax if value > vmax else value


def clip(value: npt.ArrayLike, vmin: float, vmax: float) -> npt.ArrayLike:
    return np.core.umath.maximum(np.core.umath.minimum(value, vmax), vmin)


def yaw(x1: float, y1: float, x2: float, y2: float) -> float:
    """

    Parameters
    ----------
    x1 : float
        From :math:`x`
    y1 : float
        From :math:`y`
    x2 : float
        To :math:`x`
    y2 : float
        To :math:`y`

    Returns
    -------
    float
        :math:`\\psi` (rad)

    """
    return np.arctan2(y2 - y1, x2 - x1)


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt(np.sum(np.square([x2 - x1, y2 - y1])))


def apply_rpy(value: VEC3, rpy: VEC3) -> VEC3:
    return np.dot(value, euler_to_rotmat(rpy))


def quat_mul(a: VEC4, b: VEC4) -> VEC4:
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )
