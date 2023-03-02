#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Dict
from typing import Tuple

import nptyping
import numpy as np
from jdrones.types import VEC3


class QuinticPolynomialTrajectory:

    coeffs: Dict[str, nptyping.NDArray[nptyping.Shape["1,6"], nptyping.Double]]

    def __init__(
        self,
        start_pos: VEC3,
        dest_pos: VEC3,
        T: float = 5,
        start_vel: VEC3 = (0, 0, 0),
        dest_vel: VEC3 = (0, 0, 0),
        start_acc: VEC3 = (0, 0, 0),
        dest_acc: VEC3 = (0, 0, 0),
    ):

        self.start_pos = start_pos
        self.dest_pos = dest_pos

        self.start_vel = start_vel
        self.dest_vel = dest_vel

        self.start_acc = start_acc
        self.dest_acc = dest_acc

        self.T = T

        self._solve()

    def _solve(self) -> None:
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],  # f(t=0)
                [
                    self.T**5,
                    self.T**4,
                    self.T**3,
                    self.T**2,
                    self.T,
                    1,
                ],  # f(t=T)
                [0, 0, 0, 0, 1, 0],  # f'(t=0)
                [
                    5 * self.T**4,
                    4 * self.T**3,
                    3 * self.T**2,
                    2 * self.T,
                    1,
                    0,
                ],  # f'(t=T)
                [0, 0, 0, 2, 0, 0],  # f''(t=0)
                [20 * self.T**3, 12 * self.T**2, 6 * self.T, 2, 0, 0],  # f''(t=T)
            ]
        )

        b = np.row_stack(
            [
                self.start_pos,
                self.dest_pos,
                self.start_vel,
                self.dest_vel,
                self.start_acc,
                self.dest_acc,
            ]
        )

        self.coeffs = dict(
            x=np.linalg.solve(A, b[:, 0]),
            y=np.linalg.solve(A, b[:, 1]),
            z=np.linalg.solve(A, b[:, 2]),
        )

    def acceleration(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: 20 * c[0] * t**3
            + 12 * c[1] * t**2
            + 6 * c[2] * t
            + 2 * c[3]
        )
        xdd = calc(self.coeffs["x"], t)
        ydd = calc(self.coeffs["y"], t)
        zdd = calc(self.coeffs["z"], t)
        ret = (xdd, ydd, zdd)
        return ret

    def velocity(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: 5 * c[0] * t**4
            + 4 * c[1] * t**3
            + 3 * c[2] * t**2
            + 2 * c[3] * t
            + c[4]
        )
        xd = calc(self.coeffs["x"], t)
        yd = calc(self.coeffs["y"], t)
        zd = calc(self.coeffs["z"], t)
        ret = (xd, yd, zd)
        return ret

    def position(self, t: float) -> Tuple[float, float, float]:
        calc = (
            lambda c, t: c[0] * t**5
            + c[1] * t**4
            + c[2] * t**3
            + c[3] * t**2
            + c[4] * t
            + c[5]
        )
        x = calc(self.coeffs["x"], t)
        y = calc(self.coeffs["y"], t)
        z = calc(self.coeffs["z"], t)
        ret = (x, y, z)
        return ret
