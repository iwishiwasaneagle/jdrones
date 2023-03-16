#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Dict
from typing import Tuple

import numpy as np
from jdrones.types import VEC3


class BasePolynomialTrajectory:
    """
    Parent class for all Polynomial Trajectory classes.
    """

    pass


class FifthOrderPolynomialTrajectory(BasePolynomialTrajectory):
    """
    A helper class to solve the quintic polynomial given a start and end criterion.

    Define the problem as

    .. math::
        \\begin{align}
        x &= c_0 t ^5 + c_1 t^4 + c_2 t ^3 + c_3 t ^2 + c_4 t + c_5,\\\\
        \\dot x &= 5 c_0 t ^ 4 + 4 c_1 t^3 + 3 c_2 t ^2 + 2 c_3 t + c_4,\\\\
        \\ddot x &= 20 c_0 t ^ 3 + 12 c_1 t^2 + 6 c_2 t + 2 c_3
        \\end{align}

    Which is then solved by

    .. math::
       \\begin{bmatrix}
            0 & 0 & 0 & 0 & 0 & 1 \\\\
            T^5 & T^4 & T^3 & T^2 & T & 1 \\\\
            0 & 0 & 0 & 0 & 1 & 0 \\\\
            5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0 \\\\
            0 & 0 & 0 & 2 & 0 & 0 \\\\
            20T^3 & 12T^2 & 6T & 2 & 0 & 0
        \\end{bmatrix}
        \\begin{bmatrix}
            x_{t=0}\\\\
            x_{t=T}\\\\
            \\dot x_{t=0}\\\\
            \\dot x_{t=T}\\\\
            \\ddot x_{t=0}\\\\
            \\ddot x_{t=T}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
            c_0\\\\
            c_1\\\\
            c_2\\\\
            c_3\\\\
            c_4\\\\
            c_5\\\\
        \\end{bmatrix}

    Where :math:`T` is the total time to traverse the polynomial.

    .. seealso::
        :meth:`jdrones.envs.FifthOrderPolyPositionDroneEnv.calc_traj`

    The coefficients are then saved in :code:`coeffs` and accessed by
    :meth:`acceleration`,:meth:`velocity`, and :meth:`position` as required.


    .. note::
        The trajectory is solved individually for :math:`x`, :math:`y`, and :math:`z`.

    """

    coeffs: Dict[str, tuple[float, float, float, float, float, float]]
    """Storage for the polynomial coefficients"""

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
        """
        Calculate the acceleration at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            \\ddot x = 20 c_0 t ^ 3 + 12 c_1 t^2 + 6 c_2 t + 2 c_3

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`\\ddot x, \\ddot y, \\ddot z`
        """

        def calc(c, t):
            return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]

        xdd = calc(self.coeffs["x"], t)
        ydd = calc(self.coeffs["y"], t)
        zdd = calc(self.coeffs["z"], t)
        ret = (xdd, ydd, zdd)
        return ret

    def velocity(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate the velocity at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            \\dot x = 5 c_0 t ^ 4 + 4 c_1 t^3 + 3 c_2 t ^2 + 2 c_3 t + c_4

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`\\dot x, \\dot y, \\dot z`
        """

        def calc(c, t):
            return (
                5 * c[0] * t**4
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
        """
        Calculate the position at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            x = c_0 t ^5 + c_1 t^4 + c_2 t ^3 + c_3 t ^2 + c_4 t + c_5

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`x, y, z`
        """

        def calc(c, t):
            return (
                c[0] * t**5
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


class FirstOrderPolynomialTrajectory(BasePolynomialTrajectory):
    """
    A helper class to solve the first-order polynomial given a start and end criterion.

    Define the problem as

    .. math::
        x = c_0 t + c_1

    Which is then solved by

    .. math::
       \\begin{bmatrix}
            0 & 1 \\\\
            T & 1
        \\end{bmatrix}
        \\begin{bmatrix}
            x_{t=0}\\\\
            x_{t=T}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
            c_0\\\\
            c_1
        \\end{bmatrix}

    Where :math:`T` is the total time to traverse the polynomial.

    .. seealso::
        :meth:`jdrones.envs.FirstOrderPolyPositionDroneEnv.calc_traj`

    The coefficients are then saved in :code:`coeffs` and accessed by
    :meth:`acceleration`,:meth:`velocity`, and :meth:`position` as required.


    .. note::
        The trajectory is solved individually for :math:`x`, :math:`y`, and :math:`z`.

    """

    coeffs: Dict[str, tuple[float, float, float, float, float, float]]
    """Storage for the polynomial coefficients"""

    def __init__(
        self,
        start_pos: VEC3,
        dest_pos: VEC3,
        T: float = 5,
    ):
        self.start_pos = start_pos
        self.dest_pos = dest_pos

        self.T = T

        self._solve()

    def _solve(self) -> None:
        A = np.array(
            [
                [0, 1],  # f(t=0)
                [
                    self.T,
                    1,
                ],  # f(t=T)
            ]
        )

        b = np.row_stack([self.start_pos, self.dest_pos])

        self.coeffs = dict(
            x=np.linalg.solve(A, b[:, 0]),
            y=np.linalg.solve(A, b[:, 1]),
            z=np.linalg.solve(A, b[:, 2]),
        )

    def position(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate the position at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            x = c_0 t + c_1

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`x, y, z`
        """

        def calc(c, t):
            return c[0] * t + c[1]

        x = calc(self.coeffs["x"], t)
        y = calc(self.coeffs["y"], t)
        z = calc(self.coeffs["z"], t)
        ret = (x, y, z)
        return ret
