#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import abc
from typing import Dict
from typing import Tuple

import numba
import numpy as np
from jdrones.solvers import bisection_with_right_expansion
from jdrones.types import VEC3


class IdenticalPoseError(RuntimeError):
    pass


class InvalidBoundsError(RuntimeError):
    pass


class BasePolynomialTrajectory(abc.ABC):
    """
    Parent class for all Polynomial Trajectory classes.
    """

    @property
    @abc.abstractmethod
    def T(self):
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

    coeffs: Dict[str, tuple[float, ...]]
    """Storage for the polynomial coefficients"""

    def __init__(
        self,
        start_pos: VEC3,
        dest_pos: VEC3,
        start_vel: VEC3 = (0.0, 0.0, 0.0),
        dest_vel: VEC3 = (0.0, 0.0, 0.0),
        start_acc: VEC3 = (0.0, 0.0, 0.0),
        dest_acc: VEC3 = (0.0, 0.0, 0.0),
        T: float = 5.0,
        _solve: bool = True,
    ):
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.dest_pos = np.array(dest_pos, dtype=np.float64)
        self.start_vel = np.array(start_vel, dtype=np.float64)
        self.dest_vel = np.array(dest_vel, dtype=np.float64)
        self.start_acc = np.array(start_acc, dtype=np.float64)
        self.dest_acc = np.array(dest_acc, dtype=np.float64)

        if (
            np.allclose(self.start_pos, self.dest_pos)
            and np.allclose(self.start_vel, self.dest_vel)
            and np.allclose(self.dest_acc, self.start_acc)
        ):
            raise IdenticalPoseError(
                "Start and target poses are identical. "
                "A polynomial to solve this cannot be calculated."
            )

        self._T = np.float64(T)

        if _solve:
            self._solve()

    @property
    def T(self):
        return self._T

    @staticmethod
    @numba.njit
    def calc_A(T):
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        A = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # f(t=0)
                [
                    T5,
                    T4,
                    T3,
                    T2,
                    T,
                    1.0,
                ],  # f(t=T)
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # f'(t=0)
                [
                    5 * T4,
                    4 * T3,
                    3 * T2,
                    2 * T,
                    1.0,
                    0.0,
                ],  # f'(t=T)
                [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # f''(t=0)
                [20 * T3, 12 * T2, 6 * T, 2, 0.0, 0.0],  # f''(t=T)
            ],
            dtype=np.float64,
        )
        return A

    @staticmethod
    @numba.njit
    def calc_b(pos, vel, acc, tgt_pos, tgt_vel, tgt_acc):
        b = np.empty((6, len(pos)), dtype=np.float64)
        b[0] = pos
        b[1] = tgt_pos
        b[2] = vel
        b[3] = tgt_vel
        b[4] = acc
        b[5] = tgt_acc
        return b

    @classmethod
    def calc_traj(cls, pos, vel, acc, tgt_pos, tgt_vel, tgt_acc, T):
        A = np.array([cls.calc_A(ti) for ti in T])
        b = cls.calc_b(pos, vel, acc, tgt_pos, tgt_vel, tgt_acc)
        return np.linalg.solve(A, b.T)

    def _solve(self) -> None:
        traj = self.calc_traj(
            self.start_pos,
            self.start_vel,
            self.start_acc,
            self.dest_pos,
            self.dest_vel,
            self.dest_acc,
            [self._T],
        )
        self.coeffs = dict(
            x=tuple(traj[0]),
            y=tuple(traj[1]),
            z=tuple(traj[2]),
        )

    @staticmethod
    @numba.njit
    def _position(traj, t):
        return (
            traj[0] * t**5
            + traj[1] * t**4
            + traj[2] * t**3
            + traj[3] * t**2
            + traj[4] * t
            + traj[5]
        )

    @staticmethod
    @numba.njit
    def _velocity(traj, t):
        return (
            5 * traj[0] * t**4
            + 4 * traj[1] * t**3
            + 3 * traj[2] * t**2
            + 2 * traj[3] * t
            + traj[4]
        )

    @staticmethod
    @numba.njit
    def _acceleration(traj, t):
        return 20 * traj[0] * t**3 + 12 * traj[1] * t**2 + 6 * traj[2] * t + 2 * traj[3]

    @staticmethod
    @numba.njit
    def _jerk(traj, t):
        return 60 * traj[0] * t**2 + 24 * traj[1] * t + 6 * traj[2]

    @staticmethod
    @numba.njit
    def _snap(traj, t):
        return 120 * traj[0] * t + 24 * traj[1]

    def snap(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate the snap at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            \\ddot x = 120 c_0 t + 24 c_1

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`\\ddot x, \\ddot y, \\ddot z`
        """

        x4d = self._snap(self.coeffs["x"], t)
        y4d = self._snap(self.coeffs["y"], t)
        z4d = self._snap(self.coeffs["z"], t)
        ret = (x4d, y4d, z4d)
        return ret

    def jerk(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate the jerk at time :math:`t` for :math:`x`, :math:`y`,
        and :math:`z` using

        .. math::
            \\ddot x = 60 c_0 t ^ 2 + 24 c_1 t + 6 c_2

        Parameters
        ----------
        t : float
            Time to evaluate

        Returns
        -------
        float, float, float
             :math:`\\ddot x, \\ddot y, \\ddot z`
        """

        xddd = self._jerk(self.coeffs["x"], t)
        yddd = self._jerk(self.coeffs["y"], t)
        zddd = self._jerk(self.coeffs["z"], t)
        ret = (xddd, yddd, zddd)
        return ret

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

        xdd = self._acceleration(self.coeffs["x"], t)
        ydd = self._acceleration(self.coeffs["y"], t)
        zdd = self._acceleration(self.coeffs["z"], t)
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

        xd = self._velocity(self.coeffs["x"], t)
        yd = self._velocity(self.coeffs["y"], t)
        zd = self._velocity(self.coeffs["z"], t)
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

        x = self._position(self.coeffs["x"], t)
        y = self._position(self.coeffs["y"], t)
        z = self._position(self.coeffs["z"], t)
        ret = (x, y, z)
        return ret


class OptimalFifthOrderPolynomialTrajectory(FifthOrderPolynomialTrajectory):
    def __init__(
        self,
        start_pos: VEC3,
        dest_pos: VEC3,
        start_vel: VEC3 = (0.0, 0.0, 0.0),
        dest_vel: VEC3 = (0.0, 0.0, 0.0),
        start_acc: VEC3 = (0.0, 0.0, 0.0),
        dest_acc: VEC3 = (0.0, 0.0, 0.0),
        tmax: float = 1.0,
        max_acceleration: float = 1.0,
        adaptive_acceleration: bool = False,
        tol: float = 1e-3,
        N: int = 250,
        _solve: bool = True,
    ):
        self.tmax = np.float64(tmax)
        self.max_acceleration = np.float64(max_acceleration)
        self.tol = np.float64(tol)
        self.N = N

        if adaptive_acceleration:
            self.max_acceleration = max(
                self.max_acceleration,
                max(start_acc) + self.tol,
                max(dest_acc) + self.tol,
            )
        elif np.any(np.abs(start_acc) > self.max_acceleration) or np.any(
            np.abs(dest_acc) > self.max_acceleration
        ):
            raise InvalidBoundsError(
                f"Maximum supplied acceleration is outwith the given bounds. Either"
                f"set adaptive_acceleration=True, use "
                f"{FifthOrderPolynomialTrajectory.__name__} "
                f"or adjust the supplied acceleration to be below {max_acceleration=}"
            )

        super().__init__(
            start_pos=start_pos,
            dest_pos=dest_pos,
            start_vel=start_vel,
            dest_vel=dest_vel,
            start_acc=start_acc,
            dest_acc=dest_acc,
            _solve=_solve,
        )

    @property
    def T(self):
        return self.coeffs["t"]

    @staticmethod
    @numba.njit
    def get_acceleration_at_jerk_zero(traj, tmax):
        jerk_coeffs = traj * np.array(
            [60.0, 24.0, 6.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        a = np.sqrt(np.power(jerk_coeffs[1], 2) - 4 * jerk_coeffs[0] * jerk_coeffs[2])
        b = 2 * jerk_coeffs[0]

        if b != 0.0:
            jerk_0 = np.array(
                [
                    (-2 * jerk_coeffs[1] + a) / b,
                    (-2 * jerk_coeffs[1] - a) / b,
                    0.0,
                    tmax,
                ],
                dtype=np.float64,
            )
            jerk_0[0] = min(max(jerk_0[0], 0.0), tmax)
            jerk_0[1] = min(max(jerk_0[1], 0.0), tmax)
        else:
            jerk_0 = np.array([0, tmax], dtype=np.float64)
        ddx = (
            20.0 * traj[0] * np.power(jerk_0, 3)
            + 12.0 * traj[1] * np.power(jerk_0, 2)
            + 6.0 * traj[2] * jerk_0
            + 2.0 * traj[3]
        )
        return ddx

    def get_max_acceleration(self, traj, tmax):
        a = self.get_acceleration_at_jerk_zero(traj, tmax)
        return a[np.abs(a).argmax()]

    def get_max_abs_acceleration(self, traj, tmax):
        return np.abs(self.get_max_acceleration(traj, tmax))

    def get_max_abs_accelerations_from_time(
        self, pos, vel, acc, tgt_pos, tgt_vel, tgt_acc, T
    ):
        if T <= 0:
            return np.float64(np.inf)
        if np.allclose([pos, vel, acc, tgt_pos, tgt_vel, tgt_acc], 0):
            return 0
        traj = self.calc_traj(pos, vel, acc, tgt_pos, tgt_vel, tgt_acc, [T]).squeeze()
        return self.get_max_abs_acceleration(traj, T)

    def _solve(self) -> None:
        times = []
        for i in range(3):

            def f(t):
                a = self.get_max_abs_accelerations_from_time(
                    self.start_pos[[i]],
                    self.start_vel[[i]],
                    self.start_acc[[i]],
                    self.dest_pos[[i]],
                    self.dest_vel[[i]],
                    self.dest_acc[[i]],
                    t,
                )
                return a - self.max_acceleration

            times.append(
                bisection_with_right_expansion(f, 0, self.tmax, tol=self.tol, N=self.N)
            )
        t = [ti for ti in times if ti is not None]
        if len(t) == 0:
            raise RuntimeError("No solution found")
        t = max(t)

        traj = self.calc_traj(
            self.start_pos,
            self.start_vel,
            self.start_acc,
            self.dest_pos,
            self.dest_vel,
            self.dest_acc,
            [t],
        )

        self.coeffs = dict(x=tuple(traj[0]), y=tuple(traj[1]), z=tuple(traj[2]), t=t)


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

        self._T = T

        self._solve()

    @property
    def T(self):
        return self._T

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
