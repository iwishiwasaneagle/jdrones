#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import abc
from typing import Dict
from typing import Tuple

import numpy as np
from jdrones.types import VEC3
from libjdrones import FifthOrderPolynomial as _FifthOrderPolynomial
from libjdrones import OptimalFifthOrderPolynomial as _OptimalFifthOrderPolynomial


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

    traj: _FifthOrderPolynomial

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
            self.solve()

    def solve(self):
        self.traj = _FifthOrderPolynomial(
            self.start_pos,
            self.start_vel,
            self.start_acc,
            self.dest_pos,
            self.dest_vel,
            self.dest_acc,
            self._T,
        )
        self.traj.solve()

    @property
    def T(self):
        return self._T

    @property
    def coeffs(self):
        return self.traj.get_coeffs()

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

        return self.traj.snap(t)

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

        return self.traj.jerk(t)

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

        return self.traj.acceleration(t)

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

        return self.traj.velocity(t)

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

        return self.traj.position(t)


class OptimalFifthOrderPolynomialTrajectory(FifthOrderPolynomialTrajectory):
    traj: _OptimalFifthOrderPolynomial

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
        tol: float = 1e-10,
        N: int = 1000,
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
        return self.traj.get_T()

    def solve(self):
        self.traj = _OptimalFifthOrderPolynomial(
            self.start_pos,
            self.start_vel,
            self.start_acc,
            self.dest_pos,
            self.dest_vel,
            self.dest_acc,
            self._T,
            self.max_acceleration,
            self.tol,
            self.N,
        )
        self.traj.solve()


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
