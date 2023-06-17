#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import numpy.typing as npt
import scipy as scipy
from jdrones.data_models import State
from jdrones.maths import clip_scalar


class Controller:
    """
    Base controller class
    """

    @staticmethod
    def error(measured: float, setpoint: float) -> float:
        """
        Calculate the error between measured and setpoint

        .. math::

            e = u - \\hat x

        Parameters
        ----------
        measured : float
        setpoint : float

        Returns
        -------
        float
        """
        return setpoint - measured

    def __call__(self, *, measured, setpoint):
        return self.error(measured, setpoint)

    def reset(self):
        """
        Resets the controller. Useful for when there's calculations dependent on the
        previous timestep.
        """
        raise NotImplementedError


class AngleController(Controller):
    """
    Special controller type when the inputs are angles that wrap around
    :math:`2\\pi`.
    """

    angle: bool
    """Flag to determine which method to use for calculating the error"""

    def __init__(self, angle=False):
        self.angle = angle

    @staticmethod
    def angle_error(measured: float, setpoint: float) -> float:
        """
        Calculate the angle error through

        .. math::
            \\begin{align}
                \\hat x_{\\mathit{wrapped}} &=
                \\begin{cases}
                    \\hat x - 2\\pi&, \\hat x > 0 \\\\
                    \\hat x + 2\\pi&, \\mathit{else}
                \\end{cases}\\\\
                e &= \\begin{cases}
                    u - \\hat x&, |u - \\hat x|<|u-\\hat x_{\\mathit{wrapped}}|\\\\
                    u-\\hat x_{\\mathit{wrapped}}&,\\mathit{else}
                \\end{cases}
            \\end{align}

        Parameters
        ----------
        measured : float
        setpoint : float

        Returns
        -------
        float
            Error
        """
        measured_wrapped = measured
        if measured > 0:
            measured_wrapped -= 2 * np.pi
        else:
            measured_wrapped += 2 * np.pi

        if abs(setpoint - measured) < abs(setpoint - measured_wrapped):
            return setpoint - measured
        return setpoint - measured_wrapped

    def __call__(self, *, measured, setpoint):
        if self.angle:
            return self.angle_error(measured, setpoint)
        return super().__call__(measured=measured, setpoint=setpoint)


class PID(AngleController):
    """
    Simple PID controller implementation.

    .. math::
        u(t) = K\\left[K_p e(t) + K_i \\int^t_0 e(t) dt + K_d \\frac{de(t)}{dt}\\right]


    >>> pid = PID(1,2,3,dt=0.1)
    >>> pid(measured=0,setpoint=1)
    31.2
    """

    Kp: float
    """Proportional gain"""
    Ki: float
    """Integral gain"""
    Kd: float
    """Derivative gain"""
    gain: float
    """Scaling constant :math:`K`"""
    dt: float
    """Difference in time between calculations"""

    def __init__(self, Kp, Ki, Kd, dt, angle=False, gain=1):
        super().__init__(angle=angle)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.gain = gain

        # initialize stored data
        self.reset()

    def reset(self):
        """
        Resets the error and integration to 0
        """
        self.e = 0
        self.Integration = 0

    def _calc_I(self, e):
        self.Integration += self.Ki * e * self.dt
        return self.Integration

    def _calc_P(self, e):
        return self.Kp * e

    def _calc_D(self, e):
        D = self.Kd * (e - self.e) / self.dt
        self.e = e
        return D

    def __call__(self, *, measured, setpoint):
        # PID calculations
        e = super().__call__(measured=measured, setpoint=setpoint)
        P = self._calc_P(e)
        Integration = self._calc_I(e)
        D = self._calc_D(e)

        return self.gain * (P + Integration + D)


class PID_antiwindup(PID):
    def __init__(
        self,
        *args,
        windup=5,
        **kwargs,
    ):
        self.windup = windup
        super().__init__(*args, **kwargs)

    def _calc_I(self, e):
        self.Integration += clip_scalar(e * self.dt, -self.windup, self.windup)
        return self.Integration * self.Ki


class LQR(Controller):
    """
    Simple Linear-Quadratic Regulator class that handles solving and evaluation of the
    controller.
    """

    A: npt.NDArray
    """The system matrix"""
    B: npt.NDArray
    """The control matrix"""
    Q: npt.NDArray
    """The state deviation cost matrix"""
    R: npt.NDArray
    """The control deviation cost matrix"""

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.K = self.solve(A, B, Q, R)

    @staticmethod
    def solve(A, B, Q, R):
        """Solve the continuous time lqr controller.

        .. math::

            \\frac{dx}{dt} = A x + B u

            J = \\int x^T Q x + u^T R u

        .. seealso::
            - http://www.mwm.im/lqr-controllers-with-python/

        Parameters
        ----------
        A : numpy.ndarray
            System matrix
        B : numpy.ndarray
            System control matrix
        Q : numpy.ndarray
        R : numpy.ndarray

        Returns
        -------
        K : numpy.ndarray
            LQR K matrix
        """

        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # compute the LQR gain
        K = scipy.linalg.inv(R) @ (B.T @ X)

        return K

    def reset(self):
        """
        Resets the error to :math:`\\vec e = \\vec 0_{20,1}`
        """
        self.e = State()

    def __call__(self, *, measured: State, setpoint: State) -> float:
        self.e = super().__call__(measured=measured, setpoint=setpoint)
        return self.K @ self.e.to_x()
