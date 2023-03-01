#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import scipy as scipy
from jdrones.data_models import State
from jdrones.maths import clip_scalar


class Controller:
    @staticmethod
    def error(measured, setpoint):
        return setpoint - measured

    def __call__(self, *, measured, setpoint):
        return self.error(measured, setpoint)

    def reset(self):
        raise NotImplementedError


class AngleController(Controller):
    def __init__(self, angle=False):
        self.angle = angle

    @staticmethod
    def angle_error(measured, setpoint):
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
        A
        B
        Q
        R

        Returns
        -------
        K
        """

        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

        return np.asarray(K)

    def reset(self):
        self.e = State()

    def __call__(self, *, measured: State, setpoint: State) -> float:
        self.e = super().__call__(measured=measured, setpoint=setpoint)
        return self.K @ self.e.to_x()
