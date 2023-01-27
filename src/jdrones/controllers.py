import functools
from typing import Callable

import numpy as np
from jdrones.maths import clip
from jdrones.maths import clip_scalar


class PID:
    def __init__(self, Kp, Ki, Kd, dt, *, angle=False, gain=1):
        self.angle = angle
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.gain = gain

        # initialize stored data
        self.reset()

    def reset(self):
        self.e = 0
        self.I = 0

    def _calc_I(self, e):
        self.I += self.Ki * e * self.dt
        return self.I

    def _calc_P(self, e):
        return self.Kp * e

    def _calc_D(self, e):
        D = self.Kd * (e - self.e) / self.dt
        self.e = e
        return D

    @staticmethod
    def error(measured, setpoint):
        return setpoint - measured

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

    def __call__(self, measured, setpoint):
        # PID calculations
        if self.angle:
            e = self.angle_error(measured, setpoint)
        else:
            e = self.error(measured, setpoint)

        P = self._calc_P(e)
        I = self._calc_I(e)
        D = self._calc_D(e)

        return self.gain * (P + I + D)


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
        self.I += clip(e * self.dt, -self.windup, self.windup)
        return self.I * self.Ki
