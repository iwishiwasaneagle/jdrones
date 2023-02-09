#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from jdrones.controllers import PID
from jdrones.controllers import PID_antiwindup
from jdrones.maths import clip_scalar


@pytest.fixture(params=[False])
def angle(request):
    return request.param


@pytest.fixture(params=[1])
def gain(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def pid(request, angle, gain, dt):
    return PID(*request.param, dt=dt, angle=angle, gain=gain)


@pytest.fixture(params=[1])
def windup(request):
    return request.param


@pytest.fixture(params=[(0, 0, 0)])
def pid_antiwindup(request, windup, angle, gain, dt):
    return PID_antiwindup(*request.param, windup=windup, dt=dt, angle=angle, gain=gain)


@pytest.mark.parametrize("pid", [(1, 0, 0), (10, 0, 0)], indirect=True)
@pytest.mark.parametrize("target,actual,exp", [[1, 1, 0], [20, 0, 20]])
def test_pid_only_p(pid, target, actual, gain, exp):
    act = pid(actual, target)
    assert pytest.approx(act) == gain * exp * pid.Kp


@pytest.mark.parametrize("pid", [(0, 1, 0), (0, 10, 0)], indirect=True)
@pytest.mark.parametrize("gain", [0.1, 1], indirect=True)
@pytest.mark.parametrize("dt", [1 / 240, 0.1], indirect=True)
@pytest.mark.parametrize("target,actual,exp", [[1, 1, 0], [20, 0, 20], [0, 1, -1]])
def test_pid_only_i(pid, target, actual, gain, exp, dt):
    act = pid(actual, target)
    assert pytest.approx(act) == gain * exp * pid.Ki * dt


@pytest.mark.parametrize("pid_antiwindup", [(0, 1, 0), (0, 10, 0)], indirect=True)
@pytest.mark.parametrize("gain", [0.1, 1], indirect=True)
@pytest.mark.parametrize("windup", [0.00001, 1], indirect=True)
@pytest.mark.parametrize("dt", [1 / 240, 0.1], indirect=True)
@pytest.mark.parametrize("target,actual,exp", [[1, 1, 0], [20, 0, 20], [0, 1, -1]])
def test_pid_antiwindup_only_i(pid_antiwindup, windup, target, actual, gain, exp, dt):
    act = pid_antiwindup(actual, target)
    assert pid_antiwindup.Integration <= windup
    assert (
        pytest.approx(act)
        == gain * clip_scalar(exp * dt, -windup, windup) * pid_antiwindup.Ki
    )


@pytest.mark.parametrize("pid", [(0, 0, 1), (0, 0, 10)], indirect=True)
@pytest.mark.parametrize("gain", [0.1, 1], indirect=True)
@pytest.mark.parametrize("dt", [1 / 240, 0.1], indirect=True)
@pytest.mark.parametrize("target,actual,exp", [[1, 1, 0], [20, 0, 20], [0, 1, -1]])
def test_pid_only_d(pid, target, actual, gain, exp, dt):
    act = pid(actual, target)
    assert pytest.approx(act) == gain * exp * pid.Kd / dt


@pytest.mark.parametrize("pid", [(1, 0, 0), (10, 0, 0)], indirect=True)
@pytest.mark.parametrize("angle", [True], indirect=True)
@pytest.mark.parametrize(
    "target,actual,exp",
    [
        [np.pi, -np.pi, 0],
        [0, 0, 0],
        [-np.pi / 2, np.pi / 2, np.pi],
        [0, np.pi / 2, -np.pi / 2],
    ],
)
def test_pid_angled_only_p(pid, target, actual, gain, exp):
    act = pid(actual, target)
    assert pytest.approx(act) == gain * exp * pid.Kp
