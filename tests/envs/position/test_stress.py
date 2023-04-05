#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pytest


def run(T, dt, env):
    trunc = False
    t, c = 0, 0
    env.reset()
    while not trunc and t <= T:
        setpoint = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(setpoint)
        t += dt * len(obs)
        c += 1
    assert not trunc, f"Failed after {t} simulation seconds and {c} setpoints"


LOW = (-100, -100, -100)
HIGH = (100, 100, 100)

AS_PARAM = pytest.mark.parametrize(
    "position_drone_action_space",
    [[LOW, HIGH]],
    indirect=True,
)
AS_PARAM_LA = pytest.mark.parametrize(
    "position_drone_action_space",
    [[(LOW, LOW), (HIGH, HIGH)]],
    indirect=True,
)
T_PARAM = pytest.mark.parametrize("T", [2, 10, 100])


@pytest.mark.slow_integration
@AS_PARAM
@T_PARAM
def test_long_lqr_position(T, dt, firstorderploypositiondroneenv):
    run(T, dt, firstorderploypositiondroneenv)


@pytest.mark.slow_integration
@AS_PARAM
@T_PARAM
def test_long_poly_position(T, dt, fifthorderpolypositiondroneenv):
    run(T, dt, fifthorderpolypositiondroneenv)


@pytest.mark.slow_integration
@AS_PARAM_LA
@T_PARAM
def test_long_lookahead_position(T, dt, fifthorderpolypositionlookaheaddroneenv):
    run(T, dt, fifthorderpolypositionlookaheaddroneenv)
