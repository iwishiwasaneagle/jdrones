#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pytest


def run(T, dt, env):
    trunc = False
    t, c = 0, 0
    while not trunc and t <= T:
        setpoint = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(setpoint)
        t += dt * len(obs)
        c += 1
    assert not trunc, f"Failed after {t} simulation seconds and {c} setpoints"


AS_PARAM = pytest.mark.parametrize(
    "position_drone_action_space",
    [[(-100, 100), (-100, 100), (-100, 100)]],
    indirect=True,
)
T_PARAM = pytest.mark.parametrize("T", [1, 10, 100])


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
