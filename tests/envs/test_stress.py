#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pytest


@pytest.mark.stress
@pytest.mark.parametrize(
    "position_drone_action_space", [[(-2, 2), (-2, 2), (-2, 2)]], indirect=True
)
@pytest.mark.parametrize("T", [1, 10, 100])
def test_long_lqr_position(T, dt, lqrpositiondroneenv):
    trunc = False
    t, c = 0, 0
    while not trunc and t <= T:
        setpoint = lqrpositiondroneenv.action_space.sample()
        obs, _, term, trunc, _ = lqrpositiondroneenv.step(setpoint)
        t += dt * len(obs)
        c += 1
    assert not trunc, f"Failed after {t} simulation seconds and {c} setpoints"


@pytest.mark.stress
@pytest.mark.parametrize(
    "position_drone_action_space",
    [[(-100, 100), (-100, 100), (-100, 100)]],
    indirect=True,
)
@pytest.mark.parametrize("T", [1, 10, 100, 1000])
def test_long_poly_position(T, dt, polypositiondroneenv):
    trunc = False
    t, c = 0, 0
    while not trunc and t <= T:
        setpoint = polypositiondroneenv.action_space.sample()
        obs, _, term, trunc, _ = polypositiondroneenv.step(setpoint)
        t += dt * len(obs)
        c += 1
    assert not trunc, f"Failed after {t} simulation seconds and {c} setpoints"
