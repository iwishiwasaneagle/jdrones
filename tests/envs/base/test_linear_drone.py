#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from itertools import combinations_with_replacement

import numpy as np
import pytest


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0, 0, 0.1)], indirect=True)
@pytest.mark.parametrize("vec_omega", [np.zeros(4)], indirect=True)
def test_zero_input(vec_omega, lineardroneenv):
    """
    Expect it to drop like a stone
    """
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


@pytest.mark.parametrize("vec_omega", [np.zeros(4), np.ones(4) * 0.01], indirect=True)
def test_low_input(vec_omega, lineardroneenv):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


@pytest.mark.parametrize(
    "vec_omega", [np.ones(4) * 10000, np.ones(4) * 1e10], indirect=True
)
def test_large_input(vec_omega, lineardroneenv):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, 1))


@pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(1, 0.4, 1, 0.5), [1, 0, -1]],
        [(1, 0.5, 1, 0.4), [-1, 0, -1]],
        [(1, 0.9, 1, 1.1), [1, 0, 1]],
        [(1, 1.1, 1, 0.9), [-1, 0, 1]],
        [(1, 2, 1, 0), [-1, 0, 0]],
        [(1, 0, 1, 2), [1, 0, 0]],
    ],
    indirect=["vec_omega"],
)
def test_roll_input(vec_omega, lineardroneenv, exp):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(0.5, 1, 0.4, 1), [0, 1, 1]],
        [(0.4, 1, 0.5, 1), [0, -1, 1]],
        [(1.1, 1, 0.9, 1), [0, 1, -1]],
        [
            (0.9, 1, 1.1, 1),
            [0, -1, -1],
        ],
        [(2, 1, 0, 1), [0, 1, 0]],
        [(0, 1, 2, 1), [0, -1, 0]],
        [(3, 1, 0, 1), [0, 1, -1]],
        [(0, 1, 3, 1), [0, -1, -1]],
    ],
    indirect=["vec_omega"],
)
def test_pitch_input(vec_omega, lineardroneenv, exp):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(1, 1.1, 1, 1.1), [0, 0, 1]],
        [(1.1, 1, 1.1, 1), [0, 0, -1]],
        [(1, 2, 1, 2), [0, 0, 1]],
        [(2, 1, 2, 1), [0, 0, -1]],
    ],
    indirect=["vec_omega"],
)
def test_yaw_input(vec_omega, equilibrium_prop_rpm, lineardroneenv, exp):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.parametrize(
    "rpy,exp",
    [
        [(0, 0, 0), [0, 0, -1]],
        [(0.1, 0, 0), [0, 1, -1]],
        [(-0.1, 0, 0), [0, -1, -1]],
        [(0, 0.1, 0), [1, 0, -1]],
        [(0, -0.1, 0), [-1, 0, -1]],
        [(0, 0, 0.1), [0, 0, -1]],
        [(0, 0, -0.1), [0, 0, -1]],
    ],
    indirect=["rpy"],
)
def test_vel_from_rot(vec_omega, lineardroneenv, exp):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel.round(16)), exp)


@pytest.mark.parametrize(
    "pos",
    [
        (0, 0, 0),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "velocity",
    combinations_with_replacement((0.01, 0, -0.01), 3),
    indirect=True,
)
def test_pos_from_vel(vec_omega, lineardroneenv, velocity):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.pos), np.sign(velocity))


@pytest.mark.parametrize(
    "angular_velocity",
    combinations_with_replacement((0.1, 0, -0.1), 3),
    indirect=True,
)
def test_rpy_from_ang_vel(vec_omega, lineardroneenv, angular_velocity):
    lineardroneenv.reset()
    obs, *_ = lineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.rpy), np.sign(angular_velocity))
