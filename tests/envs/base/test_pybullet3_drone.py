#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from itertools import combinations_with_replacement

import numpy as np
import pytest


@pytest.mark.parametrize(
    "equilibrium_prop_rpm",
    [
        1,
    ],
    indirect=True,
)
@pytest.mark.parametrize("drag_coeffs", [(0.1, 0.1, 0.1)], indirect=True)
@pytest.mark.parametrize(
    "velocity,rpy,exp_f",
    [
        [(1, 0, 0), (0, 0, 0), (-0.1, 0, 0)],
        [(1, 0, 0), (0, 0, 0), (-0.1, 0, 0)],
        [(0, 1, 0), (0, 0, 0), (0, -0.1, 0)],
        [(0, 0, 1), (0, 0, 0), (0, 0, -0.1)],
        [(1, 0, 0), (np.pi, 0, 0), (-0.1, 0, 0)],
        [(1, 0, 0), (0, 0, np.pi / 2), (0, -0.1, 0)],
        [(1, 0, 0), (np.pi / 2, 0, np.pi / 2), (0, -0.1, 0)],
        [(1, 0, 0), (np.pi / 2, 0, -np.pi / 2), (0, 0.1, 0)],
        [(1, 0, 0), (np.pi / 2, np.pi / 2, np.pi / 2), (0, 0, 0.1)],
        [(1, 2, 3), (0, 0, 0), (-0.1, -0.4, -0.9)],
    ],
    indirect=["velocity"],
)
def test_calculate_aerodynamic_forces(
    droneenv, drag_coeffs, rpy, state, vec_omega, exp_f
):
    droneenv.state = state
    act_f = droneenv.calculate_aerodynamic_forces(vec_omega)
    assert np.allclose(act_f, exp_f)


@pytest.mark.parametrize(
    "equilibrium_prop_rpm",
    [
        1,
    ],
    indirect=True,
)
@pytest.mark.parametrize("k_Q", [0.1, 1], indirect=True)
@pytest.mark.parametrize(
    "vec_omega,exp_q_z",
    [
        [(0, 0, 0, 0), 0],
        [(1, 1, 1, 1), 0],
        [(100, 100, 100, 100), 0],
        [(2, 1, 2, 1), -6],
        [(100, 1, 50, 400), 147501],
    ],
    indirect=["vec_omega"],
)
def test_calculate_external_torques(droneenv, state, vec_omega, k_Q, exp_q_z):
    droneenv.state = state
    act_q = droneenv.calculate_external_torques(vec_omega)
    assert np.allclose(act_q, [0, 0, exp_q_z * k_Q])


@pytest.mark.parametrize(
    "equilibrium_prop_rpm",
    [
        1,
    ],
    indirect=True,
)
@pytest.mark.parametrize("k_T", [0.5], indirect=True)
@pytest.mark.parametrize(
    "vec_omega,exp_t",
    [
        [(0, 0, 0, 0), (0, 0, 0, 0)],
        [(1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)],
        [(2, 1, 2, 1), (2, 0.5, 2, 0.5)],
        [(100, 1, 50, 400), (5000, 0.5, 1250, 80000)],
    ],
    indirect=["vec_omega"],
)
def test_calculate_propulsive_forces(droneenv, state, vec_omega, k_T, exp_t):
    droneenv.state = state
    act_t = droneenv.calculate_propulsive_forces(vec_omega)
    assert np.allclose(act_t, exp_t)


# @pytest.mark.integration
@pytest.mark.parametrize(
    "rpy",
    [
        (0, 0, 0),
        (np.pi / 2, 0, 0),
        (0, np.pi / 2, 0),
        (0, 0, np.pi / 2),
        (np.pi / 2, np.pi / 2, np.pi / 2),
    ],
    indirect=True,
)
@pytest.mark.parametrize("vec_omega", [np.zeros(4)], indirect=True)
def test_zero_input(vec_omega, rpy, droneenv):
    """
    Expect it to drop like a stone
    """
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


# @pytest.mark.integration
@pytest.mark.parametrize("vec_omega", [np.zeros(4), np.ones(4) * 0.01], indirect=True)
def test_low_input(vec_omega, droneenv):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


# @pytest.mark.integration
@pytest.mark.parametrize(
    "vec_omega", [np.ones(4) * 10000, np.ones(4) * 1e10], indirect=True
)
def test_large_input(vec_omega, droneenv):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, 1))


@pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(1, 0.4, 1, 0.5), [1, 0, -1]],
        [(1, 0.5, 1, 0.4), [-1, 0, -1]],
        [(1, 0.9, 1, 1.1), [1, 0, 1]],
        [(1, 1.1, 1, 0.9), [-1, 0, 1]],
        [(1, 2, 1, 0), [-1, 0, 1]],
        [(1, 0, 1, 2), [1, 0, 1]],
    ],
    indirect=["vec_omega"],
)
def test_roll_input(vec_omega, droneenv, exp):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


# @pytest.mark.integration
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
        [(2, 1, 0, 1), [0, 1, -1]],
        [(0, 1, 2, 1), [0, -1, -1]],
    ],
    indirect=["vec_omega"],
)
def test_pitch_input(vec_omega, droneenv, exp):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


# @pytest.mark.integration
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
def test_yaw_input(vec_omega, equilibrium_prop_rpm, droneenv, exp):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.parametrize(
    "rpy,exp",
    [
        [(0, 0, 0), [0, 0, 0]],
        [(0.1, 0, 0), [0, -1, -1]],
        [(-0.1, 0, 0), [0, 1, -1]],
        [(0, 0.1, 0), [1, 0, -1]],
        [(0, -0.1, 0), [-1, 0, -1]],
        [(0, 0, 0.1), [0, 0, 0]],
        [(0, 0, -0.1), [0, 0, 0]],
    ],
    indirect=["rpy"],
)
def test_vel_from_rot(vec_omega, rpy, droneenv, exp):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel.round(16)), exp)


@pytest.mark.parametrize(
    "velocity",
    combinations_with_replacement((1, 0, -1), 3),
    indirect=True,
)
def test_pos_from_vel(vec_omega, droneenv, pos, velocity):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.pos), velocity - pos)


@pytest.mark.parametrize(
    "angular_velocity",
    combinations_with_replacement((1, 0, -1), 3),
    indirect=True,
)
def test_rpy_from_ang_vel(vec_omega, droneenv, angular_velocity):
    droneenv.reset()
    obs, *_ = droneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.rpy), angular_velocity)


@pytest.mark.parametrize(
    "vec_omega,k_Q,ang_vel_sign",
    [
        [
            (1, 0, 0, 0),
            0,
            (0, -1, 0),
        ],
        [(0, 0, 1, 0), 0, (0, 1, 0)],
        [(0, 1, 0, 0), 0, (1, 0, 0)],
        [(0, 0, 0, 1), 0, (-1, 0, 0)],
        [(1, 0.9, 1, 0.9), 0.1, (0, 0, -1)],
        [(0.9, 1, 0.9, 1), 0.1, (0, 0, 1)],
    ],
    indirect=["vec_omega", "k_Q"],
)
def test_droneenv_correct_input_to_rot(seed, droneenv, action, k_Q, ang_vel_sign):
    """
    Step input over a short time will give a good insight if the drone is behaving
    as expected
    """
    droneenv.reset(seed=seed)
    for _ in range(5):
        obs, *_ = droneenv.step(action * 100)
    # Drone landed within 10cm of where we expected it
    assert np.allclose(np.sign(obs.ang_vel), ang_vel_sign)
