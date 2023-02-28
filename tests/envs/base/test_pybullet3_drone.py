#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from envs.base.conftest import INPUT_TO_ROT
from envs.base.conftest import LARGE_INPUT
from envs.base.conftest import LOW_INPUT
from envs.base.conftest import PITCH_INPUT
from envs.base.conftest import POSITION_FROM_VELOCITY_1_PB
from envs.base.conftest import POSITION_FROM_VELOCITY_2
from envs.base.conftest import ROLL_INPUT
from envs.base.conftest import RPY_FROM_ANG_VEL
from envs.base.conftest import VELOCITY_FROM_ROTATION
from envs.base.conftest import YAW_INPUT


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
    pbdroneenv, drag_coeffs, rpy, state, vec_omega, exp_f
):
    pbdroneenv.state = state
    act_f = pbdroneenv.calculate_aerodynamic_forces(vec_omega)
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
def test_calculate_external_torques(pbdroneenv, state, vec_omega, k_Q, exp_q_z):
    pbdroneenv.state = state
    act_q = pbdroneenv.calculate_external_torques(vec_omega)
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
def test_calculate_propulsive_forces(pbdroneenv, state, vec_omega, k_T, exp_t):
    pbdroneenv.state = state
    act_t = pbdroneenv.calculate_propulsive_forces(vec_omega)
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
def test_zero_input(vec_omega, rpy, pbdroneenv):
    """
    Expect it to drop like a stone
    """
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


# @pytest.mark.integration
@LOW_INPUT
def test_low_input(vec_omega, pbdroneenv):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


# @pytest.mark.integration
@LARGE_INPUT
def test_large_input(vec_omega, pbdroneenv):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.vel), (0, 0, 1))


@ROLL_INPUT
def test_roll_input(rhr_to_lhr, vec_omega, pbdroneenv, exp):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), exp * rhr_to_lhr)


# @pytest.mark.integration
@PITCH_INPUT
def test_pitch_input(rhr_to_lhr, vec_omega, pbdroneenv, exp):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), np.sign(rhr_to_lhr * exp))


# @pytest.mark.integration
@YAW_INPUT
def test_yaw_input(rhr_to_lhr, vec_omega, equilibrium_prop_rpm, pbdroneenv, exp):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.ang_vel), rhr_to_lhr * exp)


@VELOCITY_FROM_ROTATION
def test_vel_from_rot(vec_omega, rhr_to_lhr, rpy, pbdroneenv, exp):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(0.99 * vec_omega)
    assert np.allclose(np.sign(obs.vel.round(16)), rhr_to_lhr * exp)


@POSITION_FROM_VELOCITY_1_PB
@POSITION_FROM_VELOCITY_2
def test_pos_from_vel(rhr_to_lhr, pos, vec_omega, pbdroneenv, velocity):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.pos), rhr_to_lhr * np.sign(velocity - pos))


@RPY_FROM_ANG_VEL
def test_rpy_from_ang_vel(vec_omega, rhr_to_lhr, pbdroneenv, angular_velocity):
    pbdroneenv.reset()
    obs, *_ = pbdroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.rpy), rhr_to_lhr * angular_velocity)


@INPUT_TO_ROT
def test_correct_input_to_rot(seed, rhr_to_lhr, pbdroneenv, action, k_Q, ang_vel_sign):
    """
    Step input over a short time will give a good insight if the drone is behaving
    as expected
    """
    pbdroneenv.reset(seed=seed)
    for _ in range(5):
        obs, *_ = pbdroneenv.step(action * 100)
    # Drone landed within 10cm of where we expected it
    assert np.allclose(np.sign(obs.ang_vel), rhr_to_lhr * ang_vel_sign)
