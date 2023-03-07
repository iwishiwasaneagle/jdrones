#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from envs.conftest import INPUT_TO_ROT
from envs.conftest import LARGE_INPUT
from envs.conftest import LOW_INPUT
from envs.conftest import PITCH_INPUT
from envs.conftest import POSITION_FROM_VELOCITY_1
from envs.conftest import POSITION_FROM_VELOCITY_2
from envs.conftest import ROLL_INPUT
from envs.conftest import RPY_FROM_ANG_VEL
from envs.conftest import VELOCITY_FROM_ROTATION
from envs.conftest import YAW_INPUT


@pytest.mark.integration
@pytest.mark.parametrize("rpy", [(0, 0, 0), (0, 0, 0.1)], indirect=True)
@pytest.mark.parametrize("vec_omega", [np.zeros(4)], indirect=True)
def test_zero_input(vec_omega, quadlineardroneenv):
    """
    Expect it to drop like a stone
    """
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


@pytest.mark.integration
@LOW_INPUT
def test_low_input(vec_omega, quadlineardroneenv):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.vel), (0, 0, -1))


@pytest.mark.integration
@LARGE_INPUT
def test_large_input(vec_omega, quadlineardroneenv):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.vel), (0, 0, 1))


@pytest.mark.integration
@ROLL_INPUT
def test_roll_input(vec_omega, quadlineardroneenv, exp):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.integration
@PITCH_INPUT
def test_pitch_input(vec_omega, quadlineardroneenv, exp):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.integration
@YAW_INPUT
def test_yaw_input(vec_omega, equilibrium_prop_rpm, quadlineardroneenv, exp):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.ang_vel), exp)


@pytest.mark.integration
@VELOCITY_FROM_ROTATION
def test_vel_from_rot(vec_omega, quadlineardroneenv, exp):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(0.99 * vec_omega**2)
    assert np.allclose(np.sign(obs.vel.round(16)), exp)


@pytest.mark.integration
@POSITION_FROM_VELOCITY_1
@POSITION_FROM_VELOCITY_2
def test_pos_from_vel(vec_omega, quadlineardroneenv, velocity):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega**2)
    assert np.allclose(np.sign(obs.pos), np.sign(velocity))


@pytest.mark.integration
@RPY_FROM_ANG_VEL
def test_rpy_from_ang_vel(vec_omega, quadlineardroneenv, angular_velocity):
    quadlineardroneenv.reset()
    obs, *_ = quadlineardroneenv.step(vec_omega)
    assert np.allclose(np.sign(obs.rpy), np.sign(angular_velocity))


@pytest.mark.integration
@INPUT_TO_ROT
def test_input_to_rot(seed, quadlineardroneenv, action, k_Q, ang_vel_sign):
    """
    Step input over a short time will give a good insight if the drone is behaving
    as expected
    """
    quadlineardroneenv.reset(seed=seed)
    for _ in range(5):
        obs, *_ = quadlineardroneenv.step(action * 100)
    # Drone landed within 10cm of where we expected it
    assert np.allclose(np.sign(obs.ang_vel), ang_vel_sign)
