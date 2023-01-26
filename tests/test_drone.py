import numpy as np
import pytest


@pytest.mark.parametrize("drag_coeffs", [(0.1, 0.1, 0.1)], indirect=True)
@pytest.mark.parametrize(
    "velocity,rpy,exp_f",
    [
        [(1, 0, 0), (0, 0, 0), (-0.1, 0, 0)],
        [(0, 1, 0), (0, 0, 0), (0, -0.1, 0)],
        [(0, 0, 1), (0, 0, 0), (0, 0, -0.1)],
        [(1, 0, 0), (np.pi, 0, 0), (-0.1, 0, 0)],
        [(1, 0, 0), (0, 0, np.pi / 2), (0, -0.1, 0)],
        [(1, 0, 0), (np.pi / 2, 0, np.pi / 2), (0, -0.1, 0)],
        [(1, 0, 0), (np.pi / 2, 0, -np.pi / 2), (0, 0.1, 0)],
        [(1, 0, 0), (np.pi / 2, np.pi / 2, np.pi / 2), (0, 0, 0.1)],
        [(1, 2, 3), (0, 0, 0), (-0.1, -0.2, -0.3)],
        [(-1, 2, -3), (0, 0, 0), (0.1, -0.2, 0.3)],
    ],
    indirect=["velocity"],
)
def test_calculate_aerodynamic_forces(droneenv, state, action, exp_f):
    droneenv.state = state
    act_f = droneenv.calculate_aerodynamic_forces(action)
    assert np.allclose(act_f, exp_f)


@pytest.mark.parametrize("k_Q", [0.1, 1], indirect=True)
@pytest.mark.parametrize(
    "vec_omega,exp_q_z",
    [
        [(0, 0, 0, 0), 0],
        [(1, 1, 1, 1), 0],
        [(100, 100, 100, 100), 0],
        [(2, 1, 2, 1), -2],
        [(100, 1, 50, 400), 251],
    ],
    indirect=["vec_omega"],
)
def test_calculate_external_torques(droneenv, state, action, k_Q, exp_q_z):
    droneenv.state = state
    act_q = droneenv.calculate_external_torques(action)
    assert np.allclose(act_q, [0, 0, exp_q_z * k_Q])


@pytest.mark.parametrize("k_T", [0.5], indirect=True)
@pytest.mark.parametrize(
    "vec_omega,exp_t",
    [
        [(0, 0, 0, 0), (0, 0, 0, 0)],
        [(1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)],
        [(2, 1, 2, 1), (1, 0.5, 1, 0.5)],
        [(100, 1, 50, 400), (50, 0.5, 25, 200)],
    ],
    indirect=["vec_omega"],
)
def test_calculate_propulsive_forces(droneenv, state, action, k_T, exp_t):
    droneenv.state = state
    act_t = droneenv.calculate_propulsive_forces(action)
    assert np.allclose(act_t, exp_t)


@pytest.mark.slow_integration_test
@pytest.mark.parametrize(
    "pos,exp_final",
    [
        [(0, 0, 0.1), (0, 0, 0.1)],
        [(2, 1, 1), (2, 1, 0.1)],
        [(2, 2, 0.9), (2, 2, 0.1)],
        [(1, 2, 0.5), (1, 2, 0.1)],
    ],
    indirect=["pos"],
)
def test_droneenv_zero_inp(seed, droneenv, action, exp_final):
    droneenv.reset(seed=seed)

    for _ in range(500):
        obs, *_ = droneenv.step(action)

    # Drone landed within 10cm of where we expected it
    assert np.linalg.norm(obs.pos - exp_final) < 0.1


@pytest.mark.integration_test
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
def test_droneenv_correct_input_to_rot(seed, droneenv, action, ang_vel_sign):
    """
    Step input over a short time will give a good insight if the drone is behaving
    as expected
    """
    droneenv.reset(seed=seed)
    for _ in range(5):
        obs, *_ = droneenv.step(action * 100)
    # Drone landed within 10cm of where we expected it
    assert np.allclose(np.sign(obs.ang_vel), ang_vel_sign)
