#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest

VELOCITY_FROM_ROTATION = pytest.mark.parametrize(
    "rpy,exp",
    [
        [(0, 0, 0), [0, 0, -1]],
        [(0, 0, 0.1), [0, 0, -1]],
        [(0, 0, -0.1), [0, 0, -1]],
        [(0.1, 0, 0), [0, -1, -1]],
        [(-0.1, 0, 0), [0, 1, -1]],
        [(0, 0.1, 0), [1, 0, -1]],
        [(0, -0.1, 0), [-1, 0, -1]],
    ],
    indirect=["rpy"],
)
POSITION_FROM_VELOCITY_1 = pytest.mark.parametrize(
    "pos",
    [
        np.array((0, 0, 0)),
    ],
    indirect=True,
)
POSITION_FROM_VELOCITY_2 = pytest.mark.parametrize(
    "velocity",
    [
        (0.01, 0.01, 0.01),
        (0.01, 0.01, 0),
        (0.01, 0.01, -0.01),
        (0.01, 0, 0),
        (0.01, 0, -0.01),
        (0.01, -0.01, -0.01),
        (0, 0, 0),
        (0, 0, -0.01),
        (0, -0.01, -0.01),
        (-0.01, -0.01, -0.01),
    ],
    indirect=True,
)
RPY_FROM_ANG_VEL = pytest.mark.parametrize(
    "angular_velocity",
    [
        (1, 1, 1),
        (1, 1, 0),
        (1, 1, -1),
        (1, 0, 0),
        (1, 0, -1),
        (1, -1, -1),
        (0, 0, 0),
        (0, 0, -1),
        (0, -1, -1),
        (-1, -1, -1),
    ],
    indirect=True,
)
LOW_INPUT = pytest.mark.parametrize(
    "vec_omega", [np.zeros(4), np.ones(4) * 0.01], indirect=True
)
LARGE_INPUT = pytest.mark.parametrize(
    "vec_omega", [np.ones(4) * 10000, np.ones(4) * 1e10], indirect=True
)

ROLL_INPUT = pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(1, 0.4, 1, 0.5), [1, 0, 1]],
        [(1, 0.5, 1, 0.4), [-1, 0, 1]],
        [(1, 0.9, 1, 1.1), [1, 0, -1]],
        [(1, 1.1, 1, 0.9), [-1, 0, -1]],
        [(1, 2, 1, 0), [-1, 0, -1]],
        [(1, 0, 1, 2), [1, 0, -1]],
    ],
    indirect=["vec_omega"],
)
PITCH_INPUT = pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(0.5, 1, 0.4, 1), [0, -1, -1]],
        [(0.4, 1, 0.5, 1), [0, 1, -1]],
        [(1.1, 1, 0.9, 1), [0, -1, 1]],
        [
            (0.9, 1, 1.1, 1),
            [0, 1, 1],
        ],
        [(2, 1, 0, 1), [0, -1, 1]],
        [(0, 1, 2, 1), [0, 1, 1]],
    ],
    indirect=["vec_omega"],
)
YAW_INPUT = pytest.mark.parametrize(
    "vec_omega,exp",
    [
        [(1, 1, 1, 1), [0, 0, 0]],
        [(1, 1.1, 1, 1.1), [0, 0, -1]],
        [(1.1, 1, 1.1, 1), [0, 0, 1]],
        [(1, 2, 1, 2), [0, 0, -1]],
        [(2, 1, 2, 1), [0, 0, 1]],
    ],
    indirect=["vec_omega"],
)

INPUT_TO_ROT = pytest.mark.parametrize(
    "vec_omega,ang_vel_sign",
    [
        [(1, 0.9, 1, 0.9), (0, 0, 1)],
        [(0.9, 1, 0.9, 1), (0, 0, -1)],
    ],
    indirect=["vec_omega"],
)
