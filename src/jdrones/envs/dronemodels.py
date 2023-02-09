#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from importlib.resources import files

import numpy as np
from jdrones.types import URDFModel


def droneplus_mixing_matrix(length, k_f, k_t):
    h = k_f
    i = k_t
    j = length * h

    return np.array(
        [
            [0, 1 / (2 * j), i / 4, 1 / (4 * h)],
            [1 / (2 * j), 0, -i / 4, 1 / (4 * h)],
            [0, -1 / (2 * j), i / 4, 1 / (4 * h)],
            [-1 / (2 * j), 0, -i / 4, 1 / (4 * h)],
        ]
    )


DronePlus = URDFModel(
    l=0.1,
    k_T=0.1,
    k_Q=0.05,
    tau_T=0.1,
    tau_Q=0.1,
    drag_coeffs=(9.1785e-7, 9.1785e-7, 10.311e-7),
    filepath=str(files("jdrones.envs").joinpath("droneplus.urdf")),
    mass=1.4,
    I=(0.1, 0.1, 0.1),
    mixing_matrix=droneplus_mixing_matrix,
    max_vel_ms=1,
)
