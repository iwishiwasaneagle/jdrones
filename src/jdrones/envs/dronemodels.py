#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import functools
from importlib.resources import files

import numpy as np
from jdrones.data_models import URDFModel


@functools.cache
def droneplus_mixing_matrix(*, length, k_Q, k_T):
    return np.array(
        [
            [0, -k_T * length, 0, k_T * length],
            [-k_T * length, 0, k_T * length, 0],
            [k_Q, -k_Q, k_Q, -k_Q],
            [k_T, k_T, k_T, k_T],
        ]
    )


DronePlus = URDFModel(
    l=0.1,
    k_T=0.1,
    k_Q=0.05,
    tau_T=0.1,
    tau_Q=0.1,
    drag_coeffs=(0.1, 0.1, 0.1),
    filepath=str(files("jdrones.envs").joinpath("droneplus.urdf")),
    mass=1.4,
    I=(0.1, 0.1, 0.1),
    mixing_matrix=droneplus_mixing_matrix,
    max_vel_ms=1,
)
