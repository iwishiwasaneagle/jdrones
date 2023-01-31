import numpy as np
from jdrones.types import URDFModel


def droneplus_mixing_matrix(l, k_f, k_t):
    h = k_f
    i = k_t
    j = l * h

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
    filepath="src/jdrones/envs/droneplus.urdf",
    mass=1.4,
    I=(0.1, 0.1, 0.1),
    mixing_matrix=droneplus_mixing_matrix,
    max_vel_ms=1,
)