#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from importlib.resources import files

import numpy as np
from jdrones.data_models import URDFModel
from jdrones.types import MAT4X4


def droneplus_mixing_matrix(*, length: float, k_Q: float, k_T: float) -> MAT4X4:
    """
    .. math::
        \\vec M
        = \\begin{bmatrix}
            \\vec \\Gamma \\\\
            T
        \\end{bmatrix}
        =
        \\begin{bmatrix}
            \\Gamma_\\phi\\\\\\Gamma_\\theta\\\\\\Gamma_\\psi\\\\T
        \\end{bmatrix}
        = \\begin{bmatrix}
        0& -l k_T& 0& l k_T \\\\
        -l k_T& 0& l k_T& 0\\\\
        k_Q&-k_Q& k_Q& -k_Q \\\\
        k_T & k_T & k_T & k_T
        \\end{bmatrix}
        \\begin{bmatrix}
            P_1\\\\P_2\\\\P_3\\\\P_4
        \\end{bmatrix}

    Parameters
    ----------
    length : float
    k_Q : float
    k_T : float

    Returns
    -------
    jdrones.types.MAT4X4
        Mixing matrix
    """

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
