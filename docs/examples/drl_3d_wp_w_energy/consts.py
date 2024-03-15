#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import pathlib

import numpy as np

TOTAL_TIMESTEP = int(5e6)
DT = 1 / 100
N_ENVS = 1
N_EVAL = 100
POS_LIM = (-10, 10)
TGT_SUB_LIM = (-5, 5)
VEL_LIM = (-100, 100)
ROLL_LIM = PITCH_LIM = (-1, 1)
YAW_LIM = (0, 2 * np.pi)
ANG_VEL_LIM = (-10, 10)
PROP_OMEGA_LIM = (0, 25)

LOG_PATH = pathlib.Path("drl_3d_wp_w_energy/logs")
OPTUNA_PATH = LOG_PATH / "optuna"
TENSORBOARD_PATH = LOG_PATH / "tensorboard"

OPTUNA_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_PATH.mkdir(parents=True, exist_ok=True)
