#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np

TOTAL_TIMESTEP = int(5e6)
N_EVAL = 1000
POS_LIM = (-10, 10)
TGT_SUB_LIM = (-5, 5)
VEL_LIM = (-100, 100)
RPY_LIM = (-np.pi, np.pi)
ANG_VEL_LIM = (-100, 100)
PROP_OMEGA_LIM = (0, 50)
