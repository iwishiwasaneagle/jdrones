#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Any

import nptyping as npt

VEC3 = npt.NDArray[npt.Shape["1, 3"], npt.Double]
VEC4 = npt.NDArray[npt.Shape["1, 4"], npt.Double]
MAT3X3 = npt.NDArray[npt.Shape["3, 3"], npt.Double]
MAT4X3 = npt.NDArray[npt.Shape["4, 4"], npt.Double]
Action = npt.NDArray[Any, npt.Double]
LinearXAction = npt.NDArray[npt.Shape["12, 1"], npt.Double]
Length3Action = VEC3
Length4Action = VEC4
PropellerAction = Length4Action
AttitudeAltitudeAction = Length4Action
VelHeadAltAction = Length4Action
PositionAction = Length3Action
PositionVelAction = Length4Action
