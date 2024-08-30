#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Optional
from typing import Tuple

import numpy as np
from gymnasium import spaces
from jdrones.data_models import State
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.types import DType
from jdrones.types import PropellerAction
from libjdrones import LinearDynamicModelDroneEnv as _LinearDynamicModelDroneEnv


class LinearDynamicModelDroneEnv(BaseDroneEnv):
    """
    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("LinearDynamicModelDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<LinearDynamicModelDroneEnv<LinearDynamicModelDroneEnv-v0>>>>
    """

    _env: _LinearDynamicModelDroneEnv

    def __init__(
        self,
        initial_state: State = None,
        dt: float = 1 / 240,
    ):
        self._env = _LinearDynamicModelDroneEnv(dt)

        super().__init__(initial_state, dt)

        act_bounds = np.array(
            [[-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]],
            dtype=DType,
        )
        self.action_space = spaces.Box(
            low=act_bounds[0], high=act_bounds[1], dtype=DType
        )

    @property
    def state(self) -> State:
        return State(self._env.state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed, options=options)
        self.info = {}

        if options is not None:
            reset_state = options.get("reset_state", self.initial_state)
        else:
            reset_state = self.initial_state
        self._env.reset(reset_state)
        return self.state, self.info

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        """

        .. math::
            \\begin{align}
                \\ddot x &= -g \\theta\\\\
                \\ddot y &= g \\phi\\\\
                \\ddot z &=  g - T/m\\\\
                \\ddot \\phi &= \\tau_\\phi/I_z\\\\
                \\ddot \\theta &= \\tau_\\theta/I_y\\\\
                \\ddot \\psi &= \\tau_\\psi/I_z\\\\
            \\end{align}

        .. math::
            \\begin{bmatrix}
                x'\\\\y'\\\\z'\\\\x''\\\\y''\\\\z''\\\\\\phi'\\\\\\theta'\\\\\\psi'\\\\p'\\\\q'\\\\r'
            \\end{bmatrix}
            &=
            \\begin{bmatrix}
                0& 0& 0& 1& 0& 0& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 1& 0& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 1& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& g& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& -g& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 1& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 1& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 1\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0\\\\
                0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0& 0\\\\
            \\end{bmatrix}
            \\begin{bmatrix}
                x\\\\y\\\\z\\\\x'\\\\y'\\\\z'\\\\\\phi\\\\\\theta\\\\\\psi\\\\p\\\\q\\\\r
            \\end{bmatrix}
            +
            \\begin{bmatrix}
                0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 1 / m\\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0\\\\
                1/Ix & 0 & 0 & 0\\\\
                0 & 1/Iy & 0 & 0\\\\
                0 & & 1 / Iz & 0
            \\end{bmatrix}
            \\begin{bmatrix}
                \\tau_\\phi\\\\\\tau_\\theta\\\\\\tau_\\psi\\\\T
            \\end{bmatrix}
            +
            \\begin{bmatrix}
                0\\\\0\\\\0\\\\0\\\\0\\\\-g\\\\0\\\\0\\\\0\\\\0\\\\0\\\\0
            \\end{bmatrix}

        .. math::

            \\begin{bmatrix}
                \\tau_\\phi\\\\\\tau_\\theta\\\\\\tau_\\psi\\\\T
            \\end{bmatrix}
             &= \\begin{bmatrix}
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
        action :

        Returns
        -------

        """
        self._env.step(action)
        return self.state, 0, False, False, self.info
