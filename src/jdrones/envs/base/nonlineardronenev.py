#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Optional
from typing import Tuple

from jdrones.data_models import State
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.types import PropellerAction
from libjdrones import NonLinearDynamicModelDroneEnv as _NonLinearDynamicModelDroneEnv


class NonlinearDynamicModelDroneEnv(BaseDroneEnv):
    """
    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("NonLinearDynamicModelDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<NonlinearDynamicModelDroneEnv<NonLinearDynamicModelDroneEnv-v0>>>>
    """

    _env: _NonLinearDynamicModelDroneEnv

    def __init__(
        self,
        initial_state: State = None,
        dt: float = 1 / 240,
    ):
        self._env = _NonLinearDynamicModelDroneEnv(dt)
        super().__init__(initial_state=initial_state, dt=dt)

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
                m\\vec x '' &=
                \\begin{bmatrix}
                    0\\\\0\\\\mg
                \\end{bmatrix}
                -
                R^B_E
                \\begin{bmatrix}
                    0\\\\0\\\\k_T\\Sigma^4_{i=1} P_i^2
                \\end{bmatrix}
                -
                R^B_E (\\vec C_d R^E_B \\vec x ') \\\\
                \\vec I \\vec \\phi'' &=
                \\begin{bmatrix}
                    l k_T (P_4^2-P_2^2) \\\\
                    l k_T (P_3^2-P_1^2) \\\\
                    k_Q (P_1^2-P_2^2+P_3^2-P_4^2)
                \\end{bmatrix}
            \\end{align}

        Parameters
        ----------
        action :

        Returns
        -------

        """
        self._env.step(action)
        return self.state, 0, False, False, self.info
