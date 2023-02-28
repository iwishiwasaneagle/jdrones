#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from functools import cached_property
from typing import Tuple

import numpy as np
from gymnasium import spaces
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.transforms import euler_to_quat
from jdrones.types import PropellerAction
from jdrones.types import State
from jdrones.types import URDFModel


class LinearDynamicModelDroneEnv(BaseDroneEnv):
    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        dt: float = 1 / 240,
    ):
        super().__init__(model, initial_state, dt)

        self.A, self.B, self.C = self.get_matrices(model)

    @staticmethod
    def get_matrices(model: URDFModel):
        m = model.mass
        g = model.g
        Ix, Iy, Iz = model.I

        A = np.array(
            [
                (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # x
                (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),  # y
                (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),  # z
                (0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0),  # dx
                (0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0),  # dy
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # dz
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # phi
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # theta
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # psi
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # dphi
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # dtheta
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # dpsi
            ]
        )

        B = np.array(
            [
                (0, 0, 0, 0),  # x
                (0, 0, 0, 0),  # y
                (0, 0, 0, 0),  # z
                (0, 0, 0, 0),  # dx
                (0, 0, 0, 0),  # dy
                (0, 0, 0, 1 / m),  # dz
                (0, 0, 0, 0),  # phi
                (0, 0, 0, 0),  # theta
                (0, 0, 0, 0),  # psi
                (1 / Ix, 0, 0, 0),  # dphi
                (0, 1 / Iy, 0, 0),  # dtheta
                (0, 0, 1 / Iz, 0),  # dpsi
            ]
        )

        C = np.vstack([0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0, 0])

        return A, B, C

    @staticmethod
    def calc_dx(A, B, C, x, u):
        return (A @ x + B @ u + C.T).flatten()

    @staticmethod
    def _x_to_state(x):
        return State(
            np.concatenate(
                [
                    x[:3],
                    (0, 0, 0, 0),
                    x[6:9],
                    x[3:6],
                    x[9:12],
                    (0, 0, 0, 0),
                ]
            )
        )

    @staticmethod
    def _state_to_x(state):
        return np.concatenate([state.pos, state.vel, state.rpy, state.ang_vel])

    def calc_dstate(self, action) -> State:
        x = self._state_to_x(self.state)
        dstate = self.calc_dx(self.A, self.B, self.C, x, action)
        return self._x_to_state(dstate)

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
            l k_T& 0& -l k_T& 0\\\\
            -k_Q&k_Q& -k_Q& k_Q \\\\
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
        dstate = self.calc_dstate(self.model.rpm2rpyT(action))
        # Update step
        self.state += self.dt * dstate

        # Update derived state items
        self.state.prop_omega = action
        self.state.quat = euler_to_quat(self.state.rpy)

        # Return
        return self.state, 0, False, False, self.info

    @cached_property
    def action_space(self):
        return spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]),
        )
