#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Tuple

import numpy as np
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.transforms import euler_to_quat
from jdrones.transforms import euler_to_rotmat
from jdrones.types import PropellerAction


class NonlinearDynamicModelDroneEnv(BaseDroneEnv):
    """
    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("NonLinearDynamicModelDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<NonlinearDynamicModelDroneEnv<NonLinearDynamicModelDroneEnv-v0>>>>
    """

    @staticmethod
    def calc_dstate(action: PropellerAction, state: State, model: URDFModel):
        Inertias = np.diag(model.I)
        m = model.mass
        drag_coeffs = model.drag_coeffs
        g = model.g
        u_star = model.rpm2rpyT(np.square(action))

        unit_z = np.array([0, 0, 1]).reshape((-1, 1))

        R_W_Q = euler_to_rotmat(state.rpy)
        R_Q_W = np.linalg.inv(R_W_Q)

        body_vel = np.dot(R_W_Q, state.vel)
        drag_force = -np.sign(state.vel) * np.dot(
            R_Q_W, np.array(drag_coeffs) * np.square(body_vel)
        )
        dstate = np.concatenate(
            [
                state.vel,
                (0, 0, 0, 0),
                state.ang_vel,
                (
                    -m * g * unit_z.T + (R_W_Q @ unit_z).T * u_star[3] + drag_force
                ).flatten()
                / m,
                np.linalg.solve(Inertias, u_star[0:3]),
                (0, 0, 0, 0),
            ]
        )
        return dstate

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
        # Get state
        dstate = self.calc_dstate(action, self.state, self.model)

        # Update step
        self.state += self.dt * dstate

        # Update derived state items
        self.state.prop_omega = action
        self.state.quat = euler_to_quat(self.state.rpy)

        # Return
        return self.state, 0, False, False, self.info
