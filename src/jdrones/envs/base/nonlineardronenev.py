#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import functools
from typing import Tuple

import numba
import numpy as np
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.transforms import euler_to_quat
from jdrones.transforms import euler_to_rotmat
from jdrones.types import DType
from jdrones.types import MAT3X3
from jdrones.types import PropellerAction
from jdrones.types import VEC3


class NonlinearDynamicModelDroneEnv(BaseDroneEnv):
    """
    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("NonLinearDynamicModelDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<NonlinearDynamicModelDroneEnv<NonLinearDynamicModelDroneEnv-v0>>>>
    """

    @staticmethod
    @functools.cache
    def _get_cached_time_invariant_params(
        model: URDFModel,
    ) -> tuple[MAT3X3, MAT3X3, float, VEC3, float]:
        """
        Cache the time invariant parameters in order to save runtime execution costs

        Parameters
        ----------
        model : URDFModel

        Returns
        -------
        inertias : MAT3X3
            3x3 inertia matrix
        inv_inertias: MAT3X3
            3x3 inverted inertia matrix
        m: float
            mass
        drag_coeffs: VEC3
            3D drag coefficient vector
        g: float
            acceleration due to gravity
        """
        inertias = np.diag(np.array(model.I, dtype=DType))
        m = model.mass
        drag_coeffs = model.drag_coeffs
        g = model.g
        return (
            inertias,
            np.linalg.inv(inertias),
            m,
            np.array(drag_coeffs, dtype=DType),
            g,
        )

    @staticmethod
    @numba.njit(
        [
            "float64[:](float64[:],float64[:],float64[:],float64[:],"
            "float64[:,:],float64[:,:],float64,float64[:],float64)",
        ]
    )
    def _calc_dstate(
        rpy,
        vel,
        ang_vel,
        u_star,
        inertias,
        inv_intertias,
        m,
        drag_coeffs,
        g,
    ):
        UZ = np.array([0, 0, 1], dtype=DType).reshape((-1, 1))
        R_W_Q = euler_to_rotmat(rpy)
        R_Q_W = np.transpose(R_W_Q)

        body_vel = np.dot(R_W_Q, vel)
        drag_force = -np.sign(vel) * np.dot(R_Q_W, drag_coeffs * np.square(body_vel))
        dstate = np.zeros((20,), dtype=DType)
        dstate[:3] = vel
        dstate[7:10] = ang_vel
        dstate[10:13] = (
            -m * g * UZ.T + (R_W_Q @ UZ).T * u_star[3] + drag_force
        ).flatten() / m
        dstate[13:16] = np.dot(inv_intertias, u_star[0:3])
        return dstate

    @classmethod
    def calc_dstate(cls, action: PropellerAction, state: State, model: URDFModel):
        """
        Calculate the state derivative as outlined in
        :meth:`~jdrones.envs.base.nonlineardronenev.NonlinearDynamicModelDroneEnv.step`

        Parameters
        ----------
        action : ProperllerAction
        state :  State
        model :  URDFModel

        Returns
        -------
        State
            The state derivative
        """
        u_star = model.rpm2rpyT(np.square(action))
        return cls._calc_dstate(
            state.rpy,
            state.vel,
            state.ang_vel,
            u_star,
            *cls._get_cached_time_invariant_params(model),
        )

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
