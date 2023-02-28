#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import enum
from typing import Any
from typing import Callable
from typing import Tuple

import nptyping as npt
import numpy as np
import pandas as pd
import pybullet as p
import pydantic

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


class KLengthArray(np.ndarray):
    k: int = None

    def __new__(cls, input_array=None):
        if cls.k is None:
            raise NotImplementedError("k must be set")
        if input_array is None:
            obj = np.zeros(cls.k)
        else:
            obj = np.asarray(input_array)
            if obj.shape != (cls.k,):
                raise ValueError(f"Incorrect shape {obj.shape}")
        return obj.view(cls)


class State(KLengthArray):
    """
    Default state observation of the system
    """

    k: int = 20

    @property
    def pos(self) -> VEC3:
        """
        Position

        Returns
        -------
        float,float,float
            :math:`(x,y,z)`
        """
        return self[:3]

    @pos.setter
    def pos(self, pos: VEC3):
        self[:3] = pos

    @property
    def quat(self) -> VEC4:
        """
        Quaternion

        Returns
        -------
        float,float,float,float
            :math:`(x,y,z,w)`
        """
        return self[3:7]

    @quat.setter
    def quat(self, quat: VEC4):
        self[3:7] = quat

    @property
    def rpy(self) -> VEC3:
        """
        Roll, pitch, and yaw

        Returns
        -------
        float,float,float
             :math:`(\\phi,\\theta,\\psi)`
        """
        return self[7:10]

    @rpy.setter
    def rpy(self, rpy: VEC3):
        self[7:10] = rpy

    @property
    def vel(self) -> VEC3:
        """
        Linear velocity

        Returns
        -------
        float,float,float
             :math:`(x,y,z)`
        """
        return self[10:13]

    @vel.setter
    def vel(self, vel: VEC3):
        self[10:13] = vel

    @property
    def ang_vel(self) -> VEC3:
        """
        Angular velocity

        Returns
        -------
        float,float,float
             :math:`(p,q,r)`
        """
        return self[13:16]

    @ang_vel.setter
    def ang_vel(self, ang_vel: VEC3):
        self[13:16] = ang_vel

    @property
    def prop_omega(self) -> VEC4:
        """
        Propeller angular velocity

        Returns
        -------
        float,float,float,float
             :math:`(\\Omega_0,\\Omega_1,\\Omega_2,\\Omega_3)`
        """
        return self[16:20]

    @prop_omega.setter
    def prop_omega(self, prop_omega: VEC4):
        self[16:20] = prop_omega

    @classmethod
    def from_x(cls, x: LinearXAction):
        return cls(
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

    def to_x(self) -> LinearXAction:
        return np.concatenate([self.pos, self.vel, self.rpy, self.ang_vel])


class Conversions:
    @staticmethod
    def iter_to_df(x, *, tag, dt, N, cols):
        t = np.linspace(0, len(x) * dt, len(x))
        df = pd.DataFrame(
            x,
            columns=cols,
            index=t,
        )
        df.index.name = "t"

        if len(df) > N:
            inds = np.linspace(0, len(df) - 1, N, dtype=int)
            df = df.take(inds)

        df_long = df.melt(
            var_name="variable", value_name="value", ignore_index=False
        ).reset_index()
        df_long["tag"] = tag
        return df_long


class States(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is None:
            obj = np.array([])
        else:
            obj = np.asarray(input_array)
            if obj.shape[1] != 20:
                raise ValueError(f"Incorrect shape {obj.shape}")
        return obj.view(cls)

    def to_df(self, *, tag, dt=1, N=500):
        df = Conversions.iter_to_df(
            x=self,
            tag=tag,
            dt=dt,
            N=N,
            cols=[
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "phi",
                "theta",
                "psi",
                "vx",
                "vy",
                "vz",
                "p",
                "q",
                "r",
                "P0",
                "P1",
                "P2",
                "P3",
            ],
        )
        return df


class SimulationType(enum.IntEnum):
    """Enum to handle the support pybullet simulation types"""

    DIRECT = p.DIRECT
    """No GUI"""
    GUI = p.GUI
    """With GUI"""


class URDFModel(pydantic.BaseModel):
    g: float = 9.81
    """Acceleration due to gravity (m/s^2)"""

    l: float
    """Arm length (m)"""

    mass: float
    """Mass (kg)"""

    I: Tuple[float, float, float]
    """Inertia in the form (Ixx, Iyy, Izz) (kg/m^2)"""

    k_T: float
    """Rotor thrust gain (Ns/m)"""
    tau_T: float
    """Rotor thrust time constant"""
    k_Q: float
    """Rotor torque gain (Ns/)"""
    tau_Q: float
    """Rotor torque time constant"""

    drag_coeffs: Tuple[float, float, float]
    """Drag coefficients (x,y,z)"""

    max_vel_ms: float
    """Maximum velocity of the drone (m/s)"""

    mixing_matrix: Callable
    """Mixing matrix describing RPY + T to propeller RPM"""

    @property
    def weight(self) -> float:
        """
        Weight of the drone

        Returns
        -------
        float
            :math:`W=mg`
        """
        return self.g * self.mass

    def rpyT2rpm(
        self, rpyT: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Apply the inverse :meth:`mixing_matrix`.

        Parameters
        ----------
        rpyT: float,float,float,float
            Roll, pitch, yaw, thrust

        Returns
        -------
        float,float,float,float
            RPMs for P0,P1,P2,P3
        """
        return np.linalg.solve(
            self.mixing_matrix(length=self.l, k_T=self.k_T, k_Q=self.k_Q),
            rpyT,
        )

    def rpm2rpyT(
        self, rpm: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Apply the :meth:`mixing_matrix`.

        Parameters
        ----------
        rpm : float,float,float,float

        Returns
        -------
        float,float,float,float
            Roll, pitch, yaw, thrust
        """
        return self.mixing_matrix(length=self.l, k_T=self.k_T, k_Q=self.k_Q) @ rpm

    filepath: str
    """File path to URDF model"""


class PyBulletIds(pydantic.BaseModel):
    """
    Container to hold the IDs of the various pybullet items
    """

    client: int = None
    """Physical simulation client ID"""
    plane: int = None
    """The ground plane ID"""
    drone: int = None
    """The drone ID"""
