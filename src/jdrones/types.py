import enum
from typing import Callable
from typing import Tuple

import numpy as np
import pybullet as p
import pydantic

VEC3 = Tuple[float, float, float]
VEC4 = Tuple[float, float, float, float]
MAT3X3 = Tuple[VEC3, VEC3, VEC3]
MAT4X3 = Tuple[VEC4, VEC4, VEC4]


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


class Action(KLengthArray):
    pass


class PropellerAction(Action):
    k: int = 4

    @property
    def P0(self):
        return self[0]

    @property
    def P1(self):
        return self[1]

    @property
    def P2(self):
        return self[2]

    @property
    def P3(self):
        return self[3]


class AttitudeAltitudeAction(Action):
    k: int = 4

    @property
    def roll(self):
        return self[0]

    @property
    def pitch(self):
        return self[1]

    @property
    def yaw(self):
        return self[2]

    @property
    def z(self):
        return self[3]


class VelHeadAltAction(Action):
    k: int = 4

    @property
    def vx_b(self):
        return self[0]

    @property
    def vy_b(self):
        return self[1]

    @property
    def yaw(self):
        return self[2]

    @property
    def z(self):
        return self[3]


class PositionAction(Action):
    k: int = 3

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]


class PositionVelAction(PositionAction):
    k: int = 4

    @property
    def vx_b(self):
        return self[3]


class Observation(KLengthArray):
    pass


class State(Observation):
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
        self, roll: float, pitch: float, yaw: float, thrust: float
    ) -> Tuple[float, float, float, float]:
        """
        Apply the :meth:`mixing_matrix`.

        Parameters
        ----------
        roll : float
        pitch : float
        yaw : float
        thrust : float

        Returns
        -------
        float,float,float,float
            RPMs for P0,P1,P2,P3
        """
        rpyT = np.array([roll, pitch, yaw, thrust])
        matrix_mul = self.mixing_matrix(self.l, self.k_T, self.k_Q) * rpyT
        matrix_mul_sum = np.sum(matrix_mul, axis=1)
        return np.sign(matrix_mul_sum) * np.sqrt(np.abs(matrix_mul_sum))

    filepath: str
    """File path to URDF model"""
