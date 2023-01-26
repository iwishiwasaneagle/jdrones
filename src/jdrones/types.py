import abc
import enum
from collections import UserList
from typing import Tuple, Callable

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
    def z(self):
        return self[2]

    @property
    def yaw(self):
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
    k: int = 20

    @property
    def pos(self) -> VEC3:
        return self[:3]

    @pos.setter
    def pos(self, pos: VEC3):
        self[:3] = pos

    @property
    def quat(self) -> VEC4:
        return self[3:7]

    @quat.setter
    def quat(self, quat: VEC4):
        self[3:7] = quat

    @property
    def rpy(self) -> VEC3:
        return self[7:10]

    @rpy.setter
    def rpy(self, rpy: VEC3):
        self[7:10] = rpy

    @property
    def vel(self) -> VEC3:
        return self[10:13]

    @vel.setter
    def vel(self, vel: VEC3):
        self[10:13] = vel

    @property
    def ang_vel(self) -> VEC3:
        return self[13:16]

    @ang_vel.setter
    def ang_vel(self, ang_vel: VEC3):
        self[13:16] = ang_vel

    @property
    def prop_omega(self) -> VEC4:
        return self[16:20]

    @prop_omega.setter
    def prop_omega(self, prop_omega: VEC4):
        self[16:20] = prop_omega


class SimulationType(enum.IntEnum):
    DIRECT = p.DIRECT
    GUI = p.GUI


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

    mixing_matrix: Callable

    @property
    def weight(self) -> float:
        return self.g * self.mass

    def rpyT2rpm(self, roll, pitch, yaw, thrust) -> Tuple[float, float, float, float]:
        rpyT = np.array([roll, pitch, yaw, thrust])
        matrix_mul = self.mixing_matrix(self.l, self.k_T, self.k_Q) * rpyT
        matrix_mul_sum = np.sum(matrix_mul, axis=1)
        return np.sign(matrix_mul_sum) * np.sqrt(np.abs(matrix_mul_sum))

    filepath: str
    """File path to URDF model"""
