import numpy as np

from jdrones.types import VEC3, VEC4, MAT3X3

import pybullet as p

def quat_to_euler(quat: VEC4) -> VEC3:
    return p.getEulerFromQuaternion(quat)
def euler_to_quat(euler: VEC3) -> VEC4:
    return p.getQuaternionFromEuler(euler)



def quat_to_rotmat(quat: VEC4) -> MAT3X3:
    return np.reshape(p.getMatrixFromQuaternion(quat),(3,3))

def euler_to_rotmat(euler: VEC3) -> MAT3X3:
    return quat_to_rotmat(euler_to_quat(euler))

