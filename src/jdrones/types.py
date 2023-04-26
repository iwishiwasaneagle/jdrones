#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np

DType = np.float64
"""Default data type of 64-bit float"""
FloatLike = DType | float
"""Wrapper for multiple float-like types"""

VEC3 = tuple[FloatLike, FloatLike, FloatLike]
""":math:`(a,b,c)` vector"""
VEC4 = tuple[FloatLike, FloatLike, FloatLike, FloatLike]
""":math:`(a,b,c,d)` vector"""
MAT3X3 = tuple[VEC3, VEC3, VEC3]
""":math:`3 \\times 3` matrix"""
MAT4X3 = tuple[VEC4, VEC4, VEC4, VEC4]
""":math:`4 \\times 4` matrix"""
Action = list[FloatLike]
LinearXAction = tuple[
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
]
""":math:`\\vec x =\\left[x,y,z,\\dot x,\\dot y,\\dot z,\\phi,\\theta,\\psi,p,q,r\
\\right]` used within the linearised model."""
PropellerAction = VEC4
""":math:`(P_1,P_2,P_3,P_4)` propeller inputs"""
PositionAction = VEC3
""":math:`(x,y,z)` position inputs"""
PositionVelocityAction = tuple[VEC3, VEC3]
""":math:`(x,y,z)` position inputs, and :math:`(v_x,v_y,v_z)` velocity inputs"""
