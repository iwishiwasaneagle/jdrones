import numpy as np
import pytest
from jdrones.envs.position import FifthOrderPolyPositionWithLookAheadDroneEnv


@pytest.mark.parametrize(
    "A,B,C,exp",
    [
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0)],
        [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 1, 0)],
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (0.35355339, 0.35355339, 0.0)],
        [(1, 2, 3), (1, 3, 3), (2, 3, 3), (0.35355339, 0.35355339, 0.0)],
        [(0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)],
    ],
)
def test_calc_v_at_b(A, B, C, exp):
    N = 1
    V = 1
    A, B, C = np.array([A, B, C])
    act = FifthOrderPolyPositionWithLookAheadDroneEnv.calc_v_at_B(A, B, C, V=V, N=N)
    assert np.allclose(exp, act)
