import numpy as np
import pytest
from jdrones.envs.position import PositionDroneEnv


@pytest.mark.parametrize(
    "curpos,tgtpos,exp",
    [
        [(0, 0, 0), (1, 0, 0), 0],
        [(0, 0, 0), (0, 1, 0), np.pi / 2],
        [(0, 0, 0), (-1, 0.0000001, 0), np.pi],
        [(0, 0, 0), (-1, -0.0000001, 0), -np.pi],
        [(0, 0, 0), (0, -1, 0), -np.pi / 2],
        [(1, 2, 0), (2, 2, 0), 0],
        [(1, 1, 0), (2, 2, 0), np.pi / 4],
        [(1, 2, 0), (2, 4, 0), 1.1071487177940904],
        [(0, 0, 10), (0, 1, 10), np.pi / 2],
        [(0, 0, 10), (0, -1, 10), -np.pi / 2],
    ],
)
def test_angle_between(curpos, tgtpos, exp):
    act = PositionDroneEnv._calc_target_yaw(curpos, tgtpos)
    assert np.isclose(act, exp)
