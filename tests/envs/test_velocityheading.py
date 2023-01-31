import numpy as np
import pytest
from jdrones.envs.velocityheading import VelHeadAltDroneEnv


@pytest.mark.parametrize(
    "vel,rpy,exp",
    [
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(1, 0, 0), (0, 0, 0), (1, 0, 0)],
        [(0, 1, 0), (0, 0, 0), (0, 1, 0)],
        [(0, 0, 1), (0, 0, 0), (0, 0, 1)],
        [(1, 0, 0), (0, 0, np.pi / 2), (0, -1, 0)],
        [(0, 1, 0), (0, 0, np.pi / 2), (1, 0, 0)],
        [(0, 0, 1), (0, 0, np.pi / 2), (0, 0, 1)],
        [(1, 1, 0), (np.pi / 2, 0, 0), (1, 0, -1)],
    ],
)
def test__vi_to_vb(vel, rpy, exp):
    act = VelHeadAltDroneEnv._vi_to_vb(vel, rpy)
    assert np.allclose(act, exp)
