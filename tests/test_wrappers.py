import gymnasium
import numpy as np
import pytest
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.wrappers import EnergyCalculationWrapper


def test_energycalcwrapper_basedrone(lineardroneenv):
    wrapped = EnergyCalculationWrapper(lineardroneenv)

    wrapped.reset()

    obs, reward, term, trunc, info = wrapped.step(np.ones(4) * 100)

    assert "energy" in info
    assert info["energy"] > 0


def test_energycalcwrapper_positiondrone(lqrdroneenv):
    wrapped = EnergyCalculationWrapper(lqrdroneenv)

    wrapped.reset()

    obs, reward, term, trunc, info = wrapped.step(wrapped.action_space.sample())

    assert "energy" in info
    assert info["energy"] > 0


def test_energycalcwrapper_fail_in_init():
    with pytest.raises(ValueError):
        EnergyCalculationWrapper(gymnasium.Env())


def test_energycalcwrapper_fail_in_step():
    class TestEnv(BaseDroneEnv):
        def step(self, action):
            del self.state
            return 1, 0, False, False, {}

    env = EnergyCalculationWrapper(TestEnv())

    with pytest.raises(ValueError):
        env.step(env.action_space.sample())
