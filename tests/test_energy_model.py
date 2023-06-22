import numpy as np
import pytest

from jdrones.energy_model import (
    BaseEnergyModel,
    StaticPropellerVariableVelocityEnergyModel,
)


@pytest.fixture
def baseenergymodel(dt, urdfmodel):
    class cls(BaseEnergyModel):
        def power(self, state):
            return state

    cls.__abstractmethods__ = set()

    return cls(dt, urdfmodel)


@pytest.fixture(params=[1])
def A(request):
    return request.param


@pytest.fixture(params=[0.1])
def F(request):
    return request.param


@pytest.fixture(params=[1])
def K(request):
    return request.param


@pytest.fixture(params=[9000])
def v_b(request):
    return request.param


@pytest.fixture
def SPVVenergymodel(dt, urdfmodel, v_b, K, F, A):
    return StaticPropellerVariableVelocityEnergyModel(
        dt, urdfmodel, v_b=v_b, K=K, F=F, A=A
    )


@pytest.mark.parametrize("state", [1, 2, 3])
def test_baseenergymodel(baseenergymodel, state):
    assert baseenergymodel.energy(state) == state * baseenergymodel.dt


@pytest.mark.parametrize("low,high", [(0, 1e4)])
def test_staticpropellervariablevelocityenergymodel_low_high(
    low, high, SPVVenergymodel
):
    assert SPVVenergymodel.energy(low) < SPVVenergymodel.energy(high)

    assert SPVVenergymodel.p_blade(low) < SPVVenergymodel.p_blade(high)
    assert SPVVenergymodel.p_parasite(low) < SPVVenergymodel.p_parasite(high)
    assert SPVVenergymodel.p_induced(low) > SPVVenergymodel.p_induced(high)


@pytest.mark.parametrize(
    "v_b,K,F,A,mass,g,rho", [(9000, 1, 0.1, 1, 2, 9.81, 1.225)], indirect=True
)
def test_staticpropellervariablevelocityenergymodel_low_isnt_always_minima(
    SPVVenergymodel,
):
    low = 0
    high = 10
    N = 100

    v = np.linspace(low, high, N)
    e = SPVVenergymodel.energy(v)

    min_e = np.argmin(e)
    assert 0 < min_e < N
