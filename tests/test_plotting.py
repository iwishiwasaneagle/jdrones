import numpy as np
import pytest
from jdrones.data_models import STATE_ENUM
from jdrones.data_models import States
from jdrones.plotting import extract_state
from jdrones.plotting import extract_state_value
from jdrones.plotting import valid_df


@pytest.fixture(params=[(5, 20), (100, 20)])
def states(request):
    return States(np.random.rand(*request.param))


@pytest.fixture(params=["test"])
def tag(request):
    return request.param


@pytest.fixture
def dataframe(states, tag, dt):
    return states.to_df(tag=tag, dt=dt)


def test_valid_df(dataframe):
    assert valid_df(dataframe)


def test_invalid_df_by_sort(dataframe):
    assert not valid_df(dataframe.iloc[::-1])


def test_invalid_df_by_type(dataframe):
    assert not valid_df(int(1))


@pytest.mark.parametrize("col", ["t", "variable", "value", "tag"])
def test_invalid_df_by_column(dataframe, col):
    assert not valid_df(dataframe.drop(columns=[col]))


ALL_STATES = pytest.mark.parametrize(
    "i,variable",
    tuple(enumerate(STATE_ENUM.as_list())),
)


@ALL_STATES
def test_extract_variable(dataframe, states, i, variable):
    df = extract_state(dataframe, variable)
    assert np.allclose(df.value, states[:, i])


@ALL_STATES
def test_extract_variable_value(dataframe, states, i, variable):
    value = extract_state_value(dataframe, variable)
    assert np.allclose(value, states[:, i])
