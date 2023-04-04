import enum
import functools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jdrones.data_models import STATE_ENUM


class SUPPORTED_SEABORN_THEMES(str, enum.Enum):
    WHITEGRID = "whitegrid"
    DARKGRID = "darkgrid"
    WHITE = "white"
    DARK = "dark"
    TICKS = "ticks"


def apply_seaborn_theme(
    style: SUPPORTED_SEABORN_THEMES = SUPPORTED_SEABORN_THEMES.WHITEGRID,
):
    """
    Apply a seaborn theme
    Parameters
    ----------
    style : SUPPORTED_SEABORN_THEMES

    Returns
    -------
    """
    sns.set_theme(style=str(style.value))


def valid_df(df: pd.DataFrame) -> bool:
    """
    Ensure the :class:`pandas.DataFrame` is one, or at least in the same shape as one,
    created by :meth:`jdrones.data_models.States.to_df`.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    bool
        True if valid, False if not
    """

    is_dataframe = isinstance(df, pd.DataFrame)
    if not is_dataframe:
        return is_dataframe
    has_expected_columns = {"t", "variable", "value", "tag"} == set(df.columns)
    if not has_expected_columns:
        return has_expected_columns
    starts_at_0s = df.t.min() == 0.0
    is_sorted_by_t = (df.iloc[0].t == 0.0) & (df.iloc[-1].t == df.t.max())
    return starts_at_0s & is_sorted_by_t


def validate_df_wrapper(func):
    @functools.wraps(func)
    def fn(df, *args, **kwargs):
        if not valid_df(df):
            raise ValueError("df is invalid")
        return func(df, *args, **kwargs)

    return fn


def extract_state(df: pd.DataFrame, state: STATE_ENUM) -> pd.DataFrame:
    """
    Extract the state from a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
    state : STATE_ENUM

    Returns
    -------
    pandas.DataFrame
    """
    return df[df.variable == state]


def extract_state_value(df: pd.DataFrame, state: STATE_ENUM) -> list[float]:
    """
    Extract the state values from a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
    state : STATE_ENUM

    Returns
    -------
    list[float]
    """
    return extract_state(df, state).value


@validate_df_wrapper
def plot_state_vs_state(
    df: pd.DataFrame,
    state_a: STATE_ENUM,
    state_b: STATE_ENUM,
    ax: plt.Axes,
    label: str = None,
):
    """
    Plot the 2d a-b path

    Parameters
    ----------
    df : pandas.DataFrame
    state_a : STATE_ENUM
    state_b : STATE_ENUM
    ax : matplotlib.pyplot.Axes
    label : str
        Optional label
        (Default = None)
    """
    a = extract_state_value(df, state_a)
    b = extract_state_value(df, state_b)
    ax.set_xlabel(state_a)
    ax.set_ylabel(state_b)
    if label is not None:
        ax.plot(a, b, label=label)
    else:
        ax.plot(a, b)


@validate_df_wrapper
def plot_state_vs_state_vs_state(
    df: pd.DataFrame,
    state_a: STATE_ENUM,
    state_b: STATE_ENUM,
    state_c: STATE_ENUM,
    ax: plt.Axes,
    label: str = None,
):
    """
    Plot the 3d a-b-c path

    Parameters
    ----------
    df : pandas.DataFrame
    state_a : STATE_ENUM
    state_b : STATE_ENUM
    state_c : STATE_ENUM
    ax : matplotlib.pyplot.Axes
    label : str
        Optional label
        (Default = None)
    """
    if not hasattr(ax, "plot3D"):
        raise Exception(
            f"{ax=} does not have plot3D. Ensure the correct "
            "projection has been set."
        )
    a = extract_state_value(df, state_a)
    b = extract_state_value(df, state_b)
    c = extract_state_value(df, state_c)
    ax.set_xlabel(state_a)
    ax.set_ylabel(state_b)
    ax.set_zlabel(state_c)
    if label is not None:
        ax.plot(a, b, c, label=label)
    else:
        ax.plot(a, b, c)


@validate_df_wrapper
def plot_2d_path(df: pd.DataFrame, ax: plt.Axes, label: str = None):
    """
    Plot the 2d x-y path

    Parameters
    ----------
    df : pandas.DataFrame
    ax : matplotlib.pyplot.Axes
    label : str
        Optional label
        (Default = None)
    """
    plot_state_vs_state(df, "x", "y", ax, label)


@validate_df_wrapper
def plot_3d_path(df: pd.DataFrame, ax: plt.Axes, label: str = None):
    """
    Plot the 3d x-y-z path

    Parameters
    ----------
    df : pandas.DataFrame
    ax : matplotlib.pyplot.Axes
    label : str
        Optional label
        (Default = None)
    """
    plot_state_vs_state_vs_state(df, "x", "y", "z", ax, label)


@validate_df_wrapper
def plot_state_over_time(df: pd.DataFrame, variable: STATE_ENUM, ax: plt.Axes):
    """
    Plot a state over time

    Parameters
    ----------
    df : pandas.DataFrame
    variable : STATE_ENUM
        The state to plot
    ax : matplotlib.pyplot.Axes
    """
    a = extract_state(df, variable)
    v, t = a.value, a.t
    ax.set_ylabel(variable)
    ax.set_xlabel("t")
    ax.plot(t, v, label=variable)


@validate_df_wrapper
def plot_states_over_time(df: pd.DataFrame, variables: list[STATE_ENUM], ax: plt.Axes):
    """
    Plot states over time

    Parameters
    ----------
    df : pandas.DataFrame
    variables : list[STATE_ENUM]
        A list of states to plot
    ax : matplotlib.pyplot.Axes
    """
    for variable in variables:
        plot_state_over_time(df, variable, ax)
    ax.set_xlabel("t")


@validate_df_wrapper
def plot_standard(
    df: pd.DataFrame, figsize: tuple[float, float] = (12, 12), show: bool = True
):
    """
    Plot the standard 2-by-2 layout

    .. code::
        +------------------+----------------------+
        | 3D path          | position vs time     |
        +------------------+----------------------+
        | velocity vs time | euler angles vs time |
        +------------------+----------------------+

    Parameters
    ----------
    df : pandas.DataFrame
    figsize: float,float
        Figure size
        (Default = (12,12))
    show : bool
        If figure should be shown. Set to :code:`False` if you want to save the
        figure using :code:`plt.gcf()`
        (Default = :code:`True`)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(221, projection="3d")
    plot_3d_path(df, ax)

    for ind, states in (
        (222, ("x", "y", "z")),
        (223, ("vx", "vy", "vz")),
        (224, ("phi", "theta", "psi")),
    ):
        ax = fig.add_subplot(ind)
        plot_states_over_time(df, states, ax)
        ax.legend()

    fig.tight_layout()
    if show:
        plt.show()
