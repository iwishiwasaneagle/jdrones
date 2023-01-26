import numpy as np
import numpy.typing as npt


def clip_scalar(value: float, vmin: float, vmax: float) -> float:
    return vmin if value < vmin else vmax if value > vmax else value


def clip(value: npt.ArrayLike, vmin: float, vmax: float) -> npt.ArrayLike:
    return np.core.umath.maximum(np.core.umath.minimum(value, vmax), vmin)
