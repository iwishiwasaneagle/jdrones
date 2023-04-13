import numpy as np
import pybullet as p
import pytest
from jdrones.data_models import Limits
from jdrones.data_models import PyBulletFnGroundPlaneConfig
from jdrones.data_models import SimulationType
from jdrones.envs import PyBulletDroneEnv


def fn1(x, y):
    return np.abs(x) + np.abs(y)


def fn2(x, y):
    return x + y


def fn3(x, y):
    return -2 + x


def fn4(x, y):
    return -1 + y


def fn5(x, y):
    return np.ones(np.shape(x)) * 2


def fn6(x, y):
    return np.cos(x) + y * 0.01 + x * 0.2 + 1


@pytest.mark.parametrize("Ntest,Nact", [(11, 100)])
@pytest.mark.parametrize("limits", [(-5, 5, -5, 5)])
@pytest.mark.parametrize("fn", [fn1, fn2, fn3, fn4, fn5, fn6])
def test_add_terrain_by_fn(
    pbdroneenv: PyBulletDroneEnv,
    limits: tuple[float, float, float, float],
    fn,
    Ntest,
    Nact,
    simulation_type,
):
    debug = simulation_type == SimulationType.GUI

    p.removeBody(pbdroneenv.ids.plane)
    p.removeBody(pbdroneenv.ids.drone)
    terrain = pbdroneenv._add_terrain_by_fn(
        PyBulletFnGroundPlaneConfig(fn=fn, limits=Limits.from_list(limits), N=Nact)
    )

    MINX, MAXX, MINY, MAXY = limits
    XOFF, YOFF = (MINX + MAXX) / 2, (MINY + MAXY) / 2
    SHRINK = 0.1
    xtest, ytest = np.meshgrid(
        np.linspace(MINX + SHRINK, MAXX - SHRINK, Ntest),
        np.linspace(MINY + SHRINK, MAXY - SHRINK, Ntest),
    )
    RAYS = np.column_stack([xtest.flatten(), ytest.flatten()])
    BASE_Z = np.ones((RAYS.shape[0], 1))
    TO_Z = BASE_Z * -10000
    FROM_Z = BASE_Z * 10000
    TO_RAYS = np.hstack([RAYS, TO_Z])
    FROM_RAYS = np.hstack([RAYS, FROM_Z])

    tests = p.rayTestBatch(FROM_RAYS, TO_RAYS, numThreads=0)

    results = np.array(list(map(lambda v: (v[0], *v[3]), tests)))
    expz = fn(results[:, 1] - XOFF, results[:, 2] - YOFF)

    if debug:
        show_debug(results, expz, RAYS, terrain)

    assert not np.any(results[:, 0] == -1), "A ray has missed the terrain"
    assert np.all(results[:, 0] == terrain), "Non-terrain object hit"

    if not np.allclose(expz, results[:, 3]):
        err = expz - results[:, 3]
        errstr = (
            f"Expected z-values don't match with real z-values | Error Stats -> "
            f"mean = {np.mean(err):.4f} std = {np.std(err):.4f} min = {err.min():.4f}"
            f" max = {err.max():.4f}"
        )
        assert False, errstr


def show_debug(results, expzs, RAYS, terrainid):
    for i in np.eye(3):
        p.addUserDebugLine((0, 0, 0), i, i, lineWidth=2, parentObjectUniqueId=terrainid)
    for result, expz, ray in zip(results, expzs, RAYS):
        objectUniqueId, hitx, hity, hitz = result
        hitPosition = (hitx, hity, hitz)
        isclose = np.isclose(hitz, expz, 0.1)
        TO = (*ray, expzs.min() - 5)
        if objectUniqueId == -1:
            colour = [0, 0, 1]
        elif not isclose:
            colour = [1, 0, 0]
            TO = hitPosition
            e = expz - hitz
            formatted_e = np.format_float_positional(
                e, fractional=False, precision=4, unique=True, trim="k"
            )
            p.addUserDebugText(
                f"{formatted_e}",
                TO,
                textColorRGB=[1, 0, 0] if e < 0 else [1, 1, 0],
            )
        else:
            colour = [0, 1, 0]
            TO = hitPosition
        p.addUserDebugLine((*ray, expzs.max() + 5), TO, colour)
