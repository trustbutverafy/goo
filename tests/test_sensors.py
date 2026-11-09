"""The Sensor contract (SPEC.md §5): calibrated precision, speckle where coherence says there must be speckle,
and looks that help the way independent looks should."""
import numpy as np
import pytest
from goo.sensors.lidar import CELL
from goo.sensors.coherent import Surface, Geometry

L4 = (1530e-9, 1545e-9, 1560e-9, 1575e-9)


def flat_geometry(theta_deg, n=60, spot=0.12):
    Z = np.zeros((n, n)); th = np.radians(theta_deg); c = n * CELL / 2
    surf = Surface(Z, np.full((n, n), 0.9), np.full((n, n), 1.5))
    return Geometry(surf, (c - 150 * np.sin(th), c, 150 * np.cos(th)), spacing=0.1, spot=spot, aim_box=(c - 3, c + 3, c - 3, c + 3), seed=1)


@pytest.fixture(scope="module")
def g0(): return flat_geometry(0)


@pytest.fixture(scope="module")
def g40(): return flat_geometry(40)


def rms(g, m): e = g.errors(m["R"])["e_beam"][g.valid] * 1000; return float(np.sqrt(np.mean(e ** 2)))


def test_thermal_calibration(g0):
    # the averaged-spot model at normal incidence on white is the calibration point: 8 um
    assert 6.5 < rms(g0, g0.measure(mode="incoherent")) < 9.5


def test_one_coherent_look_is_fully_developed_speckle(g0):
    I = g0.measure()["I"][g0.valid]
    assert 0.8 < I.std() / I.mean() < 1.2          # speckle contrast of one look ~ 1


def test_looks_help_and_tilt_hurts(g0, g40):
    one, four = rms(g40, g40.measure()), rms(g40, g40.measure(lams=L4, combine="weighted"))
    assert four < 0.5 * one
    assert rms(g40, g40.measure(lams=L4, combine="weighted")) > 1.5 * rms(g0, g0.measure(lams=L4, combine="weighted"))


def test_smaller_spot_less_tilt_error():
    big, small = flat_geometry(40, spot=0.12), flat_geometry(40, spot=0.06)
    kw = dict(lams=L4, combine="weighted")
    assert rms(small, small.measure(**kw)) < 0.75 * rms(big, big.measure(**kw))


def test_sensor_sees_only_the_surface():
    import inspect
    src = inspect.getsource(Geometry)
    assert "species" not in src and ".q[" not in src and "strata" not in src
