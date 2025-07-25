import numpy as np
from numpy.testing import assert_almost_equal
from sphersgeo import SphericalPoint, SphericalPoints


def test_normalize_vectors():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    xyzn = SphericalPoints(xyz).xyz
    lengths = np.sqrt(np.sum(xyzn * xyzn, axis=-1))
    assert_almost_equal(lengths, 1.0)


def test_normalize_unit_vector():
    for i in range(3):
        xyz = np.array([0.0, 0.0, 0.0])
        xyz[i] = 1.0
        xyzn = SphericalPoint(xyz).xyz
        length = np.sqrt(np.sum(xyzn * xyzn, axis=-1))
        assert_almost_equal(length, 1.0)


def test_from_lonlat():
    lat = np.arange(-360.0, 360.0, 1.0)

    north_pole = SphericalPoints.from_lonlats(
        np.stack([lat, np.repeat(90.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(north_pole[:, 0], 0.0)
    assert_almost_equal(north_pole[:, 1], 0.0)
    assert_almost_equal(north_pole[:, 2], 1.0)

    south_pole = SphericalPoints.from_lonlats(
        np.stack([lat, np.repeat(-90.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(south_pole[:, 0], 0.0)
    assert_almost_equal(south_pole[:, 1], 0.0)
    assert_almost_equal(south_pole[:, 2], -1.0)

    equator = SphericalPoints.from_lonlats(
        np.stack([lat, np.repeat(0.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(equator[:, 2], 0.0)


def test_to_lonlat():
    lonlat = SphericalPoint(np.array([0.0, 0.0, 1.0])).to_lonlat()
    assert_almost_equal(lonlat[1], 90)

    lonlat = SphericalPoint(np.array([0.0, 0.0, -1.0])).to_lonlat()
    assert_almost_equal(lonlat[1], -90)

    lonlat = SphericalPoint(np.array([1.0, 1.0, 0.0])).to_lonlat()
    assert_almost_equal(lonlat, np.array([45.0, 0.0]))

    lonlat = SphericalPoint(np.array([1.0, -1.0, 0.0])).to_lonlat()
    assert_almost_equal(lonlat, np.array([315.0, 0.0]))
