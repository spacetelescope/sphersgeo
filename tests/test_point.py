import numpy as np
from numpy.testing import assert_almost_equal
from sphersgeo import VectorPoint, MultiVectorPoint


def test_normalize():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiVectorPoint(xyz)

    assert np.all(points.vectorlengths != 1.0)

    normalized = points.normalized

    assert_almost_equal(normalized.vectorlengths, 1.0)

    assert_almost_equal(np.sqrt(np.sum(normalized.xyz**2, axis=-1)), 1.0)


def test_already_normalized():
    for i in range(3):
        xyz = np.array([0.0, 0.0, 0.0])
        xyz[i] = 1.0
        normalized = VectorPoint(xyz).normalized.xyz
        length = np.sqrt(np.sum(normalized**2, axis=-1))
        assert_almost_equal(length, 1.0)


def test_from_lonlat():
    lat = np.arange(-360.0, 360.0, 1.0)

    north_pole = MultiVectorPoint.from_lonlats(
        np.stack([lat, np.repeat(90.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(north_pole[:, 0], 0.0)
    assert_almost_equal(north_pole[:, 1], 0.0)
    assert_almost_equal(north_pole[:, 2], 1.0)

    south_pole = MultiVectorPoint.from_lonlats(
        np.stack([lat, np.repeat(-90.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(south_pole[:, 0], 0.0)
    assert_almost_equal(south_pole[:, 1], 0.0)
    assert_almost_equal(south_pole[:, 2], -1.0)

    equator = MultiVectorPoint.from_lonlats(
        np.stack([lat, np.repeat(0.0, len(lat))], axis=1)
    ).xyz
    assert_almost_equal(equator[:, 2], 0.0)


def test_to_lonlats():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    lonlats = np.array([[0, 90], [0, -90], [45, 0], [315, 0]])

    a = VectorPoint(xyz[0])
    assert_almost_equal(a.to_lonlat(), lonlats[0])

    b = VectorPoint(xyz[1])
    assert_almost_equal(b.to_lonlat(), lonlats[1])

    c = VectorPoint(xyz[2])
    assert_almost_equal(c.to_lonlat(), lonlats[2])

    d = VectorPoint(xyz[3])
    assert_almost_equal(d.to_lonlat(), lonlats[3])

    abcd = MultiVectorPoint(xyz)
    assert_almost_equal(abcd.to_lonlats(), lonlats)


def test_str():
    assert str(VectorPoint(np.array([0.0, 1.0, 2.0]))) == "VectorPoint([0, 1, 2])"


def test_add():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    a = VectorPoint(xyz[0])
    b = VectorPoint(xyz[1])
    c = VectorPoint(xyz[2])
    d = VectorPoint(xyz[3])

    ab = MultiVectorPoint(xyz[0:2])
    bc = MultiVectorPoint(xyz[1:3])
    cd = MultiVectorPoint(xyz[2:4])
    da = MultiVectorPoint(np.stack([xyz[-1], xyz[0]]))

    assert a + b == ab
    assert b + c == bc
    assert c + d == cd
    assert d + a == da
