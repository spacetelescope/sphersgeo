import numpy as np
from numpy.testing import assert_almost_equal
from sphersgeo import MultiVectorPoint, VectorPoint


def test_normalize():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiVectorPoint(xyz)

    assert np.all(points.vector_lengths != 1.0)

    normalized = points.normalized

    assert_almost_equal(normalized.vector_lengths, 1.0)

    assert_almost_equal(np.sqrt(np.sum(normalized.xyz**2, axis=-1)), 1.0)


def test_already_normalized():
    for i in range(3):
        xyz = np.array([0.0, 0.0, 0.0])
        xyz[i] = 1.0
        normalized = VectorPoint(xyz).normalized.xyz
        length = np.sqrt(np.sum(normalized**2, axis=-1))
        assert_almost_equal(length, 1.0)


def test_from_lonlat():
    a_lonlat = np.array([60.0, 0.0])
    b_lonlat = np.array([60.0, 30.0])

    a = VectorPoint.from_lonlat(a_lonlat)
    b = VectorPoint.from_lonlat(b_lonlat)

    assert_almost_equal(a.to_lonlat(), a_lonlat)
    assert_almost_equal(b.to_lonlat(), b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [VectorPoint.from_lonlat(np.array([lon, equator_lat])) for lon in lons]
    for equator in equators:
        assert_almost_equal(equator.to_lonlat()[1], 0.0)

    multi_equator = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(equator_lat, len(lons))], axis=1)
    )
    assert equators == multi_equator.vector_points
    assert_almost_equal(multi_equator.xyz[:, 2], 0.0)

    north_pole_lat = 90.0
    north_poles = [
        VectorPoint.from_lonlat(np.array([lon, north_pole_lat])) for lon in lons
    ]
    for north_pole in north_poles:
        assert_almost_equal(north_pole.xyz, np.array([0.0, 0.0, 1.0]))

    multi_north_pole = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(north_pole_lat, len(lons))], axis=1)
    )
    assert north_poles == multi_north_pole.vector_points
    assert_almost_equal(
        multi_north_pole.xyz,
        np.repeat([[0.0, 0.0, 1.0]], len(multi_north_pole), axis=0),
    )

    south_pole_lat = -90.0
    south_poles = [
        VectorPoint.from_lonlat(np.array([lon, south_pole_lat])) for lon in lons
    ]
    for south_pole in south_poles:
        assert_almost_equal(south_pole.xyz, np.array([0.0, 0.0, -1.0]))

    multi_south_pole = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(south_pole_lat, len(lons))], axis=1)
    )
    assert south_poles == multi_south_pole.vector_points
    assert_almost_equal(
        multi_south_pole.xyz,
        np.repeat([[0.0, 0.0, -1.0]], len(multi_south_pole), axis=0),
    )


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


def test_distance():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    a = VectorPoint(xyz[0, :])
    b = VectorPoint(xyz[1, :])
    c = VectorPoint(xyz[2, :])
    d = VectorPoint(xyz[3, :])

    ab = MultiVectorPoint(xyz[:2, :])
    bc = MultiVectorPoint(xyz[1:3, :])
    cd = MultiVectorPoint(xyz[2:, :])

    assert a.distance(b) == np.pi
    assert b.distance(c) == np.pi / 2.0
    assert c.distance(d) == np.pi / 2.0

    assert a.distance(a) == 0.0

    assert_almost_equal(ab.distance(bc), 0.0)
    assert_almost_equal(bc.distance(cd), 0.0)
    assert_almost_equal(ab.distance(cd), np.pi / 2.0)


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
