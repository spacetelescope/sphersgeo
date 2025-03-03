import numpy as np
from numpy.testing import assert_allclose
from sphersgeo import MultiVectorPoint, VectorPoint


def test_normalize():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiVectorPoint(xyz)

    assert np.all(points.vector_lengths != 1.0)

    normalized = points.normalized

    assert_allclose(normalized.vector_lengths, 1.0)

    assert_allclose(np.sqrt(np.sum(normalized.xyz**2, axis=-1)), 1.0)


def test_already_normalized():
    for i in range(3):
        xyz = np.array([0.0, 0.0, 0.0])
        xyz[i] = 1.0
        normalized = VectorPoint(xyz).normalized.xyz
        length = np.sqrt(np.sum(normalized**2, axis=-1))
        assert_allclose(length, 1.0)


def test_from_lonlat():
    tolerance = 3e-11

    a_lonlat = np.array([60.0, 0.0])
    b_lonlat = np.array([60.0, 30.0])

    a = VectorPoint.from_lonlat(a_lonlat, degrees=True)
    b = VectorPoint.from_lonlat(b_lonlat, degrees=True)

    assert_allclose(a.to_lonlat(degrees=True), a_lonlat)
    assert_allclose(b.to_lonlat(degrees=True), b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [
        VectorPoint.from_lonlat(np.array([lon, equator_lat]), degrees=True)
        for lon in lons
    ]
    for equator in equators:
        assert_allclose(equator.to_lonlat(degrees=True)[1], 0.0)

    multi_equator = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(equator_lat, len(lons))], axis=1), degrees=True
    )
    assert equators == multi_equator.vector_points
    assert_allclose(multi_equator.xyz[:, 2], 0.0)

    north_pole_lat = 90.0
    north_poles = [
        VectorPoint.from_lonlat(np.array([lon, north_pole_lat]), degrees=True)
        for lon in lons
    ]
    for north_pole in north_poles:
        assert_allclose(north_pole.xyz, [0.0, 0.0, 1.0], atol=tolerance)

    multi_north_pole = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(north_pole_lat, len(lons))], axis=1), degrees=True
    )
    assert north_poles == multi_north_pole.vector_points
    assert_allclose(
        multi_north_pole.xyz,
        np.repeat([[0.0, 0.0, 1.0]], len(multi_north_pole), axis=0),
        atol=tolerance,
    )

    south_pole_lat = -90.0
    south_poles = [
        VectorPoint.from_lonlat(np.array([lon, south_pole_lat]), degrees=True)
        for lon in lons
    ]
    for south_pole in south_poles:
        assert_allclose(south_pole.xyz, [0.0, 0.0, -1.0], atol=tolerance)

    multi_south_pole = MultiVectorPoint.from_lonlats(
        np.stack([lons, np.repeat(south_pole_lat, len(lons))], axis=1), degrees=True
    )
    assert south_poles == multi_south_pole.vector_points
    assert_allclose(
        multi_south_pole.xyz,
        np.repeat([[0.0, 0.0, -1.0]], len(multi_south_pole), axis=0),
        atol=tolerance,
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
    assert_allclose(a.to_lonlat(degrees=True), lonlats[0])

    b = VectorPoint(xyz[1])
    assert_allclose(b.to_lonlat(degrees=True), lonlats[1])

    c = VectorPoint(xyz[2])
    assert_allclose(c.to_lonlat(degrees=True), lonlats[2])

    d = VectorPoint(xyz[3])
    assert_allclose(d.to_lonlat(degrees=True), lonlats[3])

    abcd = MultiVectorPoint(xyz)
    assert_allclose(abcd.to_lonlats(degrees=True), lonlats)


def test_distance():
    tolerance = 3e-8

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

    assert_allclose(ab.distance(bc), 0.0, atol=tolerance)
    assert_allclose(bc.distance(cd), 0.0, atol=tolerance)
    assert_allclose(ab.distance(cd), np.pi / 2.0, atol=tolerance)


def test_contains():
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

    abc = MultiVectorPoint(xyz[:3, :])

    assert abc.contains(a)
    assert abc.contains(b)
    assert abc.contains(c)
    assert not abc.contains(d)

    assert a.within(abc)
    assert b.within(abc)
    assert c.within(abc)
    assert not d.within(abc)


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

    a += b
    c += d

    assert a == ab
    assert c == cd
