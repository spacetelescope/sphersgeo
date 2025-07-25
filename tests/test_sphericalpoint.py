import numpy as np
from numpy.testing import assert_allclose
from sphersgeo import MultiSphericalPoint, SphericalPoint


def test_init():
    vectors = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]

    single_from_array = SphericalPoint(np.array(vectors[0]))
    single_from_tuple = SphericalPoint(tuple(vectors[0]))
    single_from_list = SphericalPoint(vectors[0])

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array

    multi_from_array = MultiSphericalPoint(np.array(vectors))
    multi_from_list_of_tuples = MultiSphericalPoint(
        [tuple(vector) for vector in vectors]
    )
    multi_from_nested_list = MultiSphericalPoint(vectors)
    multi_from_flat_list = MultiSphericalPoint(np.array(vectors).flatten().tolist())

    assert multi_from_list_of_tuples == multi_from_nested_list
    assert multi_from_list_of_tuples == multi_from_flat_list
    assert multi_from_list_of_tuples == multi_from_array
    assert multi_from_flat_list == multi_from_array


def test_normalize():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiSphericalPoint(xyz)

    assert np.all(points.vector_lengths != 1.0)

    normalized = points.normalized

    assert_allclose(normalized.vector_lengths, 1.0)

    assert_allclose(np.sqrt(np.sum(normalized.xyz**2, axis=-1)), 1.0)


def test_already_normalized():
    for i in range(3):
        xyz = np.array([0.0, 0.0, 0.0])
        xyz[i] = 1.0
        normalized = SphericalPoint(xyz).normalized.xyz
        length = np.sqrt(np.sum(normalized**2, axis=-1))
        assert_allclose(length, 1.0)


def test_from_lonlat():
    tolerance = 3e-11

    a_lonlat = np.array([60.0, 0.0])
    b_lonlat = np.array([60.0, 30.0])

    a = SphericalPoint.from_lonlat(a_lonlat, degrees=True)
    b = SphericalPoint.from_lonlat(b_lonlat, degrees=True)

    assert_allclose(a.to_lonlat(degrees=True), a_lonlat)
    assert_allclose(b.to_lonlat(degrees=True), b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [
        SphericalPoint.from_lonlat(np.array([lon, equator_lat]), degrees=True)
        for lon in lons
    ]
    for equator in equators:
        assert_allclose(equator.to_lonlat(degrees=True)[1], 0.0)

    multi_equator = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(equator_lat, len(lons))], axis=1), degrees=True
    )

    for point in equators:
        assert point.within(multi_equator)
    assert_allclose(multi_equator.xyz[:, 2], 0.0)

    north_pole_lat = 90.0
    north_poles = [
        SphericalPoint.from_lonlat(np.array([lon, north_pole_lat]), degrees=True)
        for lon in lons
    ]
    for north_pole in north_poles:
        assert_allclose(north_pole.xyz, [0.0, 0.0, 1.0], atol=tolerance)

    multi_north_pole = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(north_pole_lat, len(lons))], axis=1), degrees=True
    )

    for point in north_poles:
        assert point.within(multi_north_pole)
    assert_allclose(
        multi_north_pole.xyz,
        np.repeat([[0.0, 0.0, 1.0]], len(multi_north_pole), axis=0),
        atol=tolerance,
    )

    south_pole_lat = -90.0
    south_poles = [
        SphericalPoint.from_lonlat(np.array([lon, south_pole_lat]), degrees=True)
        for lon in lons
    ]
    for south_pole in south_poles:
        assert_allclose(south_pole.xyz, [0.0, 0.0, -1.0], atol=tolerance)

    multi_south_pole = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(south_pole_lat, len(lons))], axis=1), degrees=True
    )

    for point in south_poles:
        assert point.within(multi_south_pole)
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

    a = SphericalPoint(xyz[0])
    assert_allclose(a.to_lonlat(degrees=True), lonlats[0])

    b = SphericalPoint(xyz[1])
    assert_allclose(b.to_lonlat(degrees=True), lonlats[1])

    c = SphericalPoint(xyz[2])
    assert_allclose(c.to_lonlat(degrees=True), lonlats[2])

    d = SphericalPoint(xyz[3])
    assert_allclose(d.to_lonlat(degrees=True), lonlats[3])

    abcd = MultiSphericalPoint(xyz)
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

    a = SphericalPoint(xyz[0, :])
    b = SphericalPoint(xyz[1, :])
    c = SphericalPoint(xyz[2, :])
    d = SphericalPoint(xyz[3, :])

    ab = MultiSphericalPoint(xyz[:2, :])
    bc = MultiSphericalPoint(xyz[1:3, :])
    cd = MultiSphericalPoint(xyz[2:, :])

    assert a.distance(b) == np.pi
    assert b.distance(c) == np.pi / 2.0
    assert c.distance(d) == np.pi / 2.0

    assert a.distance(a) == 0.0

    assert_allclose(ab.distance(bc), 0.0, atol=tolerance)
    assert_allclose(bc.distance(cd), 0.0, atol=tolerance)
    assert_allclose(ab.distance(cd), np.pi / 2.0, atol=tolerance)

    A = SphericalPoint.from_lonlat(np.array([90.0, 0.0]), degrees=True)
    B = SphericalPoint.from_lonlat(np.array([-90.0, 0.0]), degrees=True)
    assert_allclose(A.distance(B), np.pi)

    A = SphericalPoint.from_lonlat(np.array([135.0, 0.0]), degrees=True)
    B = SphericalPoint.from_lonlat(np.array([-90.0, 0.0]), degrees=True)
    assert_allclose(A.distance(B), (3.0 / 4.0) * np.pi)

    A = SphericalPoint.from_lonlat(np.array([0.0, 0.0]), degrees=True)
    B = SphericalPoint.from_lonlat(np.array([0.0, 90.0]), degrees=True)
    assert_allclose(A.distance(B), np.pi / 2.0)


def test_contains():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    a = SphericalPoint(xyz[0, :])
    b = SphericalPoint(xyz[1, :])
    c = SphericalPoint(xyz[2, :])
    d = SphericalPoint(xyz[3, :])

    abc = MultiSphericalPoint(xyz[:3, :])

    assert abc.contains(a)
    assert abc.contains(b)
    assert abc.contains(c)
    assert not abc.contains(d)

    assert a.within(abc)
    assert b.within(abc)
    assert c.within(abc)
    assert not d.within(abc)


def test_str():
    assert str(SphericalPoint(np.array([0.0, 1.0, 2.0]))) == "SphericalPoint([0, 1, 2])"


def test_add():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    a = SphericalPoint(xyz[0])
    b = SphericalPoint(xyz[1])
    c = SphericalPoint(xyz[2])
    d = SphericalPoint(xyz[3])

    ab = MultiSphericalPoint(xyz[0:2])
    bc = MultiSphericalPoint(xyz[1:3])
    cd = MultiSphericalPoint(xyz[2:4])
    da = MultiSphericalPoint(np.stack([xyz[-1], xyz[0]]))

    assert a.combine(b) == ab
    assert b.combine(c) == bc
    assert c.combine(d) == cd
    assert d.combine(a) == da

    a += b
    c += d

    assert a == SphericalPoint((0, 0, 0))
    assert c == SphericalPoint((2, 0, 0))


def test_angle():
    A = SphericalPoint(np.array([1.0, 0.0, 0.0]))
    B = SphericalPoint(np.array([0.0, 1.0, 0.0]))
    C = SphericalPoint(np.array([0.0, 0.0, 1.0]))
    assert A.angle_between(B, C, degrees=False) == np.pi / 2

    # TODO: More angle tests


def test_angle_domain():
    A = SphericalPoint(np.array([0.0, 0.0, 0.0]))
    B = SphericalPoint(np.array([0.0, 0.0, 0.0]))
    C = SphericalPoint(np.array([0.0, 0.0, 0.0]))
    assert not np.isfinite(A.angle_between(B, C, degrees=False))


def test_distance_domain():
    A = SphericalPoint(np.array([np.nan, 0.0, 0.0]))
    B = SphericalPoint(np.array([0.0, 0.0, np.inf]))
    assert np.isnan(A.distance(B))

    A = MultiSphericalPoint(
        [
            (np.nan, 0, 0),
            (np.nan, 0, 0),
            (np.nan, np.nan, np.nan),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, np.nan),
            (0, 0, 0),
        ]
    )
    B = MultiSphericalPoint(
        [
            (0, 0, np.inf),
            (0, 0, np.inf),
            (0, 0, 0),
            (np.inf, np.inf, np.inf),
            (0, 0, np.inf),
            (0, 0, 0),
            (0, 0, 0),
        ]
    )

    assert np.isnan(A.distance(B))


def test_angle_nearly_coplanar_vec():
    # test from issue #222 + extra values
    A = MultiSphericalPoint(np.repeat([[1.0, 1.0, 1.0]], 5, axis=0))
    B = MultiSphericalPoint(np.repeat([[1.0, 0.9999999, 1.0]], 5, axis=0))
    C = MultiSphericalPoint(
        np.array(
            [
                [1.0, 0.5, 1.0],
                [1.0, 0.15, 1.0],
                [1.0, 0.001, 1.0],
                [1.0, 0.15, 1.0],
                [-1.0, 0.1, -1.0],
            ]
        )
    )
    # vectors = np.stack([A, B, C], axis=0)
    angles = B.angles_between(A, C, degrees=False)

    assert_allclose(angles[:-1], np.pi, rtol=0, atol=1e-16)
    assert_allclose(angles[-1], 0, rtol=0, atol=1e-32)
