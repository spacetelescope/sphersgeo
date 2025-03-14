import numpy as np
from numpy.testing import assert_allclose
from sphersgeo import MultiSphericalPoint, SphericalPoint
import sphersgeo


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

    assert SphericalPoint(single_from_array) == single_from_array

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

    # TODO: figure out error with collapse_axis: Index 4 must be less than axis length 4 for array with shape [4, 3]
    # assert MultiSphericalPoint(multi_from_array) == multi_from_array


def test_normalized():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiSphericalPoint(xyz)

    assert np.all(points.vector_lengths != 1.0)

    normalized = points.normalized

    assert_allclose(normalized.vector_lengths, 1.0)

    assert_allclose(np.sqrt(np.sum(normalized.xyz**2, axis=-1)), 1.0)


def test_already_normalized():
    for xyz in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        pre_normalized = SphericalPoint.normalize(xyz)
        assert_allclose(pre_normalized.vector_length, 1.0)


def test_from_lonlat():
    tolerance = 3e-11

    a_lonlat = np.array((60.0, 0.0))
    b_lonlat = np.array((60.0, 30.0))

    a = SphericalPoint.from_lonlat(a_lonlat, degrees=True)
    b = SphericalPoint.from_lonlat(b_lonlat, degrees=True)

    assert_allclose(a.to_lonlat(degrees=True), a_lonlat)
    assert_allclose(b.to_lonlat(degrees=True), b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [
        SphericalPoint.from_lonlat((lon, equator_lat), degrees=True) for lon in lons
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
        SphericalPoint.from_lonlat((lon, north_pole_lat), degrees=True) for lon in lons
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
        SphericalPoint.from_lonlat((lon, south_pole_lat), degrees=True) for lon in lons
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


def test_to_lonlat():
    xyz = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]

    lonlats = [(0, 90), (0, -90), (45, 0), (315, 0), (np.nan, 0)]

    a = SphericalPoint(xyz[0])
    assert_allclose(a.to_lonlat(degrees=True), lonlats[0])

    b = SphericalPoint(xyz[1])
    assert_allclose(b.to_lonlat(degrees=True), lonlats[1])

    c = SphericalPoint(xyz[2])
    assert_allclose(c.to_lonlat(degrees=True), lonlats[2])

    d = SphericalPoint(xyz[3])
    assert_allclose(d.to_lonlat(degrees=True), lonlats[3])

    e = SphericalPoint(xyz[4])
    assert_allclose(e.to_lonlat(degrees=True), lonlats[4])

    abcde = MultiSphericalPoint(xyz)
    assert_allclose(abcde.to_lonlats(degrees=True), lonlats)


def test_vector_arc_length():
    a = np.array((0.0, 0.0, 1.0))
    b = np.array((0.0, 0.0, -1.0))
    c = np.array((1.0, 1.0, 0.0))
    d = np.array((1.0, -1.0, 0.0))

    assert sphersgeo.array.vector_arc_length(a, b) == np.acos(a.dot(b))
    assert sphersgeo.array.vector_arc_length(b, c) == np.acos(b.dot(c))
    assert sphersgeo.array.vector_arc_length(c, d) == np.acos(c.dot(d))
    assert sphersgeo.array.vector_arc_length(d, a) == np.acos(d.dot(a))


def test_distance():
    tolerance = 3e-8

    xyz = np.array(
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
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
    assert b.distance(b) == 0.0
    assert c.distance(c) == 0.0
    assert d.distance(d) == 0.0

    assert_allclose(ab.distance(bc), 0.0, atol=tolerance)
    assert_allclose(bc.distance(cd), 0.0, atol=tolerance)
    assert_allclose(ab.distance(cd), np.pi / 2.0, atol=tolerance)

    assert_allclose(bc.distance(c), 0.0, atol=tolerance)
    assert_allclose(ab.distance(d), np.pi / 2.0, atol=tolerance)

    assert_allclose(ab.distance(ab), 0.0, atol=tolerance)
    assert_allclose(bc.distance(bc), 0.0, atol=tolerance)
    assert_allclose(cd.distance(cd), 0.0, atol=tolerance)

    A = SphericalPoint.from_lonlat((90.0, 0.0), degrees=True)
    B = SphericalPoint.from_lonlat((-90.0, 0.0), degrees=True)
    assert_allclose(A.distance(B), np.pi)

    A = SphericalPoint.from_lonlat((135.0, 0.0), degrees=True)
    B = SphericalPoint.from_lonlat((-90.0, 0.0), degrees=True)
    assert_allclose(A.distance(B), (3.0 / 4.0) * np.pi)

    A = SphericalPoint.from_lonlat((0.0, 0.0), degrees=True)
    B = SphericalPoint.from_lonlat((0.0, 90.0), degrees=True)
    assert_allclose(A.distance(B), np.pi / 2.0)


def test_distance_domain():
    A = SphericalPoint((np.nan, 0.0, 0.0))
    B = SphericalPoint((0.0, 0.0, np.inf))
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


def test_contains():
    xyz = np.array(
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
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
    assert str(SphericalPoint((0.0, 1.0, 2.0))) == "SphericalPoint([0, 1, 2])"
    assert str(MultiSphericalPoint([(0.0, 1.0, 2.0)])) == "MultiSphericalPoint([[0, 1, 2]])"


def test_add_combine_extend():
    xyz = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]

    a = SphericalPoint(xyz[0])
    b = SphericalPoint(xyz[1])
    c = SphericalPoint(xyz[2])
    d = SphericalPoint(xyz[3])

    ab = MultiSphericalPoint(xyz[0:2])
    bc = MultiSphericalPoint(xyz[1:3])
    cd = MultiSphericalPoint(xyz[2:4])
    da = MultiSphericalPoint([xyz[-1]] + [xyz[0]])

    abcd = MultiSphericalPoint(xyz)

    assert a + b == SphericalPoint((0, 0, 0))
    assert b + c == SphericalPoint((1, 1, -1))
    assert c + d == SphericalPoint((2, 0, 0))
    assert d + a == SphericalPoint((1, -1, 1))

    assert a.combine(b) == ab
    assert b.combine(c) == bc
    assert c.combine(d) == cd
    assert d.combine(a) == da

    a += b
    c += d

    assert a == SphericalPoint((0, 0, 0))
    assert c == SphericalPoint((2, 0, 0))

    assert ab + cd == abcd


def test_angle():
    A = SphericalPoint((1.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 1.0))
    assert B.angle_between(A, C, degrees=False) == np.pi / 2

    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((-1.0, -1.0, -1.0))
    assert B.angle_between(A, C, degrees=False) == np.pi

    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((1.0, 1.0, 1.0))
    assert B.angle_between(A, C, degrees=False) == 0.0

    A = SphericalPoint.from_lonlat((60.0, 45.0))
    B = SphericalPoint.from_lonlat((0.0, 90.0))
    C = SphericalPoint.from_lonlat((30.0, -3.0))
    assert_allclose(B.angle_between(A, C, degrees=True), 30.0)

    # TODO: More angle tests


def test_angle_domain():
    A = SphericalPoint((0.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 0.0))
    assert A.angle_between(B, C, degrees=False) == 0


def test_angle_nearly_coplanar():
    # test from issue #222 + extra values
    A = MultiSphericalPoint(np.repeat([(1.0, 1.0, 1.0)], 5, axis=0))
    B = MultiSphericalPoint(np.repeat([(1.0, 0.9999999, 1.0)], 5, axis=0))
    C = MultiSphericalPoint(
        [
            (0.0, 0.5, 1.0),
            (0.0, 0.15, 1.0),
            (0.0, 0.001, 1.0),
            (-1.0, -1.0, -1.0),
            (-1.0, 0.1, -1.0),
        ]
    )
    angles = B.angles_between(A, C, degrees=False)

    assert_allclose(angles[0], np.pi / 2)
    assert np.isfinite(angles[1:3]).all()
    assert_allclose(angles[3], np.pi / 2)
    assert_allclose(angles[4], np.pi)


def test_collinear():
    # equatorial
    A = SphericalPoint.from_lonlat((0.0, 0.0))
    B = SphericalPoint.from_lonlat((20.0, 0.0))
    C = SphericalPoint.from_lonlat((-20.0, 0.0))
    assert B.collinear(A, C)

    # meridianal
    A = SphericalPoint.from_lonlat((0.0, 20.0))
    B = SphericalPoint.from_lonlat((0.0, -20.0))
    C = SphericalPoint.from_lonlat((0.0, 0.0))
    assert B.collinear(A, C)

    # non-collinear points
    A = SphericalPoint((1.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 1.0))
    assert not B.collinear(A, C)

    # mirrored
    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((-1.0, -1.0, -1.0))
    assert B.collinear(A, C)

    # points that equal each other
    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((1.0, 1.0, 1.0))
    assert B.collinear(A, C)
