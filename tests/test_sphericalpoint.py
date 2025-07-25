import numpy as np
from numpy.testing import assert_allclose
from sphersgeo import MultiSphericalPoint, SphericalPoint
import sphersgeo
import math


def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    a = tuple(map(math.radians, a))
    b = tuple(map(math.radians, b))

    distance = 2 * math.asin(
        math.sqrt(
            (
                math.sin((b[1] - a[1]) / 2) ** 2
                + math.cos(a[1]) * math.cos(b[1]) * math.sin((b[0] - a[0]) / 2) ** 2
            )
        )
    )

    return math.degrees(distance)


def test_init():
    vectors = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]

    single_from_tuple = SphericalPoint(vectors[0])
    single_from_list = SphericalPoint(list(vectors[0]))
    single_from_numpy = SphericalPoint(np.array(vectors[0]))

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_numpy
    assert single_from_list == single_from_numpy

    assert SphericalPoint(single_from_numpy) == single_from_numpy

    multi_from_list_of_tuples = MultiSphericalPoint(vectors)
    multi_from_nested_list = MultiSphericalPoint([list(vector) for vector in vectors])
    multi_from_numpy = MultiSphericalPoint(np.array(vectors))

    assert multi_from_list_of_tuples == multi_from_nested_list
    assert multi_from_list_of_tuples == multi_from_numpy

    assert MultiSphericalPoint(multi_from_numpy) == multi_from_numpy


def test_vectors_lengths():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = MultiSphericalPoint(xyz)

    assert_allclose(points.vectors_lengths, np.sqrt(np.sum(points.xyzs**2, axis=-1)))

    # also test if normalized
    assert_allclose(points.vectors_lengths, 1.0)


def test_already_normalized():
    for xyz in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        assert SphericalPoint(xyz).vector_length == 1.0


def test_from_lonlat():
    tolerance = 3e-11

    a_lonlat = (60.0, 0.0)
    b_lonlat = (60.0, 30.0)

    a = SphericalPoint.from_lonlat(a_lonlat)
    b = SphericalPoint.from_lonlat(b_lonlat)

    assert_allclose(a.to_lonlat(), a_lonlat)
    assert_allclose(b.to_lonlat(), b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [SphericalPoint.from_lonlat((lon, equator_lat)) for lon in lons]
    for equator in equators:
        assert equator.to_lonlat()[1] == 0.0

    multi_equator = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(equator_lat, len(lons))], axis=1)
    )

    for point in equators:
        assert point.within(multi_equator)
    assert_allclose(multi_equator.xyzs[:, 2], 0.0)

    north_pole_lat = 90.0
    north_poles = [SphericalPoint.from_lonlat((lon, north_pole_lat)) for lon in lons]
    for north_pole in north_poles:
        assert_allclose(north_pole.xyz, [0.0, 0.0, 1.0], atol=tolerance)

    multi_north_pole = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(north_pole_lat, len(lons))], axis=1)
    )

    for point in north_poles:
        assert point.within(multi_north_pole)
    assert_allclose(
        multi_north_pole.xyzs,
        np.repeat([(0.0, 0.0, 1.0)], len(multi_north_pole), axis=0),
        atol=tolerance,
    )

    south_pole_lat = -90.0
    south_poles = [SphericalPoint.from_lonlat((lon, south_pole_lat)) for lon in lons]
    for south_pole in south_poles:
        assert_allclose(south_pole.xyz, [0.0, 0.0, -1.0], atol=tolerance)

    multi_south_pole = MultiSphericalPoint.from_lonlats(
        np.stack([lons, np.repeat(south_pole_lat, len(lons))], axis=1)
    )

    for point in south_poles:
        assert point.within(multi_south_pole)
    assert_allclose(
        multi_south_pole.xyzs,
        np.repeat([[0.0, 0.0, -1.0]], len(multi_south_pole), axis=0),
        atol=tolerance,
    )


def test_to_lonlat():
    xyzs = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    lonlats = [(0, 90), (0, -90), (45, 0), (315, 0), (np.nan, 0)]

    a = SphericalPoint(xyzs[0])
    assert_allclose(a.to_lonlat(), lonlats[0])

    b = SphericalPoint(xyzs[1])
    assert_allclose(b.to_lonlat(), lonlats[1])

    c = SphericalPoint(xyzs[2])
    assert_allclose(c.to_lonlat(), lonlats[2])

    d = SphericalPoint(xyzs[3])
    assert_allclose(d.to_lonlat(), lonlats[3])

    e = SphericalPoint(xyzs[4])
    assert_allclose(e.to_lonlat(), lonlats[4])

    abcde = MultiSphericalPoint(xyzs)
    assert_allclose(abcde.to_lonlats(), lonlats)


def test_xyz_radians_over_sphere_between():
    a = np.array((0.0, 0.0, 1.0))
    b = np.array((0.0, 0.0, -1.0))
    c = np.array((1.0, 1.0, 0.0))
    d = np.array((1.0, -1.0, 0.0))

    assert sphersgeo.array.xyz_radians_over_sphere_between(a, b) == np.acos(a.dot(b))
    assert sphersgeo.array.xyz_radians_over_sphere_between(b, c) == np.acos(b.dot(c))
    assert sphersgeo.array.xyz_radians_over_sphere_between(c, d) == np.acos(c.dot(d))
    assert sphersgeo.array.xyz_radians_over_sphere_between(d, a) == np.acos(d.dot(a))


def test_distance():
    # lonlats=[[  0.  90.],     [  0. -90.],      [ 45.   0.],     [315.   0.]]
    xyzs = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)]

    a = SphericalPoint(xyzs[0])
    b = SphericalPoint(xyzs[1])
    c = SphericalPoint(xyzs[2])
    d = SphericalPoint(xyzs[3])

    ab = MultiSphericalPoint(xyzs[:2])
    bc = MultiSphericalPoint(xyzs[1:3])
    cd = MultiSphericalPoint(xyzs[2:])

    assert a.distance(b) == 180.0
    assert b.distance(c) == 90.0
    assert c.distance(d) == 90.0
    assert d.distance(a) == 90.0
    assert c.distance(a) == 90.0
    assert d.distance(b) == 90.0

    assert a.distance(a) == 0.0
    assert b.distance(b) == 0.0
    assert c.distance(c) == 0.0
    assert d.distance(d) == 0.0

    assert ab.distance(bc) == 0.0
    assert bc.distance(cd) == 0.0
    assert ab.distance(cd) == 90.0

    assert bc.distance(c) == 0.0
    assert ab.distance(d) == 90.0

    assert ab.distance(ab) == 0.0
    assert bc.distance(bc) == 0.0
    assert cd.distance(cd) == 0.0

    A = SphericalPoint.from_lonlat((90.0, 0.0))
    B = SphericalPoint.from_lonlat((-90.0, 0.0))
    assert A.distance(B) == 180.0

    A = SphericalPoint.from_lonlat((135.0, 0.0))
    B = SphericalPoint.from_lonlat((-90.0, 0.0))
    assert A.distance(B) == np.rad2deg((3.0 / 4.0) * np.pi)

    A = SphericalPoint.from_lonlat((0.0, 0.0))
    B = SphericalPoint.from_lonlat((0.0, 90.0))
    assert A.distance(B) == np.rad2deg(np.pi / 2.0)


def test_distance_domain():
    A = SphericalPoint((np.nan, 0.0, 0.0))
    B = SphericalPoint((0.0, 0.0, np.inf))
    assert np.isnan(A.distance(B))

    A = MultiSphericalPoint(
        [
            (np.nan, 0, 0),
            (np.nan, 0, 0),
            (np.nan, np.nan, np.nan),
            (0, 0, np.nan),
        ]
    )
    B = MultiSphericalPoint(
        [
            (0, 0, np.inf),
            (0, 0, np.inf),
            (np.inf, np.inf, np.inf),
            (0, 0, np.inf),
            (0, 0, 0),
        ]
    )

    assert np.isnan(A.distance(B))


def test_contains():
    xyz = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)]

    a = SphericalPoint(xyz[0])
    b = SphericalPoint(xyz[1])
    c = SphericalPoint(xyz[2])
    d = SphericalPoint(xyz[3])

    abc = MultiSphericalPoint(xyz[:3])

    assert abc.contains(a)
    assert abc.contains(b)
    assert abc.contains(c)
    assert not abc.contains(d)

    assert a.within(abc)
    assert b.within(abc)
    assert c.within(abc)
    assert not d.within(abc)


def test_str():
    assert (
        str(SphericalPoint((0.0, 1.0, 2.0)))
        == "SphericalPoint([0.0, 0.4472135954999579, 0.8944271909999159])"
    )
    assert (
        str(MultiSphericalPoint([(0.0, 1.0, 2.0)]))
        == "MultiSphericalPoint([[0.0, 0.4472135954999579, 0.8944271909999159]])"
    )


def test_add():
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

    # operations between points are elementwise
    assert a + b == SphericalPoint((0, 0, 0))
    assert b + c == SphericalPoint((0.7071067811865475, 0.7071067811865475, -1.0))
    assert c + d == SphericalPoint((1.414213, 0.0, 0.0))
    assert d + a == SphericalPoint((0.7071067811865475, -0.7071067811865475, 1.0))

    a += b
    c += d

    assert a == SphericalPoint((0, 0, 0))
    assert c == SphericalPoint((1.414213, 0.0, 0.0))

    # adding between multipoints is concatenation
    ab = MultiSphericalPoint(xyz[0:2])
    cd = MultiSphericalPoint(xyz[2:4])

    abcd = MultiSphericalPoint(xyz)

    assert ab + cd == abcd


def test_two_arc_angle():
    # right angle
    A = SphericalPoint((1.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 1.0))
    assert B.two_arc_angle(A, C) == np.rad2deg(np.pi / 2)
    assert B.two_arc_angle(C, A) == np.rad2deg(np.pi / 2)

    # antipodes
    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((-1.0, -1.0, -1.0))
    assert B.two_arc_angle(A, C) == np.rad2deg(np.pi)
    assert B.two_arc_angle(C, A) == np.rad2deg(np.pi)

    # same point
    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((1.0, 1.0, 1.0))
    assert B.two_arc_angle(A, C) == 0.0
    assert B.two_arc_angle(C, A) == 0.0

    # defined from lonlat
    A = SphericalPoint.from_lonlat((60.0, 45.0))
    B = SphericalPoint.from_lonlat((0.0, 90.0))
    C = SphericalPoint.from_lonlat((30.0, -3.0))
    assert_allclose(B.two_arc_angle(A, C), 30.0)
    assert_allclose(B.two_arc_angle(C, A), 30.0)

    # TODO: More angle tests


def test_angle_domain():
    A = SphericalPoint((0.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 0.0))
    assert np.isnan(B.two_arc_angle(A, C))

    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((-1.0, -1.0, -1.0))
    assert np.isnan(B.two_arc_angle(A, C))

    A = SphericalPoint((0.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((1.0, 0.0, 0.0))
    assert np.isnan(B.two_arc_angle(A, C))

    A = SphericalPoint((0.0, 0.0, 0.0))
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((0.0, 0.0, 0.0))
    assert np.isnan(B.two_arc_angle(A, C))


def test_angle_nearly_coplanar():
    # test from issue #222 + extra values
    a = SphericalPoint((1.0, 1.0, 1.0))
    b = SphericalPoint((1.0, 0.9999999, 1.0))
    C = MultiSphericalPoint(
        [
            (0.0, 0.5, 1.0),
            (0.0, 0.15, 1.0),
            (0.0, 0.001, 1.0),
            (-1.0, -1.0, -1.0),
            (-1.0, 0.1, -1.0),
        ]
    )
    angles = [b.two_arc_angle(a, c) for c in C.parts]

    assert np.isfinite(angles[1:3]).all()

    assert_allclose(angles[0], np.rad2deg(np.pi / 2))
    assert_allclose(angles[3], np.rad2deg(np.pi / 2))
    assert_allclose(angles[4], np.rad2deg(np.pi))


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
    B = SphericalPoint((0.0, 1.0, 0.0))
    C = SphericalPoint((-1.0, -1.0, -1.0))
    assert B.collinear(A, C)

    # points that equal each other
    A = SphericalPoint((1.0, 1.0, 1.0))
    B = SphericalPoint((0.0, 0.0, 0.0))
    C = SphericalPoint((1.0, 1.0, 1.0))
    assert B.collinear(A, C)
