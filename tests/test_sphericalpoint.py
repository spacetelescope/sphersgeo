import numpy as np
import sphersgeo
from numpy.testing import assert_allclose


def test_init():
    vectors = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]

    single_from_tuple = sphersgeo.SphericalPoint(vectors[0])
    single_from_list = sphersgeo.SphericalPoint(list(vectors[0]))
    single_from_numpy = sphersgeo.SphericalPoint(np.array(vectors[0]))

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_numpy
    assert single_from_list == single_from_numpy

    assert sphersgeo.SphericalPoint(single_from_numpy) == single_from_numpy

    multi_from_list_of_tuples = sphersgeo.MultiSphericalPoint(vectors)
    multi_from_nested_list = sphersgeo.MultiSphericalPoint(
        [list(vector) for vector in vectors]
    )
    multi_from_numpy = sphersgeo.MultiSphericalPoint(np.array(vectors))

    assert multi_from_list_of_tuples == multi_from_nested_list
    assert multi_from_list_of_tuples == multi_from_numpy

    assert sphersgeo.MultiSphericalPoint(multi_from_numpy) == multi_from_numpy


def test_vectors_lengths():
    x, y, z = np.ogrid[-100:100:11, -100:100:11, -100:100:11]
    xyz = np.dstack((x.flatten(), y.flatten(), z.flatten()))[0].astype(float)
    points = sphersgeo.MultiSphericalPoint(xyz)

    assert_allclose(points.vectors_lengths, np.sqrt(np.sum(points.xyzs**2, axis=-1)))

    # also test if normalized
    assert_allclose(points.vectors_lengths, 1.0)


def test_already_normalized():
    for xyz in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        assert sphersgeo.SphericalPoint(xyz).vector_length == 1.0


def test_from_lonlat():
    tolerance = 3e-11

    a_lonlat = (60.0, 0.0)
    b_lonlat = (60.0, 30.0)

    a = sphersgeo.SphericalPoint(a_lonlat)
    b = sphersgeo.SphericalPoint(b_lonlat)

    assert_allclose(a.lonlat, a_lonlat)
    assert_allclose(b.lonlat, b_lonlat)

    lons = np.arange(-360.0, 360.0, 1.0)

    equator_lat = 0.0
    equators = [sphersgeo.SphericalPoint((lon, equator_lat)) for lon in lons]
    for equator in equators:
        assert equator.lonlat[1] == 0.0

    multi_equator = sphersgeo.MultiSphericalPoint(
        np.stack([lons, np.repeat(equator_lat, len(lons))], axis=1)
    )

    for point in equators:
        assert point.within(multi_equator)
    assert_allclose(multi_equator.xyzs[:, 2], 0.0)

    north_pole_lat = 90.0
    north_poles = [sphersgeo.SphericalPoint((lon, north_pole_lat)) for lon in lons]
    for north_pole in north_poles:
        assert_allclose(north_pole.xyz, [0.0, 0.0, 1.0], atol=tolerance)

    multi_north_pole = sphersgeo.MultiSphericalPoint(
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
    south_poles = [sphersgeo.SphericalPoint((lon, south_pole_lat)) for lon in lons]
    for south_pole in south_poles:
        assert_allclose(south_pole.xyz, [0.0, 0.0, -1.0], atol=tolerance)

    multi_south_pole = sphersgeo.MultiSphericalPoint(
        np.stack([lons, np.repeat(south_pole_lat, len(lons))], axis=1)
    )

    for point in south_poles:
        assert point.within(multi_south_pole)
    assert_allclose(
        multi_south_pole.xyzs,
        np.repeat([[0.0, 0.0, -1.0]], len(multi_south_pole), axis=0),
        atol=tolerance,
    )


def test_lonlat():
    xyzs = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    lonlats = [(0, 90), (0, -90), (45, 0), (315, 0), (np.nan, 0)]

    a = sphersgeo.SphericalPoint(xyzs[0])
    assert_allclose(a.lonlat, lonlats[0])

    b = sphersgeo.SphericalPoint(xyzs[1])
    assert_allclose(b.lonlat, lonlats[1])

    c = sphersgeo.SphericalPoint(xyzs[2])
    assert_allclose(c.lonlat, lonlats[2])

    d = sphersgeo.SphericalPoint(xyzs[3])
    assert_allclose(d.lonlat, lonlats[3])

    e = sphersgeo.SphericalPoint(xyzs[4])
    assert_allclose(e.lonlat, lonlats[4])

    abcde = sphersgeo.MultiSphericalPoint(xyzs)
    assert_allclose(abcde.lonlats, lonlats)


def test_distance_domain():
    A = sphersgeo.SphericalPoint((np.nan, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0, np.inf))
    assert np.isnan(A.distance(B))

    A = sphersgeo.MultiSphericalPoint(
        [
            (np.nan, 0, 0),
            (np.nan, 0, 0),
            (np.nan, np.nan, np.nan),
            (0, 0, np.nan),
        ]
    )
    B = sphersgeo.MultiSphericalPoint(
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

    a = sphersgeo.SphericalPoint(xyz[0])
    b = sphersgeo.SphericalPoint(xyz[1])
    c = sphersgeo.SphericalPoint(xyz[2])
    d = sphersgeo.SphericalPoint(xyz[3])

    abc = sphersgeo.MultiSphericalPoint(xyz[:3])

    assert abc.contains(a)
    assert abc.contains(b)
    assert abc.contains(c)
    assert not abc.contains(d)

    assert a.within(abc)
    assert b.within(abc)
    assert c.within(abc)
    assert not d.within(abc)


def test_wkt():
    geometries = [
        (
            sphersgeo.SphericalPoint((0.0, 1.0, 2.0)),
            "POINT (90 63.43494882292201)",
        ),
        (
            sphersgeo.MultiSphericalPoint([(0.0, 1.0, 2.0), (0.0, 0.0, 1.0)]),
            "MULTIPOINT (90 63.43494882292201, 0 90)",
        ),
    ]

    for geometry, wkt in geometries:
        assert geometry.wkt == wkt
        assert geometry.__class__(wkt) == geometry


def test_add():
    xyz = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]

    a = sphersgeo.SphericalPoint(xyz[0])
    b = sphersgeo.SphericalPoint(xyz[1])
    c = sphersgeo.SphericalPoint(xyz[2])
    d = sphersgeo.SphericalPoint(xyz[3])

    # operations between points are elementwise
    assert a + b == sphersgeo.SphericalPoint((0, 0, 0))
    assert b + c == sphersgeo.SphericalPoint(
        (0.7071067811865475, 0.7071067811865475, -1.0)
    )
    assert c + d == sphersgeo.SphericalPoint((1.414213, 0.0, 0.0))
    assert d + a == sphersgeo.SphericalPoint(
        (0.7071067811865475, -0.7071067811865475, 1.0)
    )

    a += b
    c += d

    assert a == sphersgeo.SphericalPoint((0, 0, 0))
    assert c == sphersgeo.SphericalPoint((1.414213, 0.0, 0.0))

    # adding between multipoints is concatenation
    ab = sphersgeo.MultiSphericalPoint(xyz[0:2])
    cd = sphersgeo.MultiSphericalPoint(xyz[2:4])

    abcd = sphersgeo.MultiSphericalPoint(xyz)

    assert ab + cd == abcd


def test_two_arc_angle():
    # right angle
    A = sphersgeo.SphericalPoint((1.0, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((0.0, 0.0, 1.0))
    assert A.two_arc_angle(B, C) == np.rad2deg(np.pi / 2)
    assert B.two_arc_angle(A, C) == np.rad2deg(np.pi / 2)
    assert C.two_arc_angle(A, B) == np.rad2deg(np.pi / 2)

    # antipodes
    A = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((-1.0, -1.0, -1.0))
    assert B.two_arc_angle(A, C) == np.rad2deg(np.pi)
    assert B.two_arc_angle(C, A) == np.rad2deg(np.pi)

    # same point
    A = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    assert B.two_arc_angle(A, C) == 0.0
    assert B.two_arc_angle(C, A) == 0.0

    # defined from lonlat
    A = sphersgeo.SphericalPoint((60.0, 45.0))
    B = sphersgeo.SphericalPoint((0.0, 90.0))
    C = sphersgeo.SphericalPoint((30.0, -3.0))
    assert_allclose(B.two_arc_angle(A, C), 30.0)
    assert_allclose(B.two_arc_angle(C, A), 30.0)

    # equatorial
    A = sphersgeo.SphericalPoint((0.0, 0.0))
    B = sphersgeo.SphericalPoint((15.0, 0.0))
    C = sphersgeo.SphericalPoint((30.0, 0.0))
    assert B.two_arc_angle(A, C) == 180

    A = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    C = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    assert B.two_arc_angle(A, C) == 0

    A = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    C = sphersgeo.SphericalPoint((-1.0, -1.0, -1.0))
    assert B.two_arc_angle(A, C) == 180

    A = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((1.0, 0.0, 0.0))
    assert B.two_arc_angle(A, C) == 90

    A = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    assert B.two_arc_angle(A, C) == 0


def test_angle_nearly_coplanar():
    # test from issue #222 + extra values
    a = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))  # [45.0, 35.264389682754654]
    b = sphersgeo.SphericalPoint(
        (1.0, 0.9999999, 1.0)
    )  # [44.99999713521089, 35.26439103322914]
    C = sphersgeo.MultiSphericalPoint(
        [
            (0.0, 0.5, 1.0),  # [ 90., 63.43494882]
            (0.0, 0.15, 1.0),  # [ 90., 81.46923439]
            (0.0, 0.001, 1.0),  # [ 90., 89.94270424]
            (-1.0, -1.0, -1.0),  # [225., -35.26438968]
            (-1.0, 0.1, -1.0),  # [174.28940686, -44.8574726 ]
        ]
    )
    angles = [b.two_arc_angle(a, c) for c in C.parts]

    assert np.isfinite(angles[1:3]).all()

    assert_allclose(angles[0], 90)
    assert_allclose(angles[3], 180)
    assert_allclose(angles[4], 0)


def test_colinear():
    # equatorial
    A = sphersgeo.SphericalPoint((20.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0))
    C = sphersgeo.SphericalPoint((-20.0, 0.0))
    assert A.colinear(B, C)
    assert B.colinear(A, C)
    assert C.colinear(A, B)

    # meridianal
    A = sphersgeo.SphericalPoint((0.0, 20.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0))
    C = sphersgeo.SphericalPoint((0.0, -20.0))
    assert A.colinear(B, C)
    assert B.colinear(A, C)
    assert C.colinear(A, B)

    # non-colinear points
    A = sphersgeo.SphericalPoint((1.0, 0.0, 0.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((0.0, 0.0, 1.0))
    assert not B.colinear(A, C)

    # mirrored
    A = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    B = sphersgeo.SphericalPoint((0.0, 1.0, 0.0))
    C = sphersgeo.SphericalPoint((-1.0, -1.0, -1.0))
    assert B.colinear(A, C)

    # points that equal each other
    A = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    B = sphersgeo.SphericalPoint((0.0, 0.0, 0.0))
    C = sphersgeo.SphericalPoint((1.0, 1.0, 1.0))
    assert B.colinear(A, C)
