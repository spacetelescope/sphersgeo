import pytest
import numpy as np
from numpy.testing import assert_allclose
from sphersgeo import ArcString, MultiArcString, MultiSphericalPoint, SphericalPoint


def test_midpoint():
    tolerance = 1e-6

    avec = [
        np.array([i, j], dtype=float) + 7.0
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    bvec = [
        np.array([i, j], dtype=float) + 10.0
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    for a in avec:
        A = SphericalPoint(a)
        for b in bvec:
            B = SphericalPoint(b)
            mid = ArcString([A, B]).midpoints[0]
            assert_allclose(A.distance(mid), mid.distance(B), atol=tolerance)
            assert_allclose(mid.two_arc_angle(A, B), 180, rtol=tolerance)


@pytest.mark.parametrize("a", [(0.0, 0.0), (60.0, 0.0), (23.44, 79.9999)])
@pytest.mark.parametrize(
    "b",
    [
        (40.0, 30.0),
        (180.0, 90.0),
        (-30.0, 110.0),
    ],
)
def test_interpolate_points(a, b):
    tolerance = 2e-8

    a = SphericalPoint(a)
    b = SphericalPoint(b)
    ab = ArcString([a, b])

    interpolated_points = a.interpolate_points(b, n=10)

    assert interpolated_points[0] == a
    assert interpolated_points[-1] == b

    for point in interpolated_points[1:-1]:
        assert ab.contains(point)

    interpolated_arc = ArcString(interpolated_points)

    assert_allclose(ab.length, interpolated_arc.length, atol=tolerance)

    assert_allclose(
        interpolated_arc.lengths,
        interpolated_arc.length / len(interpolated_arc),
        atol=tolerance,
    )


TEST_ARCSTRINGS = [
    (
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
        ],
        False,
        6.283185307179586,
        [[np.nan, 0.0], [45.0, -45.0], [0.0, 0.0]],
    ),
    (
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        True,
        7.853981633974483,
        [[np.nan, 0.0], [45.0, -45.0], [0.0, 0.0], [315.0, 45.0]],
    ),
    (
        [
            (0.2, 0.5, 0.7),
            (0.0, 0.5, 0.0),
            (1.0, 1.2, 0.3),
            (4.0, -1.0, 0.0),
        ],
        False,
        2.8146713482124115,
        [
            [81.77235508, 26.60503776],
            [70.28505151, 5.77878756],
            [17.75344926, 6.41019422],
        ],
    ),
    # diagonal arc
    ([(-30.0, -30.0), (30.0, 30.0)], False, 1.4454684956268309, [(0.0, 0.0)]),
    # meridional arc
    ([(60.0, 0.0), (60.0, 30.0)], False, 0.5235987755982985, [(60.0, 15.0)]),
    # equatorial arc
    ([(0.0, 0.0), (30.0, 0.0)], False, 0.5235987755982988, [(15.0, 0.0)]),
]

ids = [
    "arcstring_1",
    "arcstring_2",
    "arcstring_3",
    "diagonal_arc",
    "meridional_arc",
    "equatorial_arc",
]


@pytest.mark.parametrize("arcstring", TEST_ARCSTRINGS, ids=ids)
def test_init(arcstring):
    xyzs = arcstring[0]

    from_list_of_tuples = ArcString(xyzs)
    from_nested_list = ArcString([list(xyz) for xyz in xyzs])
    from_array = ArcString(np.array(xyzs))

    assert from_list_of_tuples == from_nested_list
    assert from_list_of_tuples == from_array
    assert from_nested_list == from_array

    assert ArcString(from_array) == from_array


def test_init_multi():
    arcstrings = [arcstring[0] for arcstring in TEST_ARCSTRINGS]

    from_lists_of_tuples = MultiArcString(arcstrings)
    from_nested_lists = MultiArcString(
        [[list(xyz) for xyz in xyzs] for xyzs in arcstrings]
    )
    from_list_of_arrays = MultiArcString([np.array(xyzs) for xyzs in arcstrings])

    assert from_lists_of_tuples == from_nested_lists
    assert from_lists_of_tuples == from_list_of_arrays

    assert MultiArcString(from_list_of_arrays) == from_list_of_arrays


@pytest.mark.parametrize(
    "geometry,wkt",
    [
        (
            ArcString(TEST_ARCSTRINGS[0][0]),
            "LINESTRING (0 90, 0 -90, 45 0, 315 0)",
        ),
        (
            ArcString(TEST_ARCSTRINGS[1][0]),
            "LINESTRING (0 90, 0 -90, 45 0, 315 0, 0 90)",
        ),
        (
            ArcString(TEST_ARCSTRINGS[2][0]),
            "LINESTRING (68.19859051364818 52.42858277246188, 90 0, 50.19442890773481 10.871582215789932, 345.9637565320735 0)",
        ),
        (
            ArcString(TEST_ARCSTRINGS[3][0]),
            "LINESTRING (330 -29.999999999999993, 29.999999999999996 29.999999999999993)",
        ),
        (
            ArcString(TEST_ARCSTRINGS[4][0]),
            "LINESTRING (59.99999999999999 0, 59.99999999999999 29.999999999999996)",
        ),
        (
            ArcString(TEST_ARCSTRINGS[5][0]),
            "LINESTRING (0 0, 29.999999999999996 0)",
        ),
        (
            MultiArcString([TEST_ARCSTRINGS[1][0], TEST_ARCSTRINGS[2][0]]),
            "MULTILINESTRING ((0 90, 0 -90, 45 0, 315 0, 0 90)), ((68.19859051364818 52.42858277246188, 90 0, 50.19442890773481 10.871582215789932, 345.9637565320735 0))",
        ),
    ],
    ids=ids + [f"{ids[1]}+{ids[2]}"],
)
def test_wkt(geometry, wkt):
    assert geometry.wkt == wkt


@pytest.mark.parametrize("arcstring", TEST_ARCSTRINGS, ids=ids)
def test_closed(arcstring):
    xyzs = arcstring[0]
    closed = arcstring[1]

    assert ArcString(xyzs).closed == closed

    if not closed:
        assert ArcString(xyzs, True).closed
        assert ArcString(xyzs + [xyzs[0]]).closed
    else:
        assert not ArcString(xyzs, False).closed


@pytest.mark.parametrize("arcstring", TEST_ARCSTRINGS, ids=ids)
def test_length(arcstring):
    xyzs = arcstring[0]
    length = arcstring[2]

    assert ArcString(xyzs).length == length


@pytest.mark.parametrize("arcstring", TEST_ARCSTRINGS, ids=ids)
def test_midpoints(arcstring):
    xyzs = arcstring[0]
    midpoints = arcstring[3]

    ArcString(xyzs).midpoints == midpoints


@pytest.mark.parametrize("arcstring", TEST_ARCSTRINGS, ids=ids)
def test_contains(arcstring):
    xyzs = arcstring[0]
    midpoints = arcstring[3]

    arcstring = ArcString(xyzs)

    for midpoint in midpoints:
        assert arcstring.contains(SphericalPoint(midpoint))

    assert arcstring.contains(MultiSphericalPoint(midpoints))


TEST_SEGMENTS = [
    [(20.0, 5.0), (25.0, 5.0)],
    [(25.0, 5.0), (25.0, 6.0)],
    [(25.0, 5.0), (25.0, 6.0), (25.0, 7.0)],
    [(25.0, 6.0), (25.0, 7.0)],
]


@pytest.mark.parametrize(
    "segments,joined,adjoins",
    [
        (
            (TEST_SEGMENTS[0], TEST_SEGMENTS[1]),
            [[(20.0, 5.0), (25.0, 5.0), (25.0, 6.0)]],
            True,
        ),
        (
            (TEST_SEGMENTS[2], TEST_SEGMENTS[3]),
            [[(25.0, 5.0), (25.0, 6.0), (25.0, 7.0)]],
            True,
        ),
        (
            TEST_SEGMENTS[:],
            [[(20.0, 5.0), (25.0, 5.0), (25.0, 6.0), (25.0, 7.0)]],
            False,
        ),
    ],
)
def test_adjoins_union(segments, joined, adjoins):
    segments = MultiArcString(segments)
    assert segments.unary_union == joined
    assert segments[0].adjoins(segments[-1]) == adjoins


TEST_POINTS = [
    (-10.0, -10.0),
    (10.0, 10.0),
    (-25.0, 10.0),
    (15.0, -10.0),
    (-20.0, 40.0),
    (20.0, 40.0),
]


@pytest.mark.parametrize(
    "a,b,intersection",
    [
        (
            ArcString(TEST_POINTS[:2]),
            ArcString(TEST_POINTS[2:4]),
            (358.316743, -1.708471),
        ),
        (ArcString(TEST_POINTS[:2]), ArcString(TEST_POINTS[4:]), None),
        # intersection with later part
        (
            ArcString((TEST_POINTS[0], TEST_POINTS[1], TEST_POINTS[4])),
            ArcString((TEST_POINTS[2], TEST_POINTS[5])),
            (0, 0),
        ),
        # multi-part geometry intersection
        (
            MultiArcString((TEST_POINTS[:2], TEST_POINTS[4:6])),
            ArcString(TEST_POINTS[2:4]),
            [(358.316743, -1.708471)],
        ),
        # ensure non-intersection of non-parallel pre-terminated arcs
        (ArcString((TEST_POINTS[2], TEST_POINTS[4])), ArcString(TEST_POINTS[:2]), None),
        # intersection with non-closed and closed arcstring
        (
            ArcString((TEST_POINTS[3], TEST_POINTS[5], TEST_POINTS[4]), closed=True),
            ArcString(TEST_POINTS[:2]),
            (0, 0),
        ),
        # intersection with self
        (
            ArcString(TEST_POINTS[:2]),
            ArcString(TEST_POINTS[:2]),
            (0, 0),
        ),
    ],
)
def test_intersection(a, b, intersection):
    a = ArcString(a)
    b = ArcString(b)

    assert a.intersects(b) == (intersection is not None)
    assert a.intersection(b) == intersection


def test_closed_not_crosses_self():
    a = ArcString(
        [(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)],
        closed=True,
    )
    b = ArcString(
        [(18.0, 6.0), (21.0, 6.0), (21.0, 7.0), (18.0, 7.0)],
        closed=False,
    )
    c = ArcString(b, closed=True)
    d = ArcString(
        [(18.0, 6.0), (21.0, 7.0), (21.0, 6.0), (18.0, 7.0)],
        closed=False,
    )

    assert a.closed
    assert not b.closed
    assert c.closed
    assert not d.closed

    assert not a.crosses_self
    assert not b.crosses_self
    assert not c.crosses_self
    assert d.crosses_self

    assert a.intersects(b)
    assert a.intersects(c)
    assert a.intersects(d)


@pytest.mark.parametrize("lonlats", [[(90, 0), (0, 45), (0, -45)]])
def test_not_crosses_self(lonlats):
    arcstring = ArcString(lonlats)

    assert not arcstring.crosses_self
    assert arcstring.crossings_with_self is None


def test_crosses_self():
    A = SphericalPoint((-10.0, -10.0))
    B = SphericalPoint((10.0, 10.0))
    C = SphericalPoint((-25.0, 10.0))
    D = SphericalPoint((15.0, -10.0))
    E = SphericalPoint((-20.0, 40.0))
    F = SphericalPoint((20.0, 40.0))

    # simple self-crossing
    ABCD = ArcString([A, B, C, D])
    assert ABCD.crosses_self
    assert_allclose(ABCD.crossings_with_self.lonlats, [(358.316743, -1.708471)])

    # longer self-crossing
    ABCDFE = ArcString([A, B, C, D, F, E])
    assert ABCDFE.crosses_self
    len(ABCDFE.crossings_with_self) == 1
    assert_allclose(ABCDFE.crossings_with_self.lonlats, [(358.316743, -1.708471)])

    # double self-crossing
    ABCDFEc = ArcString([A, B, C, D, F, E], closed=True)
    assert ABCDFEc.crosses_self
    len(ABCDFEc.crossings_with_self) == 2

    # non-self-crossing
    ACBD = ArcString([A, C, B, D])
    assert not ACBD.crosses_self
    assert ACBD.crossings_with_self is None

    # closed and looped arcstrings
    ABCDc = ArcString([A, B, C, D], closed=True)
    assert ABCDc.crosses_self
    ABCDA = ArcString([A, B, C, D, A], closed=False)
    assert ABCDA.crosses_self

    # non-closed arcstrings
    ACBDc = ArcString([A, C, B, D], closed=True)
    assert not ACBDc.crosses_self
    ACBDA = ArcString([A, C, B, D, A], closed=False)
    assert not ACBDA.crosses_self
