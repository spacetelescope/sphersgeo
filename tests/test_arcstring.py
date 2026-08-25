from pathlib import Path

import numpy as np
import pytest
import sphersgeo
from numpy.testing import assert_allclose


def read_geometry_wkt_txt(
    *filenames: Path,
) -> dict[
    str,
    sphersgeo.AnyGeometry,
]:
    lines = []
    for filename in filenames:
        with open(filename) as geometries_file:
            lines.extend(geometries_file.readlines())

    geometries = {}
    for line in lines:
        name, wkt = line.split(",", 1)
        geometries[name] = sphersgeo.from_wkt(wkt)
    return geometries


TEST_GEOMETRIES = read_geometry_wkt_txt(Path(__file__).parent / "data" / "strings.csv")


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
        A = sphersgeo.SphericalPoint(a)
        for b in bvec:
            B = sphersgeo.SphericalPoint(b)
            mid = sphersgeo.ArcString([A, B]).midpoints[0]
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

    a = sphersgeo.SphericalPoint(a)
    b = sphersgeo.SphericalPoint(b)
    ab = sphersgeo.ArcString([a, b])

    interpolated_points = a.interpolate_points(b, n=10)

    assert interpolated_points[0] == a
    assert interpolated_points[-1] == b

    assert ab.covers(interpolated_points)
    assert ab.contains(interpolated_points[1:-1])

    interpolated_arcstring = sphersgeo.ArcString(interpolated_points)

    assert_allclose(ab.length, interpolated_arcstring.length, atol=tolerance)


TEST_ARCSTRING_INPUTS = [
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ],
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    [
        (0.2, 0.5, 0.7),
        (0.0, 0.5, 0.0),
        (1.0, 1.2, 0.3),
        (4.0, -1.0, 0.0),
    ],
    [
        [81.77235508, 26.60503776],
        [70.28505151, 5.77878756],
        [17.75344926, 6.41019422],
    ],
    [(-30.0, -30.0), (30.0, 30.0)],
    [(60.0, 0.0), (60.0, 30.0)],
    [(0.0, 0.0), (30.0, 0.0)],
]


@pytest.mark.parametrize("arcstring_input", TEST_ARCSTRING_INPUTS)
def test_init(arcstring_input):
    from_list_of_tuples = sphersgeo.ArcString(arcstring_input)
    from_nested_list = sphersgeo.ArcString([list(xyz) for xyz in arcstring_input])
    from_array = sphersgeo.ArcString(np.array(arcstring_input))

    assert from_list_of_tuples == from_nested_list
    assert from_list_of_tuples == from_array
    assert from_nested_list == from_array

    assert sphersgeo.ArcString(from_array) == from_array


def test_init_multi():
    arcstring_inputs = [xyzs for xyzs in TEST_ARCSTRING_INPUTS]

    from_lists_of_tuples = sphersgeo.MultiArcString(arcstring_inputs)
    from_nested_lists = sphersgeo.MultiArcString(
        [[list(xyz) for xyz in xyzs] for xyzs in arcstring_inputs]
    )
    from_list_of_arrays = sphersgeo.MultiArcString(
        [np.array(xyzs) for xyzs in arcstring_inputs]
    )

    assert from_lists_of_tuples == from_nested_lists
    assert from_lists_of_tuples == from_list_of_arrays

    assert sphersgeo.MultiArcString(from_list_of_arrays) == from_list_of_arrays


@pytest.mark.parametrize(
    "arcstring", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_midpoints(arcstring):
    assert arcstring.contains(arcstring.midpoints)


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
    segments = sphersgeo.MultiArcString(segments)
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
            sphersgeo.ArcString(TEST_POINTS[:2]),
            sphersgeo.ArcString(TEST_POINTS[2:4]),
            (358.316743, -1.708471),
        ),
        (
            sphersgeo.ArcString(TEST_POINTS[:2]),
            sphersgeo.ArcString(TEST_POINTS[4:]),
            None,
        ),
        # intersection with later part
        (
            sphersgeo.ArcString((TEST_POINTS[0], TEST_POINTS[1], TEST_POINTS[4])),
            sphersgeo.ArcString((TEST_POINTS[2], TEST_POINTS[5])),
            (0, 0),
        ),
        # multi-part geometry intersection
        (
            sphersgeo.MultiArcString((TEST_POINTS[:2], TEST_POINTS[4:6])),
            sphersgeo.ArcString(TEST_POINTS[2:4]),
            [(358.316743, -1.708471)],
        ),
        # ensure non-intersection of non-parallel pre-terminated arcs
        (
            sphersgeo.ArcString((TEST_POINTS[2], TEST_POINTS[4])),
            sphersgeo.ArcString(TEST_POINTS[:2]),
            None,
        ),
        # intersection with non-closed and closed arcstring
        (
            sphersgeo.ArcString(
                (TEST_POINTS[3], TEST_POINTS[5], TEST_POINTS[4]), closed=True
            ),
            sphersgeo.ArcString(TEST_POINTS[:2]),
            (0, 0),
        ),
        # intersection with self
        (
            sphersgeo.ArcString(TEST_POINTS[:2]),
            sphersgeo.ArcString(TEST_POINTS[:2]),
            (0, 0),
        ),
    ],
)
def test_intersection(a, b, intersection):
    a = sphersgeo.ArcString(a)
    b = sphersgeo.ArcString(b)

    assert a.intersects(b) == (intersection is not None)
    assert a.intersection(b) == intersection


def test_closed_not_crosses_self():
    a = sphersgeo.ArcString(
        [(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)],
        closed=True,
    )
    b = sphersgeo.ArcString(
        [(18.0, 6.0), (21.0, 6.0), (21.0, 7.0), (18.0, 7.0)],
        closed=False,
    )
    c = sphersgeo.ArcString(b, closed=True)
    d = sphersgeo.ArcString(
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
    arcstring = sphersgeo.ArcString(lonlats)

    assert not arcstring.crosses_self
    assert arcstring.crossings_with_self is None


def test_crosses_self():
    A = sphersgeo.SphericalPoint((-10.0, -10.0))
    B = sphersgeo.SphericalPoint((10.0, 10.0))
    C = sphersgeo.SphericalPoint((-25.0, 10.0))
    D = sphersgeo.SphericalPoint((15.0, -10.0))
    E = sphersgeo.SphericalPoint((-20.0, 40.0))
    F = sphersgeo.SphericalPoint((20.0, 40.0))

    # simple self-crossing
    ABCD = sphersgeo.ArcString([A, B, C, D])
    assert ABCD.crosses_self
    assert_allclose(ABCD.crossings_with_self.lonlats, [(358.316743, -1.708471)])

    # longer self-crossing
    ABCDFE = sphersgeo.ArcString([A, B, C, D, F, E])
    assert ABCDFE.crosses_self
    assert len(ABCDFE.crossings_with_self) == 1
    assert_allclose(ABCDFE.crossings_with_self.lonlats, [(358.316743, -1.708471)])

    # double self-crossing
    ABCDFEc = sphersgeo.ArcString([A, B, C, D, F, E], closed=True)
    assert ABCDFEc.crosses_self
    assert len(ABCDFEc.crossings_with_self) == 2

    # non-self-crossing
    ACBD = sphersgeo.ArcString([A, C, B, D])
    assert not ACBD.crosses_self
    assert ACBD.crossings_with_self is None

    # closed and looped arcstrings
    ABCDc = sphersgeo.ArcString([A, B, C, D], closed=True)
    assert ABCDc.crosses_self
    ABCDA = sphersgeo.ArcString([A, B, C, D, A], closed=False)
    assert ABCDA.crosses_self

    # non-closed arcstrings
    ACBDc = sphersgeo.ArcString([A, C, B, D], closed=True)
    assert not ACBDc.crosses_self
    ACBDA = sphersgeo.ArcString([A, C, B, D, A], closed=False)
    assert not ACBDA.crosses_self
