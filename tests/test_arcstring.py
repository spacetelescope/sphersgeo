import numpy as np
import pytest
from numpy.testing import assert_allclose
from sphersgeo import ArcString, MultiArcString, MultiSphericalPoint, SphericalPoint
import sphersgeo


def test_init():
    xyzs_a = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]

    xyzs_b = [
        (0.2, 0.5, 0.7),
        (0.0, 0.0, 0.0),
        (1.0, 1.2, 0.3),
        (4.0, -1.0, 0.0),
    ]

    single_from_array = ArcString(xyzs_a)
    single_from_tuple = ArcString([tuple(vector) for vector in xyzs_a])
    single_from_list = ArcString(xyzs_a)

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array

    assert ArcString(single_from_array) == single_from_array

    multi_from_list_of_arrays = MultiArcString(
        [vectors for vectors in (xyzs_a, xyzs_b)]
    )
    multi_from_lists_of_tuples = MultiArcString(
        [[tuple(vector) for vector in vectors] for vectors in (xyzs_a, xyzs_b)]
    )
    multi_from_nested_lists = MultiArcString([xyzs_a, xyzs_b])

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays

    assert MultiArcString(multi_from_list_of_arrays) == multi_from_list_of_arrays


def test_closed():
    xyzs_a = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    ]
    xyzs_b = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    assert not ArcString(xyzs_a).closed
    assert ArcString(xyzs_b).closed
    assert ArcString(xyzs_a + [xyzs_a[0]]).closed


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
        A = SphericalPoint.from_lonlat(a)
        for b in bvec:
            B = SphericalPoint.from_lonlat(b)
            mid = ArcString([A, B]).midpoints.parts[0]
            assert_allclose(A.distance(mid), mid.distance(B), atol=tolerance)
            assert_allclose(mid.two_arc_angle(A, B), 180, rtol=tolerance)


def test_contains():
    diagonal_arc = ArcString(
        MultiSphericalPoint.from_lonlats([(-30.0, -30.0), (30.0, 30.0)]).xyzs
    )
    assert diagonal_arc.contains(SphericalPoint.from_lonlat((0, 0)))

    vertical_arc = ArcString(
        MultiSphericalPoint.from_lonlats([(60.0, 0.0), (60.0, 30.0)]).xyzs,
    )
    for latitude in np.arange(1.0, 29.0, 1.0):
        assert vertical_arc.contains(SphericalPoint.from_lonlat((60.0, latitude)))

    horizontal_arc = ArcString(
        MultiSphericalPoint.from_lonlats([(0.0, 60.0), (30.0, 60.0)]).xyzs,
    )
    for longitude in np.arange(1.0, 29.0, 1.0):
        assert not horizontal_arc.contains(
            SphericalPoint.from_lonlat((longitude, 60.0))
        )


@pytest.mark.parametrize("a", [(0.0, 0.0), (60.0, 0.0), (23.44, 79.9999)])
@pytest.mark.parametrize(
    "b",
    [
        (40.0, 30.0),
        (180.0, 90.0),
        (-30.0, 110.0),
    ],
)
def test_interpolate_along_arc(a, b):
    tolerance = 1e-10

    a = SphericalPoint.from_lonlat(a)
    b = SphericalPoint.from_lonlat(b)
    ab = ArcString([a, b])

    interpolated_points = MultiSphericalPoint(
        sphersgeo.array.interpolate_points_along_arc((a.xyz, b.xyz), n=10)
    ).parts

    assert interpolated_points[0] == a
    assert interpolated_points[-1] == b

    interpolated_arc = ArcString(interpolated_points)

    for point in interpolated_points[1:-1]:
        assert ab.contains(point)

    distances = interpolated_arc.lengths

    assert_allclose(distances, ab.length / len(interpolated_arc), atol=tolerance)


def test_adjoins_join():
    segment1 = ArcString(MultiSphericalPoint.from_lonlats([(20.0, 5.0), (25.0, 5.0)]))
    segment2 = ArcString(MultiSphericalPoint.from_lonlats([(25.0, 5.0), (25.0, 6.0)]))
    segment3 = ArcString(
        MultiSphericalPoint.from_lonlats([(25.0, 5.0), (25.0, 6.0), (25.0, 7.0)])
    )
    segment4 = ArcString(MultiSphericalPoint.from_lonlats([(25.0, 6.0), (25.0, 7.0)]))

    assert segment1.adjoins(segment2)
    assert segment2.adjoins(segment3)
    assert segment3.adjoins(segment1)
    assert not segment4.adjoins(segment1)

    joined12 = segment1.join(segment2)
    print(joined12.vertices.to_lonlats())

    joined34 = segment3.join(segment4)
    print(joined34.vertices.to_lonlats())

    joined1234 = joined12.join(joined34)
    print(joined1234.vertices.to_lonlats())


def test_intersection():
    A = SphericalPoint.from_lonlat((-10.0, -10.0))
    B = SphericalPoint.from_lonlat((10.0, 10.0))
    C = SphericalPoint.from_lonlat((-25.0, 10.0))
    D = SphericalPoint.from_lonlat((15.0, -10.0))
    E = SphericalPoint.from_lonlat((-20.0, 40.0))
    F = SphericalPoint.from_lonlat((20.0, 40.0))

    # simple intersection
    AB = ArcString([A, B])
    CD = ArcString([C, D])
    EF = ArcString([E, F])
    assert AB.intersects(CD)
    assert not AB.intersects(EF)
    assert_allclose(AB.intersection(CD).to_lonlats(), [(358.316743, -1.708471)])

    # intersection with later part
    ABE = ArcString([A, B, E])
    CF = ArcString([C, F])
    assert ABE.intersects(CF)

    # multi-part geometry intersection
    AB_EF = MultiArcString([AB, EF])
    assert AB_EF.intersects(CD)
    assert_allclose(AB_EF.intersection(CD).to_lonlats(), [(358.316743, -1.708471)])

    # ensure non-intersection of non-parallel pre-terminated arcs
    CE = ArcString([C, E])
    assert not CE.intersects(AB)
    assert CE.intersection(AB) is None

    # intersection with non-closed and closed arcstring
    DFE = ArcString([D, F, E])
    assert not AB.intersects(DFE)
    DFEc = ArcString([D, F, E], closed=True)
    assert AB.intersects(DFEc)

    # intersection with self
    assert AB.intersects(AB)


def test_closed_not_crosses_self():
    a = ArcString(
        MultiSphericalPoint.from_lonlats(
            [(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)]
        ),
        closed=True,
    )
    b = ArcString(
        MultiSphericalPoint.from_lonlats(
            [(18.0, 6.0), (21.0, 6.0), (21.0, 7.0), (18.0, 7.0)],
        ),
        closed=False,
    )
    c = ArcString(b, closed=True)
    d = ArcString(
        MultiSphericalPoint.from_lonlats(
            [(18.0, 6.0), (21.0, 7.0), (21.0, 6.0), (18.0, 7.0)],
        ),
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
    arcstring = ArcString(MultiSphericalPoint.from_lonlats(lonlats))

    assert not arcstring.crosses_self
    assert arcstring.crossings_with_self is None


def test_crosses_self():
    A = SphericalPoint.from_lonlat((-10.0, -10.0))
    B = SphericalPoint.from_lonlat((10.0, 10.0))
    C = SphericalPoint.from_lonlat((-25.0, 10.0))
    D = SphericalPoint.from_lonlat((15.0, -10.0))
    E = SphericalPoint.from_lonlat((-20.0, 40.0))
    F = SphericalPoint.from_lonlat((20.0, 40.0))

    # simple self-crossing
    ABCD = ArcString([A, B, C, D])
    assert ABCD.crosses_self
    assert_allclose(ABCD.crossings_with_self.to_lonlats(), [(358.316743, -1.708471)])

    # longer self-crossing
    ABCDFE = ArcString([A, B, C, D, F, E])
    assert ABCDFE.crosses_self
    len(ABCDFE.crossings_with_self) == 1
    assert_allclose(ABCDFE.crossings_with_self.to_lonlats(), [(358.316743, -1.708471)])

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
