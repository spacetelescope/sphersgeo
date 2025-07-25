import numpy as np
import pytest
from numpy.testing import assert_allclose
from sphersgeo import ArcString, MultiArcString, MultiSphericalPoint, SphericalPoint
import sphersgeo


def test_init():
    vectors_a = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]

    vectors_b = [
        [0.2, 0.5, 0.7],
        [0.0, 0.0, 0.0],
        [1.0, 1.2, 0.3],
        [4.0, -1.0, 0.0],
    ]

    single_from_array = ArcString(np.array(vectors_a))
    single_from_tuple = ArcString([tuple(vector) for vector in vectors_a])
    single_from_list = ArcString(vectors_a)
    single_from_flat_list = ArcString(np.array(vectors_a).flatten().tolist())

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array
    assert single_from_flat_list == single_from_array

    assert ArcString(single_from_array) == single_from_array

    multi_from_list_of_arrays = MultiArcString(
        [np.array(vectors) for vectors in (vectors_a, vectors_b)]
    )
    multi_from_lists_of_tuples = MultiArcString(
        [[tuple(vector) for vector in vectors] for vectors in (vectors_a, vectors_b)]
    )
    multi_from_nested_lists = MultiArcString([vectors_a, vectors_b])
    multi_from_flat_lists = MultiArcString(
        [np.array(vectors).flatten().tolist() for vectors in (vectors_a, vectors_b)]
    )

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_flat_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays
    assert multi_from_flat_lists == multi_from_list_of_arrays

    assert MultiArcString(multi_from_list_of_arrays) == multi_from_list_of_arrays


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
        A = SphericalPoint.from_lonlat(a, degrees=True)
        for b in bvec:
            B = SphericalPoint.from_lonlat(b, degrees=True)
            mid = ArcString([A, B]).midpoints.parts[0]
            assert_allclose(A.distance(mid), mid.distance(B), atol=tolerance)
            assert_allclose(mid.angle_between(A, B), 180, rtol=tolerance)


def test_contains():
    diagonal_arc = ArcString(
        MultiSphericalPoint.from_lonlat(
            [(-30.0, -30.0), (30.0, 30.0)], degrees=True
        ).xyz
    )
    assert diagonal_arc.contains(SphericalPoint.from_lonlat((0, 0), degrees=True))

    vertical_arc = ArcString(
        MultiSphericalPoint.from_lonlat([(60.0, 0.0), (60.0, 30.0)], degrees=True).xyz,
    )
    for latitude in np.arange(1.0, 29.0, 1.0):
        assert vertical_arc.contains(
            SphericalPoint.from_lonlat((60.0, latitude), degrees=True)
        )

    horizontal_arc = ArcString(
        MultiSphericalPoint.from_lonlat([(0.0, 60.0), (30.0, 60.0)], degrees=True).xyz,
    )
    for longitude in np.arange(1.0, 29.0, 1.0):
        assert not horizontal_arc.contains(
            SphericalPoint.from_lonlat((longitude, 60.0), degrees=True)
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
def test_interpolate(a, b):
    tolerance = 1e-10

    a = SphericalPoint.from_lonlat(a, degrees=True)
    b = SphericalPoint.from_lonlat(b, degrees=True)
    ab = ArcString([a, b])

    interpolated_points = MultiSphericalPoint(
        sphersgeo.array.interpolate_points_along_vector_arc(a.xyz, b.xyz, n=10)
    ).parts

    assert interpolated_points[0] == a
    assert interpolated_points[-1] == b

    interpolated_arc = ArcString(interpolated_points)

    for point in interpolated_points[1:-1]:
        assert ab.contains(point)

    distances = interpolated_arc.lengths

    assert np.allclose(distances, ab.length / len(interpolated_arc), atol=tolerance)


def test_intersection():
    A = SphericalPoint.from_lonlat((-10.0, -10.0), degrees=True)
    B = SphericalPoint.from_lonlat((10.0, 10.0), degrees=True)
    C = SphericalPoint.from_lonlat((-25.0, 10.0), degrees=True)
    D = SphericalPoint.from_lonlat((15.0, -10.0), degrees=True)
    E = SphericalPoint.from_lonlat((-20.0, 40.0), degrees=True)
    F = SphericalPoint.from_lonlat((20.0, 40.0), degrees=True)

    # simple intersection
    AB = ArcString([A, B])
    CD = ArcString([C, D])
    EF = ArcString([E, F])
    assert AB.intersects(CD)
    assert not AB.intersects(EF)
    assert_allclose(
        AB.intersection(CD).to_lonlat(degrees=True), [(358.316743, -1.708471)]
    )

    # intersection with later part
    ABE = ArcString([A, B, E])
    CF = ArcString([C, F])
    assert ABE.intersects(CF)

    # multi-part geometry intersection
    AB_EF = MultiArcString([AB, EF])
    assert AB_EF.intersects(CD)
    assert_allclose(
        AB_EF.intersection(CD).to_lonlat(degrees=True), [(358.316743, -1.708471)]
    )

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


def test_crosses_self():
    A = SphericalPoint.from_lonlat((-10.0, -10.0), degrees=True)
    B = SphericalPoint.from_lonlat((10.0, 10.0), degrees=True)
    C = SphericalPoint.from_lonlat((-25.0, 10.0), degrees=True)
    D = SphericalPoint.from_lonlat((15.0, -10.0), degrees=True)
    E = SphericalPoint.from_lonlat((-20.0, 40.0), degrees=True)
    F = SphericalPoint.from_lonlat((20.0, 40.0), degrees=True)

    # simple self-crossing
    ABCD = ArcString([A, B, C, D])
    assert ABCD.crosses_self
    assert_allclose(
        ABCD.crossings_with_self.to_lonlat(degrees=True),
        [(358.316743, -1.708471)],
    )

    # longer self-crossing
    ABCDFE = ArcString([A, B, C, D, F, E])
    assert ABCDFE.crosses_self
    len(ABCDFE.crossings_with_self) == 1
    assert_allclose(
        ABCDFE.crossings_with_self.to_lonlat(degrees=True), [(358.316743, -1.708471)]
    )

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
