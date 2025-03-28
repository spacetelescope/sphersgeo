import numpy as np
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
    tolerance = 1e-10

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
            C = ArcString((A + B).xyz).midpoints.parts[0]
            assert B.angle_between(A, C) == np.pi
            assert B.collinear(A, C)
            assert_allclose(A.distance(B), B.distance(C), atol=tolerance)


def test_contains():
    arc = ArcString(
        MultiSphericalPoint.from_lonlats(
            [(-30.0, -30.0), (30.0, 30.0)], degrees=True
        ).xyz
    )
    assert arc.contains(
        SphericalPoint.from_lonlat((349.10660535, -12.30998866), degrees=True)
    )

    vertical_arc = ArcString(
        MultiSphericalPoint.from_lonlats([(60.0, 0.0), (60.0, 30.0)], degrees=True).xyz,
    )
    for i in range(1, 29):
        assert vertical_arc.contains(
            SphericalPoint.from_lonlat((60.0, float(i)), degrees=True)
        )

    horizontal_arc = ArcString(
        MultiSphericalPoint.from_lonlats([(0.0, 60.0), (30.0, 60.0)], degrees=True).xyz,
    )
    for i in range(1, 29):
        assert not horizontal_arc.contains(
            SphericalPoint.from_lonlat((float(i), 60.0), degrees=True)
        )


def test_interpolate():
    tolerance = 1e-10

    a_lonlat = (60.0, 0.0)
    b_lonlat = (60.0, 30.0)

    a = SphericalPoint.from_lonlat(a_lonlat, degrees=True)
    b = SphericalPoint.from_lonlat(b_lonlat, degrees=True)

    lonlats = sphersgeo.array.interpolate_points_along_vector_arc(a.xyz, b.xyz, n=10)

    assert_allclose(lonlats[0], a_lonlat)
    assert_allclose(lonlats[-1], b_lonlat)

    xyzs = sphersgeo.array.interpolate(a.xyz, b.xyz, n=10)

    assert_allclose(xyzs[0], a.xyz)
    assert_allclose(xyzs[-1], b.xyz)

    arc_from_lonlats = ArcString(
        MultiSphericalPoint.from_lonlats(lonlats, degrees=True)
    )
    arc_from_xyzs = ArcString(MultiSphericalPoint(xyzs))

    for xyz in xyzs:
        point = SphericalPoint(xyz)
        assert arc_from_lonlats.contains(point)
        assert arc_from_xyzs.contains(point)

    distances_from_lonlats = arc_from_lonlats.lengths
    distances_from_xyz = arc_from_xyzs.lengths

    assert np.allclose(distances_from_lonlats, distances_from_xyz, atol=tolerance)


def test_intersection():
    A = SphericalPoint.from_lonlat((-10.0, -10.0), degrees=True)
    B = SphericalPoint.from_lonlat((10.0, 10.0), degrees=True)

    C = SphericalPoint.from_lonlat((-25.0, 10.0), degrees=True)
    D = SphericalPoint.from_lonlat((15.0, -10.0), degrees=True)

    # E = SphericalPoint.from_lonlat((-20.0, 40.0), degrees=True)
    # F = SphericalPoint.from_lonlat((20.0, 40.0), degrees=True)

    reference_intersection = (0.99912414, -0.02936109, -0.02981403)

    AB = ArcString([A, B])
    CD = ArcString([C, D])
    assert AB.intersects(CD)
    r = AB.intersection(CD)
    assert r.xyz.shape == (3,)
    assert_allclose(r.xyz, reference_intersection)

    # assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
    # r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
    # assert r.shape == (2, 3)
    # assert_allclose(r[0], reference_intersection)
    # assert np.all(np.isnan(r[1]))

    # Test parallel arc.xyzs
    r = AB.intersection(AB)
    assert np.all(np.isnan(r.xyz))
