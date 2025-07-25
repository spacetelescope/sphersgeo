import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from sphersgeo import ArcString, MultiVectorPoint, VectorPoint, arc_length, interpolate


def test_midpoint():
    avec = [
        np.array([[i, j]], dtype=float) + 7.0
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    bvec = [
        np.array([[i, j]], dtype=float) + 10.0
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    for a in avec:
        A = np.asarray(MultiVectorPoint.from_lonlats(a))
        for b in bvec:
            B = np.asarray(MultiVectorPoint.from_lonlats(b))
            C = ArcString(A + B).midpoints
            aclen = ArcString(A + C).length
            bclen = ArcString(B + C).length
            assert abs(aclen - bclen) < 1.0e-10


def test_contains():
    arc = ArcString(
        MultiVectorPoint.from_lonlats(np.array([[-30.0, -30.0], [30.0, 30.0]]))
    )
    assert arc.contains(VectorPoint.from_lonlat(np.array([349.10660535, -12.30998866])))

    vertical_arc = ArcString(
        MultiVectorPoint.from_lonlats(np.array([[60.0, 0.0], [60.0, 30.0]])),
    )
    for i in range(1, 29):
        assert vertical_arc.contains(
            VectorPoint.from_lonlat(np.array([60.0, i], dtype=float))
        )

    horizontal_arc = ArcString(
        MultiVectorPoint.from_lonlats(np.array([[0.0, 60.0], [30.0, 60.0]])),
    )
    for i in range(1, 29):
        assert not horizontal_arc.contains(
            VectorPoint.from_lonlat(np.array([i, 60.0], dtype=float))
        )


def test_interpolate():
    a_lonlat = np.array([60.0, 0.0])
    b_lonlat = np.array([60.0, 30.0])
    lonlats = interpolate(a_lonlat, b_lonlat, n=10)

    a = VectorPoint.from_lonlat(a_lonlat)
    b = VectorPoint.from_lonlat(b_lonlat)

    assert assert_allclose(lonlats[0], a_lonlat)
    assert assert_allclose(lonlats[-1], b_lonlat)

    xyzs = interpolate(a.xyz, b.xyz, n=10)

    assert assert_allclose(xyzs[0], a.xyz)
    assert assert_allclose(xyzs[-1], b.xyz)

    arc_from_lonlats = ArcString(MultiVectorPoint.from_lonlats(lonlats))
    arc_from_xyzs = ArcString(MultiVectorPoint(xyzs))

    for xyz in xyzs:
        point = VectorPoint(xyz)
        assert arc_from_lonlats.contains(point)
        assert arc_from_xyzs.contains(point)
        
    distances_from_lonlats = arc_from_lonlats.lengths
    distances_from_xyz = arc_from_xyzs.lengths

    assert (distances_from_lonlats == distances_from_xyz).all()


def test_intersection():
    A = VectorPoint.from_lonlat(np.array([-10.0, -10.0]))
    B = VectorPoint.from_lonlat(np.array([10.0, 10.0]))

    C = VectorPoint.from_lonlat(np.array([-25.0, 10.0]))
    D = VectorPoint.from_lonlat(np.array([15.0, -10.0]))

    E = VectorPoint.from_lonlat(np.array([-20.0, 40.0]))
    F = VectorPoint.from_lonlat(np.array([20.0, 40.0]))

    reference_intersection = [0.99912414, -0.02936109, -0.02981403]

    AB = ArcString(A + B)
    CD = ArcString(C + D)
    assert AB.intersects(CD)
    r = AB.intersection(CD)
    assert r.shape == (3,)
    assert_almost_equal(r, reference_intersection)

    # assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
    # r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
    # assert r.shape == (2, 3)
    # assert_almost_equal(r[0], reference_intersection)
    # assert np.all(np.isnan(r[1]))

    # Test parallel arcs
    r = AB.intersection(AB)
    assert np.all(np.isnan(r))


def test_distance():
    A = VectorPoint.from_lonlat(np.array([90.0, 0.0]))
    B = VectorPoint.from_lonlat(np.array([-90.0, 0.0]))
    assert_almost_equal(A.distance(B), np.pi)

    A = VectorPoint.from_lonlat(np.array([135.0, 0.0]))
    B = VectorPoint.from_lonlat(np.array([-90.0, 0.0]))
    assert_almost_equal(A.distance(B), (3.0 / 4.0) * np.pi)

    A = VectorPoint.from_lonlat(np.array([0.0, 0.0]))
    B = VectorPoint.from_lonlat(np.array([0.0, 90.0]))
    assert_almost_equal(A.distance(B), np.pi / 2.0)


def test_angle():
    A = VectorPoint(np.array([1.0, 0.0, 0.0]))
    B = VectorPoint(np.array([0.0, 1.0, 0.0]))
    C = VectorPoint(np.array([0.0, 0.0, 1.0]))
    assert A.angle(B, C) == (3.0 / 2.0) * np.pi

    # TODO: More angle tests


def test_angle_domain():
    A = VectorPoint(np.array([0.0, 0.0, 0.0]))
    B = VectorPoint(np.array([0.0, 0.0, 0.0]))
    C = VectorPoint(np.array([0.0, 0.0, 0.0]))
    assert A.angle(B, C) == (3.0 / 2.0) * np.pi
    assert not np.isfinite(A.angle(B, C))


def test_length_domain():
    A = VectorPoint(np.array([np.nan, 0.0, 0.0]))
    B = VectorPoint(np.array([0.0, 0.0, np.inf]))
    assert np.isnan(A.distance(B))


def test_angle_nearly_coplanar_vec():
    # test from issue #222 + extra values
    A = MultiVectorPoint(np.repeat([[1.0, 1.0, 1.0]], 5, axis=0))
    B = MultiVectorPoint(np.repeat([[1.0, 0.9999999, 1.0]], 5, axis=0))
    C = MultiVectorPoint(
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
    angles = B.angles(A, C)

    assert_allclose(angles[:-1], np.pi, rtol=0, atol=1e-16)
    assert_allclose(angles[-1], 0, rtol=0, atol=1e-32)
