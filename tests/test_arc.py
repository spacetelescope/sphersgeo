import numpy as np
import sphersgeo
from numpy.testing import assert_almost_equal


def test_midpoint():
    avec = [
        np.array(i + 7, j + 7, dtype=float)
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    bvec = [
        np.array(i + 10, j + 10, dtype=float)
        for i in range(0, 11, 5)
        for j in range(0, 11, 5)
    ]

    for a in avec:
        A = np.asarray(sphersgeo.SphericalPoints.from_lonlats(a[0], a[1]))
        for b in bvec:
            B = np.asarray(sphersgeo.SphericalPoints.from_lonlats(b[0], b[1]))
            C = sphersgeo.GreatCircleArc(A, B).midpoint
            aclen = sphersgeo.GreatCircleArc(A, C).subtends
            bclen = sphersgeo.GreatCircleArc(B, C).subtends
            assert abs(aclen - bclen) < 1.0e-10


def test_contains():
    arc = sphersgeo.GreatCircleArc(
        sphersgeo.SphericalPoint.from_lonlat(np.array([60.0, 0.0])),
        sphersgeo.SphericalPoint.from_lonlat(np.array([60.0, 30.0])),
    )
    for i in range(1, 29):
        assert arc.contains(
            sphersgeo.SphericalPoint.from_lonlat(np.array([60.0, i], dtype=float))
        )

    arc = sphersgeo.GreatCircleArc(
        sphersgeo.SphericalPoint.from_lonlat(np.array([0.0, 60.0])),
        sphersgeo.SphericalPoint.from_lonlat(np.array([30.0, 60.0])),
    )
    for i in range(1, 29):
        assert not arc.contains(
            sphersgeo.SphericalPoint.from_lonlat(np.array([float(i), 60.0]))
        )


def test_interpolate():
    arc = sphersgeo.GreatCircleArc(
        sphersgeo.SphericalPoint(np.array([60.0, 0.0])),
        sphersgeo.SphericalPoint(np.array([60.0, 30.0])),
    )
    cvec = arc.interpolate_points(n=10)

    first_length = sphersgeo.GreatCircleArc(cvec[0], cvec[1]).subtends
    for i in range(1, 9):
        length = sphersgeo.GreatCircleArc(cvec[i], cvec[i + 1]).subtends
        assert abs(length - first_length) < 1.0e-10


def test_great_circle_arc_intersection():
    A = [-10, -10]
    B = [10, 10]

    C = [-25, 10]
    D = [15, -10]

    E = [-20, 40]
    F = [20, 40]

    correct = [0.99912414, -0.02936109, -0.02981403]

    A = vector.lonlat_to_vector(*A)
    B = vector.lonlat_to_vector(*B)
    C = vector.lonlat_to_vector(*C)
    D = vector.lonlat_to_vector(*D)
    E = vector.lonlat_to_vector(*E)
    F = vector.lonlat_to_vector(*F)

    assert great_circle_arc.intersects(A, B, C, D)
    r = great_circle_arc.intersection(A, B, C, D)
    assert r.shape == (3,)
    assert_almost_equal(r, correct)

    assert np.all(great_circle_arc.intersects([A], [B], [C], [D]))
    r = great_circle_arc.intersection([A], [B], [C], [D])
    assert r.shape == (1, 3)
    assert_almost_equal(r, [correct])

    assert np.all(great_circle_arc.intersects([A], [B], C, D))
    r = great_circle_arc.intersection([A], [B], C, D)
    assert r.shape == (1, 3)
    assert_almost_equal(r, [correct])

    assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
    r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
    assert r.shape == (2, 3)
    assert_almost_equal(r[0], correct)
    assert np.all(np.isnan(r[1]))

    # Test parallel arcs
    r = great_circle_arc.intersection(A, B, A, B)
    assert np.all(np.isnan(r))


def test_great_circle_arc_length():
    A = [90, 0]
    B = [-90, 0]
    A = vector.lonlat_to_vector(*A)
    B = vector.lonlat_to_vector(*B)
    assert_almost_equal(great_circle_arc.length(A, B), np.pi)

    A = [135, 0]
    B = [-90, 0]
    A = vector.lonlat_to_vector(*A)
    B = vector.lonlat_to_vector(*B)
    assert_almost_equal(great_circle_arc.length(A, B), (3.0 / 4.0) * np.pi)

    A = [0, 0]
    B = [0, 90]
    A = vector.lonlat_to_vector(*A)
    B = vector.lonlat_to_vector(*B)
    assert_almost_equal(great_circle_arc.length(A, B), np.pi / 2.0)


def test_great_circle_arc_angle():
    A = [1, 0, 0]
    B = [0, 1, 0]
    C = [0, 0, 1]
    assert great_circle_arc.angle(A, B, C) == (3.0 / 2.0) * np.pi

    # TODO: More angle tests
