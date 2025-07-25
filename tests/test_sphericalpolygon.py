import math
import random
from pathlib import Path

import numpy as np
import pytest
from sphersgeo import (
    SphericalPolygon,
    SphericalPoint,
    MultiSphericalPolygon,
    MultiSphericalPoint,
)
from numpy.testing import assert_almost_equal
from helpers import resolve_imagename, get_point_set, ROOT_DIR

DATA_DIRECTORY = Path(__file__).parent / "data"


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

    single_from_array = SphericalPolygon(np.array(vectors_a))
    single_from_tuple = SphericalPolygon([tuple(vector) for vector in vectors_a])
    single_from_list = SphericalPolygon(vectors_a)
    single_from_flat_list = SphericalPolygon(np.array(vectors_a).flatten().tolist())

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array
    assert single_from_flat_list == single_from_array

    assert SphericalPolygon(single_from_array) == single_from_array

    multi_from_list_of_arrays = MultiSphericalPolygon(
        [np.array(vectors) for vectors in (vectors_a, vectors_b)]
    )
    multi_from_lists_of_tuples = MultiSphericalPolygon(
        [[tuple(vector) for vector in vectors] for vectors in (vectors_a, vectors_b)]
    )
    multi_from_nested_lists = MultiSphericalPolygon([vectors_a, vectors_b])
    multi_from_flat_lists = MultiSphericalPolygon(
        [np.array(vectors).flatten().tolist() for vectors in (vectors_a, vectors_b)]
    )

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_flat_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays
    assert multi_from_flat_lists == multi_from_list_of_arrays

    assert MultiSphericalPolygon(multi_from_list_of_arrays) == multi_from_list_of_arrays


def test_from_cone():
    random.seed(0)
    for i in range(50):
        polygon = SphericalPolygon.from_cone(
            SphericalPoint.from_lonlat(
                (random.randrange(-180, 180), random.randrange(20, 90)), degrees=True
            ),
            8,
            steps=64,
        )
        assert polygon.area > 0


def test_cone_area():
    expected_area = None
    for lon in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
        for lat in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
            area = SphericalPolygon.from_cone(
                SphericalPoint.from_lonlat((lon, lat)), radius=30, steps=64
            ).area()
            if expected_area is None:
                expected_area = area
                print(expected_area)
            assert_almost_equal(area, expected_area)


def test_is_clockwise():
    clockwise_poly = SphericalPolygon.from_cone(0.0, 90.0, 1.0)
    assert clockwise_poly.is_clockwise

    points = list(clockwise_poly.points)[0]
    inside = list(clockwise_poly)[0]
    outside = -1.0 * inside

    rpoints = points[::-1]
    reverse_poly = SphericalPolygon(rpoints, interior_point=inside)
    assert reverse_poly.is_clockwise()

    complement_poly = SphericalPolygon(points, inside=outside)
    assert not complement_poly.is_clockwise()


def test_overlap():
    y_eps = 1e-8

    def build_polygon(offset: float):
        offset = float(offset)
        points = []
        corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
        for lon, lat in corners:
            points.append(
                SphericalPoint.from_lonlat(
                    (lon + offset, lat + y_eps), degrees=True
                ).xyz
            )
        poly = SphericalPolygon(points, None, None)
        return poly

    first_poly = build_polygon(0.0)
    for offset in range(11):
        second_poly = build_polygon(offset)
        overlap_area = first_poly.intersection(second_poly).area / first_poly.area
        calculated_area = (10.0 - offset) / 10.0
        assert abs(overlap_area - calculated_area) < 0.0005


def test_from_wcs():
    import sphersgeo
    from astropy.io import fits

    header = fits.getheader(DATA_DIRECTORY / "j8bt06nyq_flt.fits", ext=("SCI", 1))

    poly = sphersgeo.from_wcs.polygon_from_wcs(header)
    for lonlat in poly.to_lonlat(degrees=True):
        lon = lonlat[0]
        lat = lonlat[1]
        assert np.all(np.absolute(lon - 6.027148333333) < 0.2)
        assert np.all(np.absolute(lat + 72.08351111111) < 0.2)


def test_intersects_poly_simple():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    ra2 = np.array([-5, 15, 15, -5, -5], dtype=float)
    dec2 = np.array([20, 20, -10, -10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(np.stack((ra2, dec2), axis=1))

    assert poly1.intersects(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = SphericalPolygon.from_lonlat(np.stack((ra2, dec2), axis=1))

    assert poly1.intersects(poly2)


def test_intersects_poly_fully_contained():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-5, 5, 5, -5, -5], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert poly1.intersects(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert poly1.intersects(poly2)


def test_hard_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert poly1.intersects(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert poly1.intersects(poly2)


def test_not_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 5, 5, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([-20, -20, -10, -10, -20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert not poly1.intersects(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(np.stack((lon1, lat1), axis=1))

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(np.stack((lon2, lat2), axis=1))

    assert not poly1.intersects(poly2)


def test_point_in_poly():
    point = SphericalPoint((-0.27475449, 0.47588873, -0.83548781))
    poly = SphericalPolygon(
        [
            (0.04821217, -0.29877206, 0.95310589),
            (0.04451801, -0.47274119, 0.88007608),
            (-0.14916503, -0.46369786, 0.87334649),
            (-0.16101648, -0.29210164, 0.94273555),
            (0.04821217, -0.29877206, 0.95310589),
        ],
        (-0.03416009, -0.36858623, 0.9289657),
    )
    assert not poly.contains(point)


def test_point_in_poly_lots():
    from astropy.io import fits
    from sphersgeo.from_wcs import polygon_from_wcs

    header = fits.getheader(resolve_imagename(ROOT_DIR, "1904-77_TAN.fits"), ext=0)

    poly1 = polygon_from_wcs(header, 1, crval=[0, 87])
    poly2 = polygon_from_wcs(header, 1, crval=[20, 89])
    poly3 = polygon_from_wcs(header, 1, crval=[180, 89])

    points = get_point_set()
    count = 0
    for point in points:
        if poly1.contains(point) or poly2.contains(point) or poly3.contains(point):
            count += 1

    assert count == 5
    assert poly1.intersects(poly2)
    assert not poly1.intersects(poly3)
    assert not poly2.intersects(poly3)


def test_area():
    triangles = [
        ([(90, 0), (0, 45), (0, -45), (90, 0)], np.pi * 0.5),
        ([(90, 0), (0, 22.5), (0, -22.5), (90, 0)], np.pi * 0.25),
        ([(90, 0), (0, 11.25), (0, -11.25), (90, 0)], np.pi * 0.125),
    ]

    for tri, area in triangles:
        poly = SphericalPolygon(tri)
        assert_almost_equal(poly.area, area)


def test_fast_area():
    # Clockwise
    a = SphericalPolygon()

    # Clockwise
    b = SphericalPolygon(
        [
            (0.35331737, 0.6351013, -0.68688658),
            (0.3536442, 0.63515101, -0.68667239),
            (0.35360581, 0.63521041, -0.68663722),
            (0.35328338, 0.63515742, -0.68685217),
            (0.35328614, 0.63515318, -0.68685467),
            (0.35328374, 0.63515279, -0.68685627),
            (0.35331737, 0.6351013, -0.68688658),
        ]
    )

    # Counterclockwise
    c = SphericalPolygon(
        [
            (0.35327617, 0.6351561, -0.6868571),
            (0.35295533, 0.63510299, -0.68707112),
            (0.35298984, 0.63505081, -0.68710162),
            (0.35331262, 0.63510039, -0.68688987),
            (0.35327617, 0.6351561, -0.6868571),
        ],
        interior_point=(-0.35327617, -0.6351561, 0.6868571),
    )

    aarea = a.area
    barea = b.area
    carea = c.area

    assert aarea > 0.0 and aarea < 2.0 * np.pi
    assert barea > 0.0 and barea < 2.0 * np.pi
    assert carea > 2.0 * np.pi and carea < 4.0 * np.pi


@pytest.mark.parametrize("repeat_pts", [False, True])
def test_convex_hull(repeat_pts):
    lonlats = np.array(
        [
            (0.02, 0.06),
            (0.10, 0.00),
            (0.05, 0.05),
            (0.03, 0.01),
            (0.04, 0.12),
            (0.07, 0.08),
            (0.00, 0.03),
            (0.06, 0.02),
            (0.08, 0.04),
            (0.13, 0.03),
            (0.08, 0.10),
            (0.14, 0.11),
            (0.15, 0.01),
            (0.12, 0.13),
            (0.01, 0.09),
            (0.11, 0.07),
        ]
    )

    if repeat_pts:
        lonlats = lonlats + lonlats[::-1]

    points = MultiSphericalPoint.from_lonlats(lonlats, degrees=True)

    # The method call
    poly = points.convex_hull

    boundary = poly.boundary
    on_boundary = []
    for p in lonlats:
        match = False
        for b in boundary:
            distance = math.sqrt((p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2)
            if distance < 0.005:
                match = True
                break
        on_boundary.append(match)

    result = [
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]

    for b, r in zip(on_boundary, result):
        assert b == r, "Polygon boundary has correct points"
