import math
import random
from pathlib import Path

import numpy as np
import pytest
from sphersgeo import SphericalPolygon
from numpy.testing import assert_almost_equal

DATA_DIRECTORY = Path(__file__).parent / "data"

def test_init():
    a = SphericalPolygon((np.array[[]]))


def test_is_clockwise():
    clockwise_poly = SphericalPolygon.from_cone(0.0, 90.0, 1.0)
    assert clockwise_poly.is_clockwise()

    points = list(clockwise_poly.points)[0]
    inside = list(clockwise_poly.inside)[0]
    outside = -1.0 * inside

    rpoints = points[::-1]
    reverse_poly = SphericalPolygon(rpoints, inside=inside)
    assert reverse_poly.is_clockwise()

    complement_poly = SphericalPolygon(points, inside=outside)
    assert not complement_poly.is_clockwise()


def test_overlap():
    y_eps = 1e-8

    def build_polygon(offset):
        points = []
        corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
        for corner in corners:
            x, y = corner
            points.append(np.asarray(vector.lonlat_to_vector(x + offset, y + y_eps)))
        poly = SphericalPolygon(points)
        return poly

    first_poly = build_polygon(0.0)
    for i in range(11):
        offset = float(i)
        second_poly = build_polygon(offset)
        overlap_area = first_poly.overlap(second_poly)
        calculated_area = (10.0 - offset) / 10.0
        assert abs(overlap_area - calculated_area) < 0.0005


def test_from_wcs():
    from astropy.io import fits

    header = fits.getheader(DATA_DIRECTORY / "j8bt06nyq_flt.fits", ext=("SCI", 1))

    poly = SphericalPolygon.from_wcs(header)
    for lonlat in poly.to_lonlat():
        lon = lonlat[0]
        lat = lonlat[1]
        assert np.all(np.absolute(lon - 6.027148333333) < 0.2)
        assert np.all(np.absolute(lat + 72.08351111111) < 0.2)


def test_intersects_poly_simple():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    ra2 = np.array([-5, 15, 15, -5, -5], dtype=float)
    dec2 = np.array([20, 20, -10, -10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(ra2, dec2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)


def test_intersects_poly_fully_contained():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = np.array([-5, 5, 5, -5, -5], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert poly1.intersects_poly(poly2)


def test_hard_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert poly1.intersects_poly(poly2)


def test_not_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 5, 5, 30], dtype=float)
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([-20, -20, -10, -10, -20], dtype=float)
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert not poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    poly1 = SphericalPolygon.from_lonlat(lon1, lat1)

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    poly2 = SphericalPolygon.from_lonlat(lon2, lat2)

    assert not poly1.intersects_poly(poly2)


def test_point_in_poly():
    point = VectorPoint(np.asarray([-0.27475449, 0.47588873, -0.83548781]))
    points = np.asarray(
        [
            [0.04821217, -0.29877206, 0.95310589],
            [0.04451801, -0.47274119, 0.88007608],
            [-0.14916503, -0.46369786, 0.87334649],
            [-0.16101648, -0.29210164, 0.94273555],
            [0.04821217, -0.29877206, 0.95310589],
        ]
    )
    inside = np.asarray([-0.03416009, -0.36858623, 0.9289657])
    poly = SphericalPolygon(points, inside)
    assert not poly.contains_point(point)

    lonlat = point.to_lonlat()
    assert not poly.contains_lonlat(lonlat)


def test_point_in_poly_lots():
    from astropy.io import fits

    header = fits.getheader(resolve_imagename(ROOT_DIR, "1904-77_TAN.fits"), ext=0)

    poly1 = polygon.SphericalPolygon.from_wcs(header, 1, crval=[0, 87])
    poly2 = polygon.SphericalPolygon.from_wcs(header, 1, crval=[20, 89])
    poly3 = polygon.SphericalPolygon.from_wcs(header, 1, crval=[180, 89])

    points = get_point_set()
    count = 0
    for point in points:
        if (
            poly1.contains_point(point)
            or poly2.contains_point(point)
            or poly3.contains_point(point)
        ):
            count += 1

    assert count == 5
    assert poly1.intersects_poly(poly2)
    assert not poly1.intersects_poly(poly3)
    assert not poly2.intersects_poly(poly3)


def test_cone():
    random.seed(0)
    for i in range(50):
        lon = random.randrange(-180, 180)
        lat = random.randrange(20, 90)
        _ = polygon.SphericalPolygon.from_cone(lon, lat, 8, steps=64)


def test_area():
    triangles = [
        ([[90, 0], [0, 45], [0, -45], [90, 0]], np.pi * 0.5),
        ([[90, 0], [0, 22.5], [0, -22.5], [90, 0]], np.pi * 0.25),
        ([[90, 0], [0, 11.25], [0, -11.25], [90, 0]], np.pi * 0.125),
    ]

    for tri, area in triangles:
        tri = np.array(tri)
        x, y, z = vector.lonlat_to_vector(tri[:, 1], tri[:, 0])
        points = np.dstack((x, y, z))[0]
        poly = polygon.SphericalPolygon(points)
        calc_area = poly.area()
        assert_almost_equal(calc_area, area)


def test_cone_area():
    saved_area = None
    for lon in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
        for lat in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
            area = polygon.SphericalPolygon.from_cone(lon, lat, 30, steps=64).area()
            if saved_area is None:
                saved_area = area
            assert_almost_equal(area, saved_area)


def test_fast_area():
    a = np.array(  # Clockwise
        [
            [0.35327617, 0.6351561, -0.6868571],
            [0.35295533, 0.63510299, -0.68707112],
            [0.35298984, 0.63505081, -0.68710162],
            [0.35331262, 0.63510039, -0.68688987],
            [0.35327617, 0.6351561, -0.6868571],
        ]
    )

    b = np.array(
        [  # Clockwise
            [0.35331737, 0.6351013, -0.68688658],
            [0.3536442, 0.63515101, -0.68667239],
            [0.35360581, 0.63521041, -0.68663722],
            [0.35328338, 0.63515742, -0.68685217],
            [0.35328614, 0.63515318, -0.68685467],
            [0.35328374, 0.63515279, -0.68685627],
            [0.35331737, 0.6351013, -0.68688658],
        ]
    )

    c = np.array(
        [  # Counterclockwise
            [0.35327617, 0.6351561, -0.6868571],
            [0.35331262, 0.63510039, -0.68688987],
            [0.35298984, 0.63505081, -0.68710162],
            [0.35295533, 0.63510299, -0.68707112],
            [0.35327617, 0.6351561, -0.6868571],
        ]
    )

    c_inside = [-0.35327617, -0.6351561, 0.6868571]

    apoly = polygon.SphericalPolygon(a)
    bpoly = polygon.SphericalPolygon(b)
    cpoly = polygon.SphericalPolygon(c, inside=c_inside)

    aarea = apoly.area()
    barea = bpoly.area()
    carea = cpoly.area()

    assert aarea > 0.0 and aarea < 2.0 * np.pi
    assert barea > 0.0 and barea < 2.0 * np.pi
    assert carea > 2.0 * np.pi and carea < 4.0 * np.pi


@pytest.mark.parametrize("repeat_pts", [False, True])
def test_convex_hull(repeat_pts):
    lon = (
        0.02,
        0.10,
        0.05,
        0.03,
        0.04,
        0.07,
        0.00,
        0.06,
        0.08,
        0.13,
        0.08,
        0.14,
        0.15,
        0.12,
        0.01,
        0.11,
    )
    lat = (
        0.06,
        0.00,
        0.05,
        0.01,
        0.12,
        0.08,
        0.03,
        0.02,
        0.04,
        0.03,
        0.10,
        0.11,
        0.01,
        0.13,
        0.09,
        0.07,
    )

    if repeat_pts:
        lon = lon + lon[::-1]
        lat = lat + lat[::-1]

    lon_lat = list(zip(lon, lat))

    points = []
    for l, b in lon_lat:
        points.append(list(vector.lonlat_to_vector(l, b)))
    points = np.asarray(points)

    # The method call
    poly = polygon.SphericalPolygon.convex_hull(points)

    boundary = list(poly.points)[0]
    boundary_lon, boundary_lat = vector.vector_to_lonlat(
        boundary[:, 0], boundary[:, 1], boundary[:, 2]
    )
    boundary = list(zip(boundary_lon, boundary_lat))
    on_boundary = []
    for p in lon_lat:
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
