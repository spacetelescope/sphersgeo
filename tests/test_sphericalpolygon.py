import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sphersgeo import (
    MultiSphericalPoint,
    MultiSphericalPolygon,
    SphericalPoint,
    SphericalPolygon,
)

DATA_DIRECTORY = Path(__file__).parent / "data"

TEST_POINTS = [
    (0.88955854, 87.53857137),
    (20.6543883, 87.60498618),
    (343.19474696, 85.05565535),
    (8.94286202, 85.50465173),
    (27.38417684, 85.03404907),
    (310.53503934, 88.56749324),
    (0, 60),
    (0, 90),
    (12, 66),
]


def test_init():
    lonlats_a = [
        (0.0, 90.0),
        (0.0, -90.0),
        (45.0, -45.0),
        (45.0, 45.0),
    ]

    lonlats_b = [
        (0.2, 0.5, 0.7),
        (0.0, 0.0, 0.0),
        (1.0, 1.2, 0.3),
        (4.0, -1.0, 0.0),
    ]

    single_from_array = SphericalPolygon(np.array(lonlats_a))
    single_from_tuple = SphericalPolygon([tuple(vector) for vector in lonlats_a])
    single_from_list = SphericalPolygon(lonlats_a)

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array

    assert SphericalPolygon(single_from_array) == single_from_array

    assert SphericalPolygon((lonlats_a, (45.0, 0.0))) == single_from_list
    assert SphericalPolygon((lonlats_a, (0.0, 0.0))) != single_from_list

    multi_from_list_of_arrays = MultiSphericalPolygon(
        [np.array(vectors) for vectors in (lonlats_a, lonlats_b)]
    )
    multi_from_lists_of_tuples = MultiSphericalPolygon(
        [[tuple(vector) for vector in vectors] for vectors in (lonlats_a, lonlats_b)]
    )
    multi_from_nested_lists = MultiSphericalPolygon([lonlats_a, lonlats_b])

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays

    assert MultiSphericalPolygon(multi_from_list_of_arrays) == multi_from_list_of_arrays


@pytest.mark.parametrize("lon", (0, 120, 240))
@pytest.mark.parametrize("lat", (0, 30, 60, 90))
def test_from_cone(lon, lat):
    polygon = SphericalPolygon.from_cone((lon, lat), radius=10, steps=64)

    assert len(polygon.vertices.xyzs) == 63
    assert_almost_equal(polygon.area, 312, decimal=0)


@pytest.mark.parametrize(
    "lonlats,is_clockwise",
    [
        (
            [
                (18.0, 6.0),
                (20.0, 5.0),
                (25.0, 5.0),
                (25.0, 10.0),
                (20.0, 10.0),
                (18.0, 7.0),
            ],
            False,
        ),
        (
            [
                (18.0, 7.0),
                (20.0, 10.0),
                (25.0, 10.0),
                (25.0, 5.0),
                (20.0, 5.0),
                (18.0, 6.0),
            ],
            True,
        ),
    ],
)
def test_is_clockwise(lonlats, is_clockwise):
    poly = SphericalPolygon(lonlats)
    assert poly.is_clockwise == is_clockwise


def test_symmetric_difference():
    a = SphericalPolygon([(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)])
    b = SphericalPolygon([(5.0, 5.0), (25.0, 5.0), (25.0, 15.0), (5.0, 15.0)])

    symmetric_difference = a.symmetric_difference(b)

    # TODO: add more validation
    assert symmetric_difference is not None


def test_overlap():
    y_eps = 1e-8

    def build_polygon(offset: float):
        lonlats = np.array([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)])
        lonlats[:, 0] += offset
        lonlats[:, 1] += y_eps
        poly = SphericalPolygon(lonlats)
        return poly

    first_poly = build_polygon(0.0)
    for offset in range(11):
        second_poly = build_polygon(offset)
        overlap_area = first_poly.intersection(second_poly).area / first_poly.area
        calculated_area = (10.0 - offset) / 10.0
        assert abs(overlap_area - calculated_area) < 0.0005


@pytest.mark.parametrize("test_point", TEST_POINTS)
@pytest.mark.parametrize("rotation", [0, 32])
@pytest.mark.parametrize(
    "bounding_box,pixel_shape",
    [(((-0.5, 4096 - 0.5), (-0.5, 4096 - 0.5)), None)],
)
def test_from_wcs(test_point, rotation, bounding_box, pixel_shape):
    import astropy.coordinates as coord
    import astropy.modeling.models as amm
    import astropy.units as u
    from gwcs import WCS, coordinate_frames
    from sphersgeo.from_wcs import polygon_from_wcs

    transform = (amm.Shift(-2048) & amm.Shift(-2048)) | (
        amm.Scale(0.11 / 3600.0) & amm.Scale(0.11 / 3600.0)
        | amm.Rotation2D(rotation)
        | amm.Pix2Sky_TAN()
        | amm.RotateNative2Celestial(*test_point, 180.0)
    )
    detector_frame = coordinate_frames.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky_frame = coordinate_frames.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    wcsobj = WCS([(detector_frame, transform), (sky_frame, None)])
    if pixel_shape is not None:
        wcsobj.pixel_shape = pixel_shape
    if bounding_box is not None:
        wcsobj.bounding_box = bounding_box

    polygon = polygon_from_wcs(wcsobj)

    assert polygon.area > 0
    assert polygon.contains(SphericalPoint(test_point))


def test_point_in_poly():
    point = SphericalPoint((-0.27475449, 0.47588873, -0.83548781))
    poly = SphericalPolygon(
        (
            [
                (0.04821217, -0.29877206, 0.95310589),
                (0.04451801, -0.47274119, 0.88007608),
                (-0.14916503, -0.46369786, 0.87334649),
                (-0.16101648, -0.29210164, 0.94273555),
                (0.04821217, -0.29877206, 0.95310589),
            ],
            (-0.03416009, -0.36858623, 0.9289657),
        )
    )
    assert not poly.contains(point)


@pytest.mark.parametrize(
    "lonlats,expected_area",
    [
        ([(90, 0), (0, 45), (0, -45)], 450.0),
        ([(90, 0), (0, 22.5), (0, -22.5)], 675.0),
        ([(90, 0), (0, 11.25), (0, -11.25)], 22.5),
        (
            np.array(
                [
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                ]
            ),
            0.43272229160307324,
        ),
        (
            np.array(
                [
                    (5.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 15.0),
                    (5.0, 15.0),
                ]
            ),
            3.4650697731667437,
        ),
        (
            np.array(
                [
                    (18.0, 6.0),
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                    (18.0, 7.0),
                ]
            ),
            54.50117948865602,
        ),
        (
            np.array(
                [
                    (18.0, 7.0),
                    (20.0, 10.0),
                    (25.0, 10.0),
                    (25.0, 5.0),
                    (20.0, 5.0),
                    (18.0, 6.0),
                ]
            ),
            112.56362825102903,
        ),
        (
            np.array(
                [
                    (18.0, 6.0),
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                    (19.0, 8.0),  # concave point
                    (18.0, 7.0),
                ]
            ),
            54.49242654476835,
        ),
    ],
)
def test_area(lonlats, expected_area):
    assert_almost_equal(SphericalPolygon(lonlats).area, expected_area)


@pytest.mark.parametrize(
    "lonlats,expected_centroid_lonlat",
    [
        ([(90, 0), (0, 45), (0, -45)], (35.2643897, 0.0)),
        ([(90, 0), (0, 22.5), (0, -22.5)], (28.4220791, 0.0)),
        ([(90, 0), (0, 11.25), (0, -11.25)], (27.012286, 0.0)),
        (
            np.array(
                [
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                ]
            ),
            (22.5, 7.5070637),
        ),
        (
            np.array(
                [
                    (5.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 15.0),
                    (5.0, 15.0),
                ]
            ),
            (15.0, 10.1510817),
        ),
        (
            np.array(
                [
                    (18.0, 6.0),
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                    (18.0, 7.0),
                ]
            ),
            (21.4828013, 8.0147551),
        ),
        (
            np.array(
                [
                    (18.0, 6.0),
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                    (19.0, 8.0),
                    (18.0, 7.0),
                ]
            ),
            (21.4828013, 8.0147551),
        ),
    ],
)
def test_centroid(lonlats, expected_centroid_lonlat):
    poly = SphericalPolygon(lonlats)

    assert_almost_equal(poly.centroid.lonlat, expected_centroid_lonlat)


@pytest.mark.parametrize(
    "lonlats,expected_area,expected_on_boundary",
    [
        (
            np.array(
                [
                    (20.0, 5.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                ]
            ),
            25,
            [True, True, True, True],
        ),
        (
            np.array(
                [
                    (18.0, 6.0),
                    (21.0, 6.0),
                    (20.0, 5.0),
                    (21.0, 7.0),
                    (19.0, 8.0),
                    (25.0, 5.0),
                    (25.0, 10.0),
                    (20.0, 10.0),
                    (18.0, 7.0),
                ]
            ),
            25,
            [True, False, True, False, True, True, True, True, True],
        ),
        (
            np.array(
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
            ),
            np.pi,
            [
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
            ],
        ),
        (
            np.array(
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
            ),
            np.pi,
            [
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
            ],
        ),
    ],
)
def test_convex_hull(lonlats, expected_area, expected_on_boundary):
    points = MultiSphericalPoint(lonlats)

    convex_hull = points.convex_hull

    assert convex_hull.area == expected_area

    boundary_lonlats = convex_hull.boundary.vertices.lonlats

    def lonlat_in_lonlats(
        lonlat: tuple[float, float], lonlats: list[tuple[float, float]]
    ):
        for boundary_lonlat in boundary_lonlats:
            if (
                math.sqrt(
                    (lonlat[0] - boundary_lonlat[0]) ** 2
                    + (lonlat[1] - boundary_lonlat[1]) ** 2
                )
                < 0.005
            ):
                return True
                break
        else:
            return False

    on_boundary = [lonlat_in_lonlats(lonlat, boundary_lonlats) for lonlat in lonlats]

    for b, r in zip(on_boundary, expected_on_boundary):
        assert b == r, "convex hull incorrect"
