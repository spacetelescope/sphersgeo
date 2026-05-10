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


def test_init():
    lonlats = [
        (0.0, 90.0),
        (0.0, -90.0),
        (45.0, -45.0),
        (45.0, 45.0),
    ]

    xyzs = [
        (0.2, 0.5, 0.7),
        (0.0, 0.0, 0.0),
        (1.0, 1.2, 0.3),
        (4.0, -1.0, 0.0),
    ]

    single_from_array = SphericalPolygon(np.array(lonlats))
    single_from_tuple = SphericalPolygon([tuple(vector) for vector in lonlats])
    single_from_list = SphericalPolygon(lonlats)

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array

    assert SphericalPolygon(single_from_array) == single_from_array

    assert SphericalPolygon((lonlats, (45.0, 0.0))) == single_from_list
    assert SphericalPolygon((lonlats, (0.0, 0.0))) != single_from_list

    multi_from_list_of_arrays = MultiSphericalPolygon(
        [np.array(vectors) for vectors in (lonlats, xyzs)]
    )
    multi_from_lists_of_tuples = MultiSphericalPolygon(
        [[tuple(vector) for vector in vectors] for vectors in (lonlats, xyzs)]
    )
    multi_from_nested_lists = MultiSphericalPolygon([lonlats, xyzs])

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays

    assert MultiSphericalPolygon(multi_from_list_of_arrays) == multi_from_list_of_arrays


@pytest.mark.parametrize("lon", (0, 120, 240))
@pytest.mark.parametrize("lat", (0, 30, 60, 90))
def test_from_cone(lon, lat):
    polygon = SphericalPolygon.from_cone((lon, lat), radius=10, steps=64)

    assert len(polygon.vertices.xyzs) == 63
    assert_almost_equal(polygon.area, 312, decimal=0)


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


TEST_POLYGONS = [
    (
        [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
        0.03038215667460244,
        (5.0, 5.005959),
        True,
    ),
    (
        [(90.0, 0.0), (0.0, 45.0), (0.0, -45.0)],
        1.5707963267948968,
        (35.2643897, 0.0),
        True,
    ),
    (
        [(90.0, 0.0), (0.0, 22.5), (0.0, -22.5)],
        0.7853981633974486,
        (33.155842, 0.0),
        True,
    ),
    (
        [(90.0, 0.0), (0.0, 11.25), (0.0, -11.25)],
        0.39269908169872403,
        (32.648859, 0.0),
        True,
    ),
    (
        [
            (20.0, 5.0),
            (25.0, 5.0),
            (25.0, 10.0),
            (20.0, 10.0),
        ],
        0.007552428735220221,
        (22.5, 7.502247),
        True,
    ),
    (
        [
            (5.0, 5.0),
            (25.0, 5.0),
            (25.0, 15.0),
            (5.0, 15.0),
        ],
        0.06047687635308728,
        (15.0, 10.122954),
        True,
    ),
    (
        [
            (18.0, 6.0),
            (20.0, 5.0),
            (25.0, 5.0),
            (25.0, 10.0),
            (20.0, 10.0),
            (18.0, 7.0),
        ],
        0.009368119881222133,
        (21.864038, 7.428367),
        True,
    ),
    (
        [
            # clockwise (inverse polygon comprising most of the sphere)
            (18.0, 7.0),
            (20.0, 10.0),
            (25.0, 10.0),
            (25.0, 5.0),
            (20.0, 5.0),
            (18.0, 6.0),
        ],
        12.55700249447795,
        (-158.135962, -7.428367),
        False,
    ),
    (
        [
            (18.0, 6.0),
            (20.0, 5.0),
            (25.0, 5.0),
            (25.0, 10.0),
            (20.0, 10.0),
            (19.0, 8.0),  # concave vertex
            (18.0, 7.0),
        ],
        0.009215352192799085,
        (21.911468, 7.413186),
        False,
    ),
    (
        [
            # clockwise (inverse polygon comprising most of the sphere)
            (18.0, 7.0),
            (19.0, 8.0),  # concave vertex
            (20.0, 10.0),
            (25.0, 10.0),
            (25.0, 5.0),
            (20.0, 5.0),
            (18.0, 6.0),
        ],
        12.557155262166374,
        (-158.088532, -7.413186),
        False,
    ),
    (
        # nearly degenerate, from https://github.com/spacetelescope/spherical_geometry/issues/192
        [
            [21.18490548, 19.72227505],
            [21.46931577, 7.44460937],
            [21.73600364, -4.72309959],
        ],
        8.082013188722373e-07,
        (21.468642, 7.481543),
        True,
    ),
]


@pytest.mark.parametrize("polygon", TEST_POLYGONS)
def test_area(polygon):
    assert_almost_equal(SphericalPolygon(polygon[0]).area / 3282.8065632, polygon[1])


@pytest.mark.parametrize("polygon", TEST_POLYGONS)
def test_centroid(polygon):
    assert_almost_equal(SphericalPolygon(polygon[0]).centroid.lonlat, polygon[2])


@pytest.mark.parametrize("polygon", TEST_POLYGONS)
def test_is_convex(polygon):
    assert SphericalPolygon(polygon[0]).is_convex == polygon[3]


@pytest.mark.parametrize("polygon", TEST_POLYGONS)
def test_contains_point(polygon):
    assert SphericalPolygon(polygon[0]).contains(SphericalPoint(polygon[2]))


def test_symmetric_difference():
    a = SphericalPolygon([(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)])
    b = SphericalPolygon([(5.0, 5.0), (25.0, 5.0), (25.0, 15.0), (5.0, 15.0)])

    symmetric_difference = a.symmetric_difference(b)

    # TODO: add more validation
    assert symmetric_difference is not None


@pytest.mark.parametrize("polygon", TEST_POLYGONS)
@pytest.mark.parametrize("x_offset", list(range(11)))
@pytest.mark.parametrize("y_offset", [1e-8])
def test_overlap(polygon, x_offset, y_offset):
    lonlats = np.asanyarray(polygon[0])

    original_polygon = SphericalPolygon(lonlats)
    offsetted_polygon = SphericalPolygon(lonlats + [x_offset, y_offset])

    assert np.allclose(
        original_polygon.intersection(offsetted_polygon).area / original_polygon.area,
        (10.0 - x_offset) / 10.0,
    )


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
