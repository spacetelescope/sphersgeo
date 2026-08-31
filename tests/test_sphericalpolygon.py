import codecs
from pathlib import Path

import numpy as np
import pytest
import sphersgeo
from numpy.testing import assert_almost_equal

TEST_POLYGONS = pytest.helpers.read_geometry_wkt_txt(
    Path(__file__).parent / "data" / "polygons.csv"
)
TEST_SINGLEPOLYGONS = {
    name: polygon
    for name, polygon in TEST_POLYGONS.items()
    if isinstance(polygon, sphersgeo.SphericalPolygon)
}
TEST_MULTIPOLYGONS = {
    name: multipolygon
    for name, multipolygon in TEST_POLYGONS.items()
    if isinstance(multipolygon, sphersgeo.MultiSphericalPolygon)
}


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

    single_from_array = sphersgeo.SphericalPolygon(np.array(lonlats))
    single_from_tuple = sphersgeo.SphericalPolygon(
        [tuple(vector) for vector in lonlats]
    )
    single_from_list = sphersgeo.SphericalPolygon(lonlats)

    assert single_from_tuple == single_from_list
    assert single_from_tuple == single_from_array
    assert single_from_list == single_from_array

    assert sphersgeo.SphericalPolygon(single_from_array) == single_from_array

    assert sphersgeo.SphericalPolygon((lonlats, (45.0, 0.0))) == single_from_list
    assert sphersgeo.SphericalPolygon((lonlats, (0.0, 0.0))) != single_from_list

    multi_from_list_of_arrays = sphersgeo.MultiSphericalPolygon(
        [np.array(vectors) for vectors in (lonlats, xyzs)]
    )
    multi_from_lists_of_tuples = sphersgeo.MultiSphericalPolygon(
        [[tuple(vector) for vector in vectors] for vectors in (lonlats, xyzs)]
    )
    multi_from_nested_lists = sphersgeo.MultiSphericalPolygon([lonlats, xyzs])

    assert multi_from_lists_of_tuples == multi_from_nested_lists
    assert multi_from_lists_of_tuples == multi_from_list_of_arrays

    assert (
        sphersgeo.MultiSphericalPolygon(multi_from_list_of_arrays)
        == multi_from_list_of_arrays
    )


@pytest.mark.parametrize(
    "center_a,radius_a",
    [
        ((-173, 78), 13),
        ((-170, 43), 11),
        ((-177, 84), 14),
    ],
)
@pytest.mark.parametrize(
    "center_b,radius_b", [((178, -31), 9), ((173, -44), 11), ((175, -85), 5)]
)
@pytest.mark.parametrize("steps", [16, 64])
def test_cone_nonintersection(center_a, radius_a, center_b, radius_b, steps):
    a = sphersgeo.SphericalPolygon.from_cone(
        center_a,
        radius_a,
        steps,
    )

    b = sphersgeo.SphericalPolygon.from_cone(
        center_b,
        radius_b,
        steps,
    )

    assert a.intersection(b) is None


def test_complement_regression():
    """https://github.com/spacetelescope/spherical_geometry/issues/278"""

    p1 = sphersgeo.SphericalPolygon.from_cone((90, 0), 100)
    p2 = sphersgeo.SphericalPolygon.from_cone((270, 0), 100)

    origin = sphersgeo.SphericalPoint((0, 0))
    assert p1.contains(origin)
    assert p2.contains(origin)

    p12 = p1.intersection(p2)
    assert p12.contains(origin)


@pytest.mark.parametrize("lon", (0, 120, 240))
@pytest.mark.parametrize("lat", (0, 30, 60, 90))
@pytest.mark.parametrize("radius", [10, 20])
@pytest.mark.parametrize("steps", [16, 64])
def test_from_cone(lon, lat, radius, steps):
    polygon = sphersgeo.SphericalPolygon.from_cone((lon, lat), radius, steps)

    assert_almost_equal(polygon.area, 2 * np.pi * (1 - np.cos(radius)))
    assert len(polygon.vertices.xyzs) == steps - 1


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


@pytest.mark.parametrize(
    "polygon,expected",
    zip(
        TEST_SINGLEPOLYGONS.values(),
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ),
    ids=TEST_SINGLEPOLYGONS.keys(),
)
def test_is_convex(polygon, expected):
    polygon: sphersgeo.SphericalPolygon = polygon[-1]
    assert polygon.is_convex == expected


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
    assert polygon.contains(sphersgeo.SphericalPoint(test_point))


@pytest.mark.parametrize(
    "polygon", TEST_SINGLEPOLYGONS.values(), ids=TEST_SINGLEPOLYGONS.keys()
)
@pytest.mark.parametrize("x_offset", list(range(11)))
@pytest.mark.parametrize("y_offset", [1e-8])
def test_polygon_offset(polygon, x_offset, y_offset):
    polygon: sphersgeo.SphericalPolygon = polygon[-1]
    offsetted_polygon = sphersgeo.SphericalPolygon(
        polygon.vertices.xyzs + [x_offset, y_offset]
    )

    assert_almost_equal(
        polygon.intersection(offsetted_polygon).area / polygon.area,
        (10.0 - x_offset) / 10.0,
    )


@pytest.mark.parametrize(
    "polygon", TEST_SINGLEPOLYGONS.values(), ids=TEST_SINGLEPOLYGONS.keys()
)
def test_polygon_vertices_on_convex_hull(polygon):
    polygon: sphersgeo.SphericalPolygon = polygon[-1]
    convex_hull = polygon.boundary.vertices.convex_hull

    polygon_vertices_shared_with_convex_hull = [
        convex_hull.boundary.vertices.contains(vertex) for vertex in polygon.vertices
    ]

    if polygon.is_convex:
        assert all(polygon_vertices_shared_with_convex_hull)
        assert convex_hull.area == polygon.area
    else:
        assert any(polygon_vertices_shared_with_convex_hull)
        assert convex_hull.area > polygon.area


DEGENERATE_POLYGONS = {
    "samepoints": np.array(4 * [[1, 0, 0]]),
    "quartersphere": np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]]),
    "hasnullvector": [
        (0, 0),
        (15, 0),
        (75, 0),
        (75, 15),
        (0, 0, 0),  # null vector
        (15, 25),
        (0, 25),
        (0, 15),
        (0, 0),
    ],
    "ongreatcircle": np.stack([90 * np.arange(5), 5 * [0]], axis=1),
}


@pytest.mark.parametrize(
    "xyzs",
    list(DEGENERATE_POLYGONS.values()),
    ids=DEGENERATE_POLYGONS.keys(),
)
def test_degenerate_polygon(xyzs):
    with pytest.raises(ValueError):
        sphersgeo.SphericalPolygon(xyzs)


def test_difficult_intersections():
    # Tests a number of intersections of real data that have been
    # problematic in previous revisions of spherical_geometry

    with open(
        Path(__file__).parent / "data" / "difficult_intersections.txt", "rb"
    ) as file:
        lines = file.readlines()

    def to_array(line):
        xyzs = np.frombuffer(codecs.decode(line.strip(), "hex_codec"), dtype="<f8")
        return xyzs.reshape((len(xyzs) // 3, 3))

    for index in range(0, len(lines), 4):
        p1_points, p1_inside, p2_points, p2_inside = [
            to_array(line) for line in lines[index : index + 4]
        ]
        p1 = sphersgeo.SphericalPolygon((p1_points, p1_inside[0]))
        p2 = sphersgeo.SphericalPolygon((p2_points, p2_inside[0]))

        intersection = p1.intersection(p2)
        assert intersection.area <= p1.area + p2.area


def test_union():
    poly1 = sphersgeo.SphericalPolygon.from_cone((0, 60), 7)
    poly2 = sphersgeo.SphericalPolygon.from_cone((0, 72), 7)
    poly3 = sphersgeo.SphericalPolygon.from_cone((20, 60), 7)
    poly4 = sphersgeo.SphericalPolygon.from_cone((20, 72), 7)
    poly5 = sphersgeo.SphericalPolygon.from_cone((35, 55), 7)
    poly6 = sphersgeo.SphericalPolygon.from_cone((60, 60), 3)

    union = poly1.union(poly2).union(poly3).union(poly4).union(poly5).union(poly6)

    assert (
        union.area
        <= poly1.area + poly2.area + poly3.area + poly4.area + poly5.area + poly6.area
    )
