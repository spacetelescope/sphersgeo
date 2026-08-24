from pathlib import Path

import pytest

import sphersgeo


def expected_boundary_type(geometry: sphersgeo.AnyGeometry) -> sphersgeo.AnyGeometry:
    if isinstance(geometry, sphersgeo.SphericalPoint | sphersgeo.MultiSphericalPoint):
        expected_type = None
    elif isinstance(geometry, sphersgeo.ArcString | sphersgeo.MultiArcString):
        expected_type = sphersgeo.MultiSphericalPoint
    elif isinstance(
        geometry, sphersgeo.SphericalPolygon | sphersgeo.MultiSphericalPolygon
    ):
        expected_type = sphersgeo.ArcString | sphersgeo.MultiArcString
    return expected_type


def read_geometry_wkt_txt(
    *filenames: Path,
) -> dict[
    str,
    sphersgeo.AnyGeometry,
]:
    lines = []
    for filename in filenames:
        with open(filename) as geometries_file:
            lines.extend(geometries_file.readlines())

    geometries = {}
    for line in lines:
        name, wkt = line.split(",", 1)
        geometries[name] = sphersgeo.from_wkt(wkt)
    return geometries


TEST_GEOMETRIES = read_geometry_wkt_txt(
    Path(__file__).parent / "data" / "points.csv",
    Path(__file__).parent / "data" / "strings.csv",
    Path(__file__).parent / "data" / "polygons.csv",
)

TEST_MULTIGEOMETRIES = {
    name: geometry
    for name, geometry in TEST_GEOMETRIES.items()
    if isinstance(geometry, sphersgeo.MultiGeometry)
}


@pytest.mark.parametrize(
    "geometry",
    TEST_GEOMETRIES.values(),
    ids=TEST_GEOMETRIES.keys(),
)
def test_vertices(geometry: sphersgeo.AnyGeometry):
    vertices = geometry.vertices
    assert isinstance(vertices, sphersgeo.MultiSphericalPoint)
    assert len(vertices) > 0


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_boundary(geometry: sphersgeo.AnyGeometry):
    assert isinstance(geometry.boundary, expected_boundary_type(geometry))


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_representative(geometry: sphersgeo.AnyGeometry):
    assert geometry.representative.within(geometry)


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_centroid(geometry: sphersgeo.AnyGeometry):
    assert geometry.centroid is not None


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_convex_hull(geometry: sphersgeo.AnyGeometry):
    convex_hull = geometry.convex_hull
    assert isinstance(convex_hull, sphersgeo.SphericalPolygon)
    assert convex_hull.covers(geometry)
    assert convex_hull.area >= geometry.area


@pytest.mark.parametrize(
    "geometry,expected",
    # TODO fill out expected areas
    zip(TEST_GEOMETRIES.values(), [0.0 for _ in TEST_GEOMETRIES]),
    ids=TEST_GEOMETRIES.keys(),
)
def test_area(geometry: sphersgeo.AnyGeometry, expected):
    assert geometry.area == expected


@pytest.mark.parametrize(
    "geometry,expected",
    # TODO fill out expected lengths
    zip(TEST_GEOMETRIES.values(), [0.0 for _ in TEST_GEOMETRIES]),
    ids=TEST_GEOMETRIES.keys(),
)
def test_length(geometry: sphersgeo.AnyGeometry, expected):
    assert geometry.length == expected


@pytest.mark.parametrize(
    "multigeometry,expected",
    zip(
        TEST_MULTIGEOMETRIES.values(),
        [0 for _ in TEST_MULTIGEOMETRIES],
    ),
)
def test_multigeometry_len(
    multigeometry,
    expected,
):
    assert len(multigeometry) == expected


@pytest.mark.parametrize(
    "multigeometry",
    TEST_MULTIGEOMETRIES.values(),
)
def test_multigeometry_append(multigeometry: sphersgeo.MultiGeometry):
    original_length = len(multigeometry)
    multigeometry.append(multigeometry[0])
    assert len(multigeometry) == original_length + 1


@pytest.mark.parametrize(
    "multigeometry",
    TEST_MULTIGEOMETRIES.values(),
)
def test_multigeometry_extend(multigeometry: sphersgeo.MultiGeometry):
    original_length = len(multigeometry)
    multigeometry.extend(multigeometry)
    assert len(multigeometry) == original_length * 2


@pytest.mark.parametrize(
    "multigeometry",
    TEST_MULTIGEOMETRIES.values(),
)
def test_multigeometry_unary_intersection(multigeometry: sphersgeo.MultiGeometry):
    unary_intersection = multigeometry.unary_intersection
    assert isinstance(unary_intersection, type(multigeometry) | None)


@pytest.mark.parametrize(
    "multigeometry",
    TEST_MULTIGEOMETRIES.values(),
)
def test_multigeometry_unary_symmetric_difference(
    multigeometry: sphersgeo.MultiGeometry,
):
    unary_symmetric_difference = multigeometry.unary_symmetric_difference
    assert isinstance(unary_symmetric_difference, type(multigeometry) | None)


@pytest.mark.parametrize(
    "multigeometry,expected",
    [
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pts_northpole_southpole"],
        ),
        (
            TEST_GEOMETRIES["pts_northpole_repeated"],
            TEST_GEOMETRIES["pt_northpole"],
        ),
    ],
)
def test_multigeometry_unary_union(multigeometry: sphersgeo.MultiGeometry, expected):
    assert multigeometry.unary_union == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            TEST_GEOMETRIES["pt_northpole"],
            TEST_GEOMETRIES["pt_southpole"],
            180.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pt_southpole"],
            TEST_GEOMETRIES["pt_equator1"],
            90.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pt_equator1"],
            TEST_GEOMETRIES["pt_equator2"],
            90.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pts_southpole_equator1"],
            0.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pts_equator1_equator2"],
            90.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pt_southpole"],
            90.0,
        ),
    ],
    [
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pt_equator1"],
            90.0,
        ),
    ],
)
def test_distance(a, b, expected):
    assert a.distance(a) == 0.0
    assert b.distance(b) == 0.0
    assert a.distance(b) == expected


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_equals_self(geometry: sphersgeo.AnyGeometry):
    assert geometry == geometry


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected equals
    [
        (),
    ],
)
def test_equals(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_covers_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.covers(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected covers
    [
        (),
    ],
)
def test_covers(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_not_contains_within_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.contains(geometry)
    assert not geometry.within(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected contains / within
    [
        (),
    ],
)
def test_contains_within(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_not_crosses_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.crosses(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected crosses
    [
        (),
    ],
)
def test_crosses(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_touches_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.touches(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected touches
    [
        (),
    ],
)
def test_touches(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_not_overlaps_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.overlaps(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected overlaps
    [
        (),
    ],
)
def test_overlaps(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_intersects_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.intersects(geometry)
    assert not geometry.disjoint(geometry)


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected intersects / disjoint
    [
        (),
    ],
)
def test_intersects_disjoint(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_self_intersection(geometry: sphersgeo.AnyGeometry):
    assert geometry.intersection(geometry) == geometry


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected intersections
    [
        (),
    ],
)
def test_intersection(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_self_difference(geometry: sphersgeo.AnyGeometry):
    assert geometry.difference(geometry) == None


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected differences
    [
        (),
    ],
)
def test_difference(a, b, expected):
    pass


@pytest.mark.parametrize(
    "geometry", TEST_GEOMETRIES.values(), ids=TEST_GEOMETRIES.keys()
)
def test_self_union(geometry: sphersgeo.AnyGeometry):
    assert geometry.union(geometry) == geometry


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    "a,b,expected",
    # TODO fill out expected unions
    [
        (),
    ],
)
def test_union(a, b, expected):
    pass
