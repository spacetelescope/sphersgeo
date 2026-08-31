from pathlib import Path
from types import NoneType

import pytest
import sphersgeo

TEST_GEOMETRIES = pytest.helpers.read_geometry_wkt_txt(
    Path(__file__).parent / "data" / "points.csv",
    Path(__file__).parent / "data" / "strings.csv",
    Path(__file__).parent / "data" / "polygons.csv",
)
TEST_MULTIGEOMETRIES = {
    name: entry[-1]
    for name, entry in TEST_GEOMETRIES.items()
    if isinstance(
        entry[-1],
        sphersgeo.MultiGeometry,
    )
}


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_vertices(geometry: sphersgeo.AnyGeometry):
    vertices = geometry.vertices
    assert isinstance(vertices, sphersgeo.MultiSphericalPoint)
    assert len(vertices) > 0


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_boundary(geometry: sphersgeo.AnyGeometry):
    if isinstance(geometry, sphersgeo.SphericalPoint | sphersgeo.MultiSphericalPoint):
        expected_boundary_type = NoneType
    elif isinstance(geometry, sphersgeo.ArcString | sphersgeo.MultiArcString):
        expected_boundary_type = sphersgeo.MultiSphericalPoint | NoneType
    elif isinstance(
        geometry, sphersgeo.SphericalPolygon | sphersgeo.MultiSphericalPolygon
    ):
        expected_boundary_type = sphersgeo.ArcString | sphersgeo.MultiArcString

    assert isinstance(geometry.boundary, expected_boundary_type)


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_representative(geometry: sphersgeo.AnyGeometry):
    assert geometry.representative.within(geometry)


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_centroid(geometry: sphersgeo.AnyGeometry):
    assert geometry.centroid is not None


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_convex_hull(geometry: sphersgeo.AnyGeometry):
    convex_hull = geometry.convex_hull

    if isinstance(geometry, sphersgeo.SphericalPoint):
        assert isinstance(convex_hull, NoneType)
    else:
        assert isinstance(convex_hull, sphersgeo.SphericalPolygon)
        assert convex_hull.covers(geometry)
        assert convex_hull.area + 1e-11 >= geometry.area


@pytest.mark.parametrize(
    "geometry,expected",
    [(entry[-1], entry[0]) for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_area(geometry: sphersgeo.AnyGeometry, expected):
    assert geometry.area == expected


@pytest.mark.parametrize(
    "geometry,expected",
    [(entry[-1], entry[1]) for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_length(geometry: sphersgeo.AnyGeometry, expected):
    assert geometry.length == expected


@pytest.mark.parametrize(
    "multigeometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
)
def test_multigeometry_len(multigeometry):
    assert len(multigeometry) > 0


@pytest.mark.parametrize(
    "multigeometry",
    [entry[-1] for entry in TEST_MULTIGEOMETRIES.values()],
)
def test_multigeometry_append(
    multigeometry: sphersgeo.MultiGeometry,
):
    original_len = len(multigeometry)
    multigeometry.append(multigeometry[0])
    assert len(multigeometry) == original_len + 1


@pytest.mark.parametrize(
    "multigeometry",
    [entry[-1] for entry in TEST_MULTIGEOMETRIES.values()],
)
def test_multigeometry_extend(
    multigeometry: sphersgeo.MultiGeometry,
):
    original_len = len(multigeometry)
    original_length = multigeometry.length
    multigeometry.extend(multigeometry)
    assert len(multigeometry) == original_len * 2
    assert multigeometry.length == original_length * 2


@pytest.mark.parametrize(
    "multigeometry",
    [entry[-1] for entry in TEST_MULTIGEOMETRIES.values()],
)
def test_multigeometry_unary_intersection(
    multigeometry: sphersgeo.MultiGeometry,
):
    unary_intersection = multigeometry.unary_intersection
    assert isinstance(unary_intersection, type(multigeometry) | None)


@pytest.mark.parametrize(
    "multigeometry",
    [entry[-1] for entry in TEST_MULTIGEOMETRIES.values()],
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
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
        ),
        (
            TEST_GEOMETRIES["pts_northpole_repeated"][-1],
            TEST_GEOMETRIES["pt_northpole"][-1],
        ),
    ],
)
def test_multigeometry_unary_union(
    multigeometry: sphersgeo.MultiGeometry,
    expected,
):
    assert multigeometry.unary_union == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            TEST_GEOMETRIES["pt_northpole"][-1],
            TEST_GEOMETRIES["pt_southpole"][-1],
            180.0,
        ),
        (
            TEST_GEOMETRIES["pt_southpole"][-1],
            TEST_GEOMETRIES["pt_equator1"][-1],
            90.0,
        ),
        (
            TEST_GEOMETRIES["pt_equator1"][-1],
            TEST_GEOMETRIES["pt_equator2"][-1],
            90.0,
        ),
        (
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
            TEST_GEOMETRIES["pts_southpole_equator1"][-1],
            0.0,
        ),
        (
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
            TEST_GEOMETRIES["pts_equator1_equator2"][-1],
            90.0,
        ),
        (
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
            TEST_GEOMETRIES["pt_southpole"][-1],
            90.0,
        ),
        (
            TEST_GEOMETRIES["pts_northpole_southpole"][-1],
            TEST_GEOMETRIES["pt_equator1"][-1],
            90.0,
        ),
    ],
)
def test_distance(a, b, expected):
    assert a.distance(a) == 0.0
    assert b.distance(b) == 0.0
    assert a.distance(b) == expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_equals_self(geometry: sphersgeo.AnyGeometry):
    assert geometry == geometry


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["arc_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pgn_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator2"], False),
        # TODO add more expected equals
    ],
)
def test_equals(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]
    assert (geometry_a == geometry_b) is expected
    assert (geometry_a != geometry_b) is not expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_covers_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.covers(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["arc_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pgn_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator2"], False),
        (
            TEST_GEOMETRIES["pts_northpole_southpole"],
            TEST_GEOMETRIES["pt_northpole"],
            True,
        ),
        (TEST_GEOMETRIES["pts_other1"], TEST_GEOMETRIES["pt_northpole"], False),
        (TEST_GEOMETRIES["arc_diagonal1"], TEST_GEOMETRIES["pt_origin"], True),
        (TEST_GEOMETRIES["arc_meridion1"], TEST_GEOMETRIES["pt_origin"], True),
        (TEST_GEOMETRIES["arc_equator1"], TEST_GEOMETRIES["pt_origin"], True),
        # TODO add more expected covers
    ],
)
def test_covers(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.covers(geometry_b) is expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_not_contains_within_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.contains(geometry)
    assert not geometry.within(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["arc_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pgn_other1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator1"], False),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_equator2"], False),
        # TODO add more expected contains / within
    ],
)
def test_contains_within(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.contains(geometry_b) is expected
    assert geometry_a.within(geometry_b) is not expected
    assert geometry_b.within(geometry_a) is expected
    assert geometry_b.contains(geometry_a) is not expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_not_crosses_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.crosses(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["arc_meridion1"], TEST_GEOMETRIES["arc_equator1"], True),
        (TEST_GEOMETRIES["arc_meridion1"], TEST_GEOMETRIES["arc_diagonal1"], True),
        (TEST_GEOMETRIES["arc_meridion1"], TEST_GEOMETRIES["arc_vertical1"], False),
        # TODO add expected crosses
    ],
)
def test_crosses(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.crosses(geometry_b) is expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_touches_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.touches(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (
            TEST_GEOMETRIES["pt_northpole"],
            TEST_GEOMETRIES["pts_northpole_southpole"],
            True,
        ),
        # TODO add expected touches
    ],
)
def test_touches(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.touches(geometry_b) is expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_not_overlaps_self(geometry: sphersgeo.AnyGeometry):
    assert not geometry.overlaps(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (
            TEST_GEOMETRIES["pt_northpole"],
            TEST_GEOMETRIES["pts_northpole_southpole"],
            True,
        ),
        # TODO add expected overlaps
    ],
)
def test_overlaps(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.overlaps(geometry_b) is expected


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_intersects_self(geometry: sphersgeo.AnyGeometry):
    assert geometry.intersects(geometry)
    assert not geometry.disjoint(geometry)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"], False),
        (
            TEST_GEOMETRIES["pt_northpole"],
            TEST_GEOMETRIES["pts_northpole_southpole"],
            True,
        ),
        (TEST_GEOMETRIES["pgn_other5"], TEST_GEOMETRIES["pgn_other7"], True),
        (TEST_GEOMETRIES["pgn_other10"], TEST_GEOMETRIES["pgn_other11"], True),
        (TEST_GEOMETRIES["pgn_other10"], TEST_GEOMETRIES["pgn_other12"], True),
        (TEST_GEOMETRIES["pgn_other10"], TEST_GEOMETRIES["pgn_other13"], True),
        (TEST_GEOMETRIES["pgn_other14"], TEST_GEOMETRIES["pgn_other15"], False),
        (TEST_GEOMETRIES["pgn_intersectioncrash1"], TEST_GEOMETRIES["pgn_intersectioncrash2"], True),
        (TEST_GEOMETRIES["pgn_intersectioncrash3"], TEST_GEOMETRIES["pgn_intersectioncrash4"], True),
        # TODO add expected intersects / disjoint
    ],
)
def test_intersects(a, b, expected):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    assert geometry_a.intersects(geometry_b) is expected
    assert geometry_a.disjoint(geometry_b) is not expected



@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_self_intersection(geometry: sphersgeo.AnyGeometry):
    assert geometry.intersection(geometry) == geometry


@pytest.mark.parametrize(
    "a,b",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"]),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pts_northpole_southpole"]),
        # TODO add expected intersections
    ],
)
def test_intersection(a, b):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    intersection = geometry_a.intersection(geometry_b)

    assert isinstance(intersection, sphersgeo.GeometryCollection)


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_self_difference(geometry: sphersgeo.AnyGeometry):
    assert geometry.difference(geometry) is None


@pytest.mark.parametrize(
    "a,b",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"]),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pts_northpole_southpole"]),
        # TODO add expected differences
    ],
)
def test_difference(a, b):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    difference = geometry_a.difference(geometry_b)

    assert isinstance(difference, sphersgeo.MultiGeometry | None)
    assert difference.area <= geometry_a.area + geometry_b.area


@pytest.mark.parametrize(
    "geometry",
    [entry[-1] for entry in TEST_GEOMETRIES.values()],
    ids=TEST_GEOMETRIES.keys(),
)
def test_self_union(geometry: sphersgeo.AnyGeometry):
    assert geometry.union(geometry) == geometry


@pytest.mark.parametrize(
    "a,b",
    [
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pt_southpole"]),
        (TEST_GEOMETRIES["pt_northpole"], TEST_GEOMETRIES["pts_northpole_southpole"]),
        # TODO add expected unions
    ],
)
def test_union(a, b):
    geometry_a: sphersgeo.AnyGeometry = a[-1]
    geometry_b: sphersgeo.AnyGeometry = b[-1]

    union = geometry_a.union(geometry_b)

    assert union.area <= geometry_a.area + geometry_b.area
    # assert union.covers(geometry_a)
    # assert union.covers(geometry_b)
