from typing import Any, TypeAlias, overload

import numpy as np

__all__ = [
    "ArcString",
    "MultiArcString",
    "MultiSphericalPoint",
    "MultiSphericalPolygon",
    "SphericalPoint",
    "SphericalPolygon",
]

AnyGeometry: TypeAlias = (
    SphericalPoint
    | MultiSphericalPoint
    | ArcString
    | MultiArcString
    | SphericalPolygon
    | MultiSphericalPolygon
)

AnyGeometryInputs: TypeAlias = (
    SphericalPointInputs
    | MultiSphericalPointInputs
    | ArcStringInputs
    | MultiArcStringInputs
    | SphericalPolygonInputs
    | MultiSphericalPolygonInputs
)

GeometryCollection: TypeAlias = tuple[
    MultiSphericalPoint, MultiArcString, MultiSphericalPolygon
]


class Geometry:
    @property
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def boundary(self) -> MultiArcString | ArcString | MultiSphericalPoint | None:
        """
        lower dimension geometry that bounds this geometry's interior

        The boundary of a polygon is a closed arcstring,
        the boundary of an arcstring is two endpoints (unless closed),
        and the boundary of a point (and a closed arcstring) is null.
        """

    @property
    def representative(self) -> SphericalPoint:
        """point guaranteed to be within this geometry"""

    @property
    def centroid(self) -> SphericalPoint:
        """mean position of all possible points within this geometry"""

    @property
    def convex_hull(self) -> SphericalPolygon | None:
        """smallest convex polygon containing this geometry"""

    @property
    def area(self) -> float:
        """surface area of this geometry in square degrees"""

    @property
    def length(self) -> float:
        """angular length of this geometry in degrees"""

    @property
    def wkt(self) -> str:
        """well-known text representation of this geometry in degrees"""

    def distance(
        self,
        other: AnyGeometryInputs,
    ) -> float:
        """shortest geodesic from this geometry to another"""

    def equals(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this and the other geometry's interiors are identical and the geometry types are the same.

        For further explanation of Equals see `ArcGIS Equals <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#equals>`_
        or Shapely's `object.equals`.
        """

    def covers(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether the other geometry is a subset of this geometry
        (every point of the other geometry is a point on the interior OR boundary of this geometry).
        """

    def contains(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this geometry covers the other geometry AND the interiors share at least one point.

        Contains is the inverse of Within.

        For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
        or Shapely's `object.contains`.
        """

    def within(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether the other geometry covers this geometry AND the interiors share at least one point.

        Within is the inverse of Contains.

        For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
        or Shapely's `object.contains`.
        """

    def crosses(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this arcstring / polygon and the other arcstring / polygon share only SOME (not all) interior points, but do NOT overlap.

        Two arcstrings cross if they meet at point(s) only, and at least one of the shared points is internal to both arcstrings.
        An arcstring and polygon cross if they share an arcstring on the interior of the polygon, which is NOT equal to the entire arcstring.

        For further explanation of Crosses see `ArcGIS Crosses <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#crosses>`_
        or Shapely's `object.crosses`.
        """

    def touches(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this and the other geometry share any vertices but do not overlap.

        For further explanation of Touches see `ArcGIS Touches <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#touches>`_
        or Shapely's `object.touches`.
        """

    def overlaps(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this and the other geometry are of the same geometry type,
        AND their intersection is also of the same geometry type BUT is not equal to either.

        For further explanation of Overlaps see `ArcGIS Overlaps <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#overlaps>`_
        or Shapely's `object.overlaps`.
        """

    def intersects(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this and the other geometry share ANY point(s).
        If this geometries contains, is within, crosses, touches, or overlaps the other geometry, they intersect.

        For further explanation of Intersects see `ArcGIS Intersects <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#intersects>`_
        or Shapely's `object.intersects`.
        """

    def disjoint(
        self,
        other: AnyGeometryInputs,
    ) -> bool:
        """
        Whether this and the other geometry do NOT share ANY point(s).

        Disjoint is the inverse of Intersects.

        For further explanation of Disjoint see `ArcGIS Disjoint <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#disjoint>`_
        or Shapely's `object.disjoint`.
        """

    @overload
    def intersection(
        self: SphericalPoint | MultiSphericalPoint,
        other,
    ) -> tuple[MultiSphericalPoint | None, None, None]: ...
    @overload
    def intersection(
        self,
        other: SphericalPointInputs | MultiSphericalPointInputs,
    ) -> tuple[MultiSphericalPoint | None, None, None]: ...
    @overload
    def intersection(
        self: ArcString | MultiArcString,
        other,
    ) -> tuple[MultiSphericalPoint | None, MultiArcString | None, None]: ...
    @overload
    def intersection(
        self,
        other: ArcStringInputs | MultiArcStringInputs,
    ) -> tuple[MultiSphericalPoint, MultiArcString, None]: ...
    def intersection(
        self,
        other: AnyGeometryInputs,
    ) -> GeometryCollection:
        """
        region(s) of this geometry that overlap the other geometry

        Intersection is the inverse of Difference.

        For further explanation of Intersection see Shapely's `object.intersection`.
        """

    @overload
    def difference(
        self: SphericalPoint | MultiSphericalPoint,
        other,
    ) -> MultiSphericalPoint | None: ...
    @overload
    def difference(
        self: ArcStringInputs | MultiArcStringInputs,
        other,
    ) -> MultiArcString | None: ...
    @overload
    def difference(
        self: SphericalPolygon | MultiSphericalPolygon,
        other,
    ) -> MultiSphericalPolygon | None: ...
    def difference(
        self,
        other: AnyGeometryInputs,
    ) -> MultiGeometry | None:
        """
        region(s) of this geometry that do not intersect or overlap with the other geometry

        Difference is the inverse of Intersection.

        For further explanation of Difference see Shapely's `object.difference`.
        """

    def union(self, other: AnyGeometryInputs) -> GeometryCollection:
        """
        dissolved union of this geometry and the other geometry

        For further explanation of Union see Shapely's `object.union`.
        """


class MultiGeometry:
    @overload
    def parts(self: MultiSphericalPoint) -> list[SphericalPoint]: ...
    @overload
    def parts(self: MultiArcString) -> list[ArcString]: ...
    @overload
    def parts(self: MultiSphericalPolygon) -> list[SphericalPolygon]: ...
    @property
    def parts(
        self,
    ) -> list[SphericalPoint] | list[ArcString] | list[SphericalPolygon]:
        """geometries comprising this collection"""

    def __len__(self) -> int:
        """number of geometries in this collection"""

    @overload
    def __getitem__(
        self: MultiSphericalPoint, index
    ) -> SphericalPoint | MultiSphericalPoint | None: ...
    @overload
    def __getitem__(
        self: MultiArcString, index
    ) -> ArcString | MultiArcString | None: ...
    @overload
    def __getitem__(
        self: MultiSphericalPolygon, index
    ) -> SphericalPolygon | MultiSphericalPolygon | None: ...
    def __getitem__(self, index: int) -> None: ...

    @overload
    def append(self: MultiSphericalPoint, other: SphericalPointInputs): ...
    @overload
    def append(self: MultiArcString, other: ArcStringInputs): ...
    @overload
    def append(self: MultiSphericalPolygon, other: SphericalPolygonInputs): ...
    def append(self, other):
        """append the geometry to this collection"""

    @overload
    def extend(self: MultiSphericalPoint, other: MultiSphericalPointInputs): ...
    @overload
    def extend(self: MultiArcString, other: MultiArcStringInputs): ...
    @overload
    def extend(self: MultiSphericalPolygon, other: MultiSphericalPolygonInputs): ...
    def extend(self, other):
        """extend this collection with geometries from the other collection"""

    @overload
    def unary_union(self: MultiSphericalPoint) -> MultiSphericalPoint: ...
    @overload
    def unary_union(self: MultiArcString) -> MultiArcString: ...
    @overload
    def unary_union(self: MultiSphericalPolygon) -> MultiSphericalPolygon: ...
    def unary_union(
        self,
    ) -> MultiGeometry:
        """
        dissolved union of these geometries

        For further explanation of Unary Union see Shapely's `unary_union`.
        """

    @overload
    def unary_intersection(self: MultiSphericalPoint) -> MultiSphericalPoint | None: ...
    @overload
    def unary_intersection(self: MultiArcString) -> MultiArcString | None: ...
    @overload
    def unary_intersection(
        self: MultiSphericalPolygon,
    ) -> MultiSphericalPolygon | None: ...
    def unary_intersection(
        self,
    ) -> MultiGeometry | None:
        """
        overlapping regions between these geometries, if any

        For further explanation of Intersection see Shapely's `object.intersection`.
        """

    @overload
    def unary_symmetric_difference(
        self: MultiSphericalPoint,
    ) -> MultiSphericalPoint | None: ...
    @overload
    def unary_symmetric_difference(self: MultiArcString) -> MultiArcString | None: ...
    @overload
    def unary_symmetric_difference(
        self: MultiSphericalPolygon,
    ) -> MultiSphericalPolygon | None: ...
    def unary_symmetric_difference(
        self,
    ) -> MultiGeometry | None:
        """
        non-overlapping regions between these geometries

        For further explanation of Symmetric Difference see Shapely's `object.symmetric_difference`.
        """


SphericalPointInputs: TypeAlias = (
    tuple[float, float]
    | np.ndarray[tuple[np.typing.Literal[2]], np.dtype[np.float64]]
    | tuple[float, float, float]
    | np.ndarray[tuple[np.typing.Literal[3]], np.dtype[np.float64]]
    | list[float]
    | str
    | SphericalPoint
)


class SphericalPoint(Geometry):
    """single point on the sphere, represented internally as a 3-dimensional Cartesian point (X, Y, Z) with origin at the center of the unit sphere"""

    def __init__(
        self,
        point: SphericalPointInputs,
    ) -> SphericalPoint:
        """
        Create a `SphericalPoint` from angular coordinates (longitude, latitude)::

            from sphersgeo import SphericalPoint

            a = SphericalPoint((60.0, 30.0))
            b = SphericalPoint((60.0, 0.0))
            c = SphericalPoint((-30.0, -30.0))

        … or Cartesian coordinates (X, Y, Z)::

            from sphersgeo import SphericalPoint

            a = SphericalPoint((0.43301270189221946, 0.75, 0.5))
            b = SphericalPoint((0.5, 0.8660254037844386, 0.0))
            c = SphericalPoint((0.75, -0.4330127018922193, -0.5))
            d = SphericalPoint((0.0, 0.0, 1.0))
            e = SphericalPoint((0.0, 0.0, -1.0))
        """

    @property
    def xyz(self) -> tuple[float, float, float]:
        """coordinates of this point as X, Y, and Z from the center of the sphere"""

    @property
    def lonlat(self) -> tuple[float, float]:
        """coordinates of this point as longitude and latitude"""

    @property
    def antipode(self) -> SphericalPoint:
        """antipodal point on the opposite side of the sphere"""

    def two_arc_angle(self, a: SphericalPointInputs, c: SphericalPointInputs) -> float:
        """
        given three points on the sphere:

          - `a`
          - `b` (this point)
          - `c`

        retrieves the turning angle, in radians, at `b` formed by arcs `ab` and `bc`
        """

    def colinear(self, a: SphericalPointInputs, b: SphericalPointInputs) -> bool:
        """whether this point lies on an arc between two other points"""

    def is_clockwise_turn(
        self, a: SphericalPointInputs, b: SphericalPointInputs
    ) -> bool:
        """whether the angle formed between this point and two other points is a clockwise turn"""

    def interpolate_points(
        self, end: SphericalPointInputs, n: int
    ) -> MultiSphericalPoint:
        """create n number of equally-spaced points on the arc between this point and another point"""

    @property
    def vector_length(self) -> float:
        """length of the underlying xyz vector"""

    def vector_cross(self, other: SphericalPointInputs) -> SphericalPoint:
        """cross product of this xyz vector with another xyz vector"""

    def vector_dot(self, other: SphericalPointInputs) -> float:
        """dot product of this xyz vector with another xyz vector"""

    def vector_rotate_around(
        self, other: SphericalPointInputs, theta: float
    ) -> SphericalPoint:
        """rotate this xyz vector by theta radians around another xyz vector"""

    def to(self, other: SphericalPointInputs) -> ArcString:
        """arc to another point"""

    @property
    def boundary(self) -> None: ...
    def __add__(self, other: SphericalPointInputs) -> SphericalPoint: ...
    def __sub__(self, other: SphericalPointInputs) -> SphericalPoint: ...
    def __mul__(self, other: SphericalPointInputs) -> SphericalPoint: ...
    def __div__(self, other: SphericalPointInputs) -> SphericalPoint: ...
    def __eq__(self, other) -> bool: ...


MultiSphericalPointInputs: TypeAlias = (
    list[SphericalPointInputs]
    | np.ndarray[tuple[Any, np.typing.Literal[2]], np.dtype[np.float64]]
    | np.ndarray[tuple[Any, np.typing.Literal[3]], np.dtype[np.float64]]
    | str
    | MultiSphericalPoint
)


class MultiSphericalPoint(Geometry, MultiGeometry):
    """collection of multiple points on the sphere"""

    def __init__(
        self,
        points: MultiSphericalPointInputs,
    ) -> MultiSphericalPoint:
        """
        Create a `MultiSphericalPoint` from a list of `SphericalPoint` s::

            from sphersgeo import SphericalPoint, MultiSphericalPoint

            a = SphericalPoint((60.0, 30.0))
            b = SphericalPoint((60.0, 0.0))
            c = SphericalPoint((0.75, -0.4330127018922193, -0.5))

            abc = MultiSphericalPoint([a, b, c])

        … or from the inputs required to make a list of `SphericalPoint` s::

            from sphersgeo import MultiSphericalPoint

            abc = MultiSphericalPoint(
                [(60.0, 30.0), (60.0, 0.0), (0.75, -0.4330127018922193, -0.5)]
            )

        … or from a `numpy.ndarray` of shape Nx2 (longitude, latitude) or Nx3 (X, Y, Z)::

            import numpy as np
            from sphersgeo import MultiSphericalPoint

            abc = MultiSphericalPoint(np.array([(60.0, 30.0), (60.0, 0.0), (-30.0, -30.0)]))
            abc = MultiSphericalPoint(
                np.array(
                    [
                        (0.43301270189221946, 0.75, 0.5),
                        (0.5, 0.8660254037844386, 0.0),
                        (0.75, -0.4330127018922193, -0.5),
                    ]
                )
            )
        """

    @property
    def xyzs(
        self,
    ) -> np.ndarray[tuple[Any, np.typing.Literal[3]], np.dtype[np.float64]]:
        """coordinates of these points as X, Y, and Z (Nx3 `numpy.ndarray`)"""

    @property
    def lonlats(
        self,
    ) -> np.ndarray[tuple[Any, np.typing.Literal[2]], np.dtype[np.float64]]:
        """coordinates of these points as longitude and latitude (Nx2 `numpy.ndarray`)"""

    def nearest(self, other: SphericalPointInputs) -> tuple[SphericalPoint, float]:
        """retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point across the unit sphere"""

    @property
    def vectors_lengths(self) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
        """lengths of the underlying (X, Y, Z) vectors"""

    @property
    def boundary(self) -> None: ...
    def __iadd__(self, other: MultiSphericalPointInputs): ...
    def __add__(self, other: MultiSphericalPointInputs) -> MultiSphericalPoint: ...
    def __eq__(self, other) -> bool: ...


ArcStringInputs: TypeAlias = MultiSphericalPointInputs | ArcString


class ArcString(Geometry):
    """
    series of great circle arcs across the sphere, which can be open at the end or closed (returns to the initial point)

    A great circle arc is the shortest geodesic distance over the surface of the sphere between any two points.
    Arcstrings are comprised of an **ordered** collection of spherical points.
    Arcstrings can also be **closed**, in which case the final point is considered as connected back to the first point.
    """

    def __init__(
        self,
        points: ArcStringInputs,
        closed: bool = False,
    ) -> ArcString:
        """
        Create an `ArcString` from a `MultiSphericalPoint`::

            from sphersgeo import ArcString, MultiSphericalPoint

            abc = ArcString(
                MultiSphericalPoint([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)]), closed=True
            )

        … or from the inputs required to make a `MultiSphericalPoint`::

            from sphersgeo import ArcString

            de = ArcString(
                [
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ]
            )
        """

    @property
    def lengths(self) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
        """angle subtended on the sphere by each arc"""

    @property
    def midpoints(self) -> MultiSphericalPoint:
        """midpoints of each arc"""

    @property
    def arcs(self) -> list[ArcString]:
        """decomposes this arcstring into individual arcs of 2 points each"""

    @property
    def closed(self) -> bool:
        """whether this arcstring is "closed" (the last vertex is considered connected to the first)"""

    @closed.setter
    def closed(self, closed: bool):
        """ "close" this arcstring (consider the last vertex as connected to the first)"""

    def crossings(self, other: ArcStringInputs | MultiArcStringInputs):
        """remove redundant vertices that already lie along an arc in this arcstring"""

    @property
    def crosses_self(self) -> bool:
        """whether this arcstring crosses itself"""

    @property
    def crossings_with_self(self) -> MultiSphericalPoint:
        """points at which this arcstring crosses itself"""

    def adjoins(self, other: ArcStringInputs) -> bool:
        """whether this arcstring shares endpoints with another (ignoring closed arcstrings which have no endpoints)"""

    def simplify(self):
        """remove redundant vertices that already lie along an arc in this arcstring"""

    @property
    def boundary(self) -> MultiSphericalPoint | None: ...
    def __eq__(self, other) -> bool: ...


MultiArcStringInputs: TypeAlias = list[ArcStringInputs] | str | MultiArcString


class MultiArcString(Geometry):
    """collection of multiple series of great circle arcs (arcstrings)"""

    def __init__(
        self,
        arcstrings: MultiArcStringInputs,
    ) -> MultiArcString:
        """
        Create a `MultiArcString` from a list of `ArcString` s::

            from sphersgeo import ArcString, MultiArcString

            abc = ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)], closed=True)
            de = ArcString([(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
            abc_de = MultiArcString([abc, de])

        """

    @property
    def boundary(self) -> MultiSphericalPoint | None: ...
    def __iadd__(self, other: MultiArcStringInputs): ...
    def __add__(self, other: MultiArcStringInputs) -> MultiArcString: ...
    def __eq__(self, other) -> bool: ...


SphericalPolygonInputs: TypeAlias = (
    ArcStringInputs
    | tuple[ArcStringInputs, SphericalPointInputs]
    | str
    | SphericalPolygon
)


class SphericalPolygon(Geometry):
    """
    polygon on the sphere, represented by a **counterclockwise** `ArcString` (assumed to be closed) to form the boundary, and a `SphericalPoint` guaranteed to be inside the polygon (inferred if not provided)

    .. attention::
        The inside of a `SphericalPolygon` is **always** assumed to be the area **to the left of the boundary**.
        Thus, passing a clockwise boundary will make the "inside" of the polygon
        the **entire surface area of the sphere sans the area enclosed by the boundary**.

    .. attention:: `SphericalPolygon` s in `sphersgeo` do NOT have holes.
    """

    def __init__(
        self,
        polygon: SphericalPolygonInputs,
    ) -> SphericalPolygon:
        """
        Create a `SphericalPolygon` from an `ArcString`::

            from sphersgeo import ArcString, SphericalPolygon

            abc = SphericalPolygon(ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)]))

        … or from the inputs required to make an `ArcString`::

            from sphersgeo import SphericalPolygon

            abc = SphericalPolygon(
                [
                    (0.43301270189221946, 0.75, 0.5),
                    (0.5, 0.8660254037844386, 0.0),
                    (0.75, -0.4330127018922193, -0.5),
                ]
            )
        """

    @classmethod
    def from_cone(
        self,
        center: SphericalPointInputs,
        radius: float,
        steps: int = 16,
    ) -> SphericalPolygon: ...
    @property
    def convex(self) -> bool:
        """whether this polygon is convex, that is, all possible arcs between points inside the polygon can never leave the enclosed space"""

    @property
    def inverse(self) -> SphericalPolygon: ...
    def simplify(self):
        """remove redundant vertices that already lie along an arc in the boundary"""

    @property
    def boundary(self) -> ArcString: ...
    def __eq__(self, other) -> bool: ...


MultiSphericalPolygonInputs: TypeAlias = (
    list[SphericalPolygonInputs] | str | MultiSphericalPolygon
)


class MultiSphericalPolygon(Geometry):
    """collection of multiple polygons on the sphere"""

    def __init__(self, polygons: MultiSphericalPolygonInputs) -> MultiSphericalPolygon:
        """
        Create a `MultiSphericalPolygon` from a list of `SphericalPolygon` s::

            from sphersgeo import SphericalPolygon, MultiSphericalPolygon

            abc_def = MultiSphericalPolygon(
                [
                    SphericalPolygon(
                        [
                            (0.43301270189221946, 0.75, 0.5),
                            (0.5, 0.8660254037844386, 0.0),
                            (0.75, -0.4330127018922193, -0.5),
                        ]
                    ),
                    SphericalPolygon(
                        [
                            (0.0, 0.0, 1.0),
                            (0.0, 0.0, -1.0),
                            (1.0, 1.0, 0.0),
                        ]
                    ),
                ]
            )

        … or from the inputs required to make a list of `SphericalPolygon` s::

            from sphersgeo import MultiSphericalPolygon

            abc_def = MultiSphericalPolygon(
                [
                    [
                        (0.43301270189221946, 0.75, 0.5),
                        (0.5, 0.8660254037844386, 0.0),
                        (0.75, -0.4330127018922193, -0.5),
                    ],
                    [
                        (0.0, 0.0, 1.0),
                        (0.0, 0.0, -1.0),
                        (1.0, 1.0, 0.0),
                    ],
                ]
            )
        """

    @property
    def boundary(self) -> MultiArcString: ...
    def __iadd__(self, other: MultiSphericalPolygonInputs): ...
    def __add__(self, other: MultiSphericalPolygonInputs) -> MultiSphericalPolygon: ...
    def __eq__(self, other) -> bool: ...


def from_wkt(wkt: str) -> AnyGeometry:
    """construct geometry from well-known text representation"""
