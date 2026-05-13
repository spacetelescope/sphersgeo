from typing import Any

import numpy as np

__all__ = [
    "SphericalPoint",
    "MultiSphericalPoint",
    "ArcString",
    "MultiArcString",
    "SphericalPolygon",
    "MultiSphericalPolygon",
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
        ...

    @property
    def representative(self) -> SphericalPoint:
        """point guaranteed to be within this geometry"""
        ...

    @property
    def centroid(self) -> SphericalPoint:
        """mean position of all possible points within this geometry"""
        ...

    @property
    def convex_hull(self) -> SphericalPolygon | None:
        """smallest convex polygon containing this geometry"""
        ...

    @property
    def area(self) -> float:
        """surface area of this geometry in square degrees"""
        ...

    @property
    def length(self) -> float:
        """angular length of this geometry in degrees"""
        ...


class GeometricRelationships:
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this and the other geometry's interiors are identical and the geometry types are the same.

        For further explanation of Equals see `ArcGIS Equals <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#equals>`_
        or Shapely's `object.equals`.
        """
        ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this and the other geometry share ANY point(s).
        If this geometries contains, is within, crosses, touches, or overlaps the other geometry, they intersect.

        For further explanation of Intersects see `ArcGIS Intersects <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#intersects>`_
        or Shapely's `object.intersects`.
        """
        ...

    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this and the other geometry share any vertices but do not overlap.

        For further explanation of Touches see `ArcGIS Touches <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#touches>`_
        or Shapely's `object.touches`.
        """
        ...

    def disjoint(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this and the other geometry do NOT share ANY point(s).

        Disjoint is the inverse of Intersects.

        For further explanation of Disjoint see `ArcGIS Disjoint <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#disjoint>`_
        or Shapely's `object.disjoint`.
        """
        ...

    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this arcstring / polygon and the other arcstring / polygon share only SOME (not all) interior points, but do NOT overlap.

        Two arcstrings cross if they meet at point(s) only, and at least one of the shared points is internal to both arcstrings.
        An arcstring and polygon cross if they share an arcstring on the interior of the polygon, which is NOT equal to the entire arcstring.

        For further explanation of Crosses see `ArcGIS Crosses <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#crosses>`_
        or Shapely's `object.crosses`.
        """
        ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether the other geometry covers this geometry AND the interiors share at least one point.

        Within is the inverse of Contains.

        For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
        or Shapely's `object.contains`.
        """
        ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this geometry covers the other geometry AND the interiors share at least one point.

        Contains is the inverse of Within.

        For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
        or Shapely's `object.contains`.
        """
        ...

    def overlaps(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether this and the other geometry are of the same geometry type,
        AND their intersection is also of the same geometry type BUT is not equal to either.

        For further explanation of Overlaps see `ArcGIS Overlaps <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#overlaps>`_
        or Shapely's `object.overlaps`.
        """
        ...

    def covers(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool:
        """
        Whether the other geometry is a subset of this geometry
        (every point of the other geometry is a point on the interior OR boundary of this geometry).
        """
        ...


class GeometricOperations:
    def union(
        self, other: SphericalPoint | MultiSphericalPoint
    ) -> MultiSphericalPoint | None:
        """
        union of points from this geometry and the other geometry

        For further explanation of Union see Shapely's `object.union`.
        """
        ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float:
        """shortest great-circle distance over the sphere from any part of this geometry to another"""
        ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> (
        SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon
        | None
    ):
        """
        any part of this geometry that is within another

        NOTE: this function is NOT rigorous;
        it will ONLY return the lower order of geometry being compared
        and will NOT handle touching, colinear overlap, or degenerate cases
        """
        ...

    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> (
        SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon
    ):
        """
        points in this object not in the other geometric object, and the points in the other not in this geometric object.

        Splits this geometry into a multi-geometry, at the crossing with the other geometry.

        For further explanation of Symmetric Difference see Shapely's `object.symmetric_difference`.
        """
        ...


class SphericalPoint(Geometry, GeometricRelationships, GeometricOperations):
    """
    single point on the sphere, represented internally as a 3-dimensional Cartesian point (X, Y, Z) with origin at the center of the unit sphere

    Create a `SphericalPoint` from angular coordinates (longitude, latitude)::

        from sphersgeo import SphericalPoint

        a = SphericalPoint((60.0, 30.0))
        b = SphericalPoint((60.0, 0.0))
        c = SphericalPoint((-30.0, -30.0))

    \... or Cartesian coordinates (X, Y, Z)::

        from sphersgeo import SphericalPoint

        a = SphericalPoint((0.43301270189221946, 0.75, 0.5))
        b = SphericalPoint((0.5, 0.8660254037844386, 0.0))
        c = SphericalPoint((0.75, -0.4330127018922193, -0.5))
        d = SphericalPoint((0.0, 0.0, 1.0))
        e = SphericalPoint((0.0, 0.0, -1.0))
    """

    def __init__(
        self,
        point: tuple[float, float, float]
        | np.ndarray[tuple[np.typing.Literal[3]], np.dtype[np.float64]]
        | tuple[float, float]
        | list[float],
    ): ...

    @property
    def xyz(self) -> tuple[float, float, float]:
        """coordinates of this point as X, Y, and Z from the center of the sphere"""
        ...

    @property
    def lonlat(self) -> tuple[float, float]:
        """coordinates of this point as longitude and latitude"""
        ...

    @property
    def antipode(self) -> SphericalPoint:
        """antipodal point on the opposite side of the sphere"""
        ...

    def two_arc_angle(self, a: SphericalPoint, c: SphericalPoint) -> float:
        """
        given three points on the sphere:

          - `a`
          - `b` (this point)
          - `c`

        retrieves the turning angle, in radians, at `b` formed by arcs `ab` and `bc`
        """
        ...

    def colinear(self, a: SphericalPoint, b: SphericalPoint) -> bool:
        """whether this point lies on an arc between two other points"""
        ...

    def is_clockwise_turn(self, a: SphericalPoint, b: SphericalPoint) -> bool:
        """whether the angle formed between this point and two other points is a clockwise turn"""
        ...

    def interpolate_points(self, end: SphericalPoint, n: int) -> MultiSphericalPoint:
        """create n number of equally-spaced points on the arc between this point and another point"""
        ...

    @property
    def vector_length(self) -> float:
        """length of the underlying xyz vector"""
        ...

    def vector_cross(self, other: SphericalPoint) -> SphericalPoint:
        """cross product of this xyz vector with another xyz vector"""
        ...

    def vector_dot(self, other: SphericalPoint) -> float:
        """dot product of this xyz vector with another xyz vector"""
        ...

    def vector_rotate_around(
        self, other: SphericalPoint, theta: float
    ) -> SphericalPoint:
        """rotate this xyz vector by theta angle around another xyz vector"""
        ...

    def to(self, other: SphericalPoint) -> ArcString:
        """arc to another point"""
        ...

    @property
    def boundary(self) -> None: ...

    def __add__(self, other: SphericalPoint) -> SphericalPoint: ...

    def __sub__(self, other: SphericalPoint) -> SphericalPoint: ...

    def __mul__(self, other: SphericalPoint) -> SphericalPoint: ...

    def __div__(self, other: SphericalPoint) -> SphericalPoint: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class MultiSphericalPoint(Geometry, GeometricRelationships, GeometricOperations):
    """
    collection of multiple points on the sphere

    Create a `MultiSphericalPoint` from a list of `SphericalPoint` s::

        from sphersgeo import SphericalPoint, MultiSphericalPoint

        a = SphericalPoint((60.0, 30.0))
        b = SphericalPoint((60.0, 0.0))
        c = SphericalPoint((0.75, -0.4330127018922193, -0.5))

        abc = MultiSphericalPoint([a, b, c])

    \... or from the inputs required to make a list of `SphericalPoint` s::

        from sphersgeo import MultiSphericalPoint

        abc = MultiSphericalPoint(
            [(60.0, 30.0), (60.0, 0.0), (0.75, -0.4330127018922193, -0.5)]
        )

    \... or from a `numpy.ndarray` of shape Nx2 (longitude, latitude) or Nx3 (X, Y, Z)::

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

    def __init__(
        self,
        points: list[tuple[float, float, float]]
        | list[SphericalPoint]
        | tuple[float, float]
        | np.ndarray[tuple[Any, np.typing.Literal[3]], np.dtype[np.float64]],
    ): ...

    @property
    def xyzs(
        self,
    ) -> np.ndarray[tuple[Any, np.typing.Literal[3]], np.dtype[np.float64]]:
        """coordinates of these points as X, Y, and Z (Nx3 `numpy.ndarray`)"""
        ...

    @property
    def lonlats(
        self,
    ) -> np.ndarray[tuple[Any, np.typing.Literal[2]], np.dtype[np.float64]]:
        """coordinates of these points as longitude and latitude (Nx2 `numpy.ndarray`)"""
        ...

    def nearest(self, other: SphericalPoint) -> tuple[SphericalPoint, float]:
        """retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point across the unit sphere"""
        ...

    @property
    def vectors_lengths(self) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
        """lengths of the underlying (X, Y, Z) vectors"""
        ...

    @property
    def parts(
        self,
    ) -> list[SphericalPoint]: ...

    def __len__(self) -> int:
        """number of geometries in this collection"""
        ...

    def __getitem__(self, index: int) -> SphericalPoint: ...

    def append(self, other: SphericalPoint):
        """append the geometry to this collection"""
        ...

    def extend(self, other: MultiSphericalPoint):
        """extend this collection with geometries from the other collection"""
        ...

    @property
    def boundary(self) -> None: ...

    def __iadd__(self, other: MultiSphericalPoint): ...

    def __add__(self, other: MultiSphericalPoint) -> MultiSphericalPoint: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class ArcString(Geometry, GeometricRelationships, GeometricOperations):
    """
    series of great circle arcs across the sphere, which can be open at the end or closed (returns to the initial point)

    A great circle arc is the shortest geodesic distance over the surface of the sphere between any two points.
    Arcstrings are comprised of an **ordered** collection of spherical points.
    Arcstrings can also be **closed**, in which case the final point is considered as connected back to the first point.

    Create an `ArcString` from a `MultiSphericalPoint`::

        from sphersgeo import ArcString, MultiSphericalPoint

        abc = ArcString(
            MultiSphericalPoint([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)]), closed=True
        )

    \... or from the inputs required to make a `MultiSphericalPoint`::

        from sphersgeo import ArcString

        de = ArcString(
            [
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ]
        )

    """

    def __init__(
        self,
        points: MultiSphericalPoint,
        closed: bool = False,
    ): ...

    def __len__(self) -> int:
        """number of arcs in this arcstring"""
        ...

    @property
    def lengths(self) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
        """angle subtended on the sphere by each arc"""
        ...

    @property
    def midpoints(self) -> MultiSphericalPoint:
        """midpoints of each arc"""
        ...

    @property
    def arcs(self) -> list[ArcString]:
        """decomposes this arcstring into individual arcs of 2 points each"""
        ...

    @property
    def closed(self) -> bool:
        """whether this arcstring is "closed" (the last vertex is considered connected to the first)"""
        ...

    @closed.setter
    def closed(self, closed: bool):
        """ "close" this arcstring (consider the last vertex as connected to the first)"""
        ...

    @property
    def crosses_self(self) -> bool:
        """whether this arcstring crosses itself"""
        ...

    @property
    def crossings_with_self(self) -> MultiSphericalPoint:
        """points at which this arcstring crosses itself"""
        ...

    def adjoins(self, other: ArcString) -> bool:
        """whether this arcstring shares endpoints with another (ignoring closed arcstrings which have no endpoints)"""
        ...

    def join(self, other: ArcString) -> ArcString | None:
        """join this arcstring to another"""
        ...

    def simplify(self):
        """remove redundant vertices that already lie along an arc in this arcstring"""
        ...

    @property
    def boundary(self) -> MultiSphericalPoint | None: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class MultiArcString(Geometry, GeometricRelationships, GeometricOperations):
    """
    collection of multiple series of great circle arcs (arcstrings)

    Create a `MultiArcString` from a list of `ArcString` s::

        from sphersgeo import ArcString, MultiArcString

        abc = ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)], closed=True)
        de = ArcString([(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
        abc_de = MultiArcString([abc, de])

    """

    def __init__(
        self,
        arcstrings: list[ArcString],
    ): ...

    @property
    def parts(
        self,
    ) -> list[ArcString]: ...

    def __len__(self) -> int:
        """number of geometries in this collection"""
        ...

    def __getitem__(self, index: int) -> ArcString: ...

    def append(self, other: ArcString):
        """append the geometry to this collection"""
        ...

    def extend(self, other: MultiArcString):
        """extend this collection with geometries from the other collection"""
        ...

    @property
    def boundary(self) -> MultiSphericalPoint | None: ...

    def __iadd__(self, other: MultiArcString): ...

    def __add__(self, other: MultiArcString) -> MultiArcString: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class SphericalPolygon(Geometry, GeometricRelationships, GeometricOperations):
    """
    polygon on the sphere, represented by a **counterclockwise** `ArcString` (assumed to be closed) to form the boundary, and a `SphericalPoint` guaranteed to be inside the polygon (inferred if not provided)

    .. attention::
        The inside of a `SphericalPolygon` is **always** assumed to be the area **to the left of the boundary**.
        Thus, passing a clockwise boundary will make the "inside" of the polygon
        the **entire surface area of the sphere sans the area enclosed by the boundary**.

    .. attention:: `SphericalPolygon` s in `sphersgeo` do NOT have holes.

    Create a `SphericalPolygon` from an `ArcString`::

        from sphersgeo import ArcString, SphericalPolygon

        abc = SphericalPolygon(ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)]))

    \... or from the inputs required to make an `ArcString`::

        from sphersgeo import SphericalPolygon

        abc = SphericalPolygon(
            [
                (0.43301270189221946, 0.75, 0.5),
                (0.5, 0.8660254037844386, 0.0),
                (0.75, -0.4330127018922193, -0.5),
            ]
        )

    """

    def __init__(
        self,
        polygon: ArcString | tuple[ArcString, SphericalPoint],
    ): ...

    @classmethod
    def from_cone(
        self,
        center: SphericalPoint,
        radius: float,
        steps: int = 16,
    ) -> SphericalPolygon: ...

    @property
    def convex() -> bool:
        """whether this polygon is convex, that is, all possible arcs between points inside the polygon can never leave the enclosed space"""
        ...

    @property
    def inverse(self) -> SphericalPolygon: ...

    def simplify(self):
        """remove redundant vertices that already lie along an arc in the boundary"""
        ...

    @property
    def boundary(self) -> ArcString: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class MultiSphericalPolygon(Geometry, GeometricRelationships, GeometricOperations):
    """
    collection of multiple polygons on the sphere

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

    \... or from the inputs required to make a list of `SphericalPolygon` s::

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

    def __init__(self, polygons: list[SphericalPolygon]): ...

    @property
    def boundary(self) -> MultiArcString: ...

    @property
    def parts(
        self,
    ) -> list[SphericalPolygon]: ...

    def __len__(self) -> int:
        """number of geometries in this collection"""
        ...

    def __getitem__(self, index: int) -> SphericalPolygon: ...

    def append(self, other: SphericalPolygon):
        """append the geometry to this collection"""
        ...

    def extend(self, other: MultiSphericalPolygon):
        """extend this collection with geometries from the other collection"""
        ...

    def __iadd__(self, other: MultiSphericalPolygon): ...

    def __add__(self, other: MultiSphericalPolygon) -> MultiSphericalPolygon: ...

    def __eq__(self, other) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...
