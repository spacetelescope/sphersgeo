from enum import Enum
from typing import List, Tuple

from numpy import float64
from numpy.typing import NDArray


class SphericalPoint:
    """3D Cartesian vector (XYZ) representing a point on the sphere"""

    def __init__(
        self,
        point: tuple[float, float, float]
        | NDArray[float64]
        | tuple[float, float]
        | list[float],
    ): ...

    @property
    def xyz(self) -> Tuple[float, float, float]:
        """xyz vector as a 1-dimensional array of 3 floats"""
        ...

    @property
    def lonlat(self) -> Tuple[float, float]:
        """convert this point on the sphere to angular coordinates"""
        ...

    def two_arc_angle(self, a: SphericalPoint, b: SphericalPoint) -> float:
        """angle on the sphere between this point and two other points"""
        ...

    def collinear(self, a: SphericalPoint, b: SphericalPoint) -> bool:
        """whether this point shares a line with two other points"""
        ...

    def is_clockwise_turn(self, a: SphericalPoint, b: SphericalPoint) -> bool:
        """whether the angle formed between this point and two other points is a clockwise turn"""
        ...

    def interpolate_points(self, end: SphericalPoint, n: int) -> MultiSphericalPoint:
        """create n number of points equally spaced on an arc between this point and another point"""
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
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def boundary(self) -> None: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def convex_hull(self) -> None: ...
    @property
    def area(self) -> float: ...
    @property
    def length(self) -> float: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def disjoint(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def overlaps(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def covers(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def union(
        self, other: SphericalPoint | MultiSphericalPoint
    ) -> MultiSphericalPoint | None: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    def __add__(self, other: SphericalPoint) -> SphericalPoint: ...
    def __sub__(self, other: SphericalPoint) -> SphericalPoint: ...
    def __mul__(self, other: SphericalPoint) -> SphericalPoint: ...
    def __div__(self, other: SphericalPoint) -> SphericalPoint: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class MultiSphericalPoint:
    """collection of points on the sphere"""

    def __init__(
        self,
        points: list[tuple[float, float, float]]
        | list[SphericalPoint]
        | tuple[float, float]
        | NDArray[float64],
    ): ...

    @property
    def xyzs(self) -> NDArray[float64]:
        """xyz vectors as a 2-dimensional array of Nx3 floats"""
        ...

    @property
    def lonlats(self) -> NDArray[float64]:
        """convert to angle coordinates along the sphere"""
        ...

    def nearest(self, other: SphericalPoint) -> tuple[SphericalPoint, float]:
        """retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point across the unit sphere"""
        ...

    @property
    def vectors_lengths(self) -> NDArray[float64]:
        """
        lengths of the underlying xyz vectors
        """
        ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def boundary(self) -> None: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def convex_hull(self) -> SphericalPolygon:
        """
        Smallest convex polygon containing these points

        Implements Andrew's monotone chain algorithm.

        References
        ----------
        - https://www.researchgate.net/profile/Jayaram-Ma-2/publication/303522254/figure/fig1/AS:365886075621376@1464245446409/Monotone-Chain-Algorithm-and-graphic-illustration.png
        - https://github.com/google/s2geometry/blob/master/src/s2/s2convex_hull_query.cc#L123
        """
        ...

    @property
    def area(self) -> float: ...
    @property
    def length(self) -> float: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def disjoint(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def overlaps(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def covers(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def union(
        self, other: SphericalPoint | MultiSphericalPoint
    ) -> MultiSphericalPoint | None: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    @property
    def parts(self) -> List[SphericalPoint]: ...
    def __len__(self) -> int:
        """number of points in this collection"""
        ...

    def __getitem__(self, index: int) -> SphericalPoint: ...
    def append(self, other: SphericalPoint): ...
    def extend(self, other: MultiSphericalPoint): ...
    def __iadd__(self, other: MultiSphericalPoint): ...
    def __add__(self, other: MultiSphericalPoint) -> MultiSphericalPoint: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class ArcString:
    """series of great circle arcs along the sphere"""

    def __init__(
        self,
        points: MultiSphericalPoint,
        closed: bool = False,
    ): ...
    def __len__(self) -> int:
        """number of arcs in this arcstring"""
        ...

    @property
    def lengths(self) -> NDArray[float64]:
        """angle subtended on the sphere by each arc"""
        ...

    @property
    def midpoints(self) -> MultiSphericalPoint:
        """midpoints of each arc"""
        ...

    @property
    def arcs(self) -> list[ArcString]:
        """each individual arc in this arcstring"""
        ...

    @property
    def closed(self) -> bool:
        """whether this arcstring is "closed" (the last vertex is connected to the first)"""
        ...

    @closed.setter
    def closed(self, closed: bool):
        """ "close" this arcstring (connect the last vertex to the first)"""
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
        """whether this arcstring shares endpoints with another, ignoring closed arcstrings"""
        ...

    def join(self, other: ArcString) -> ArcString | None:
        """join this arcstring to another"""
        ...

    def simplify(self):
        """remove redundant vertices that already lie along the boundary"""
        ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def boundary(self) -> None: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def convex_hull(self) -> SphericalPolygon: ...
    @property
    def area(self) -> float: ...
    @property
    def length(self) -> float: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def disjoint(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def overlaps(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def covers(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def union(self, other: ArcString | MultiArcString) -> MultiArcString | None: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class MultiArcString:
    """collection of arcstrings"""

    def __init__(
        self,
        arcstrings: list[ArcString],
    ): ...
    @property
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def boundary(self) -> None: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def convex_hull(self) -> SphericalPolygon: ...
    @property
    def area(self) -> float: ...
    @property
    def length(self) -> float: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def disjoint(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def overlaps(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def covers(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def union(self, other: ArcString | MultiArcString) -> MultiArcString | None: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    @property
    def parts(self) -> List[ArcString]: ...
    def __len__(self) -> int:
        """number of arcstrings in this collection"""
        ...

    def __getitem__(self, index: int) -> ArcString: ...
    def append(self, other: ArcString): ...
    def extend(self, other: MultiArcString): ...
    def __iadd__(self, other: MultiArcString): ...
    def __add__(self, other: MultiArcString) -> MultiArcString: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class SphericalPolygon:
    """
    polygon on the sphere, comprising:
    1. a non-intersecting collection of connected arcs (arcstring) that connects back to its first point (closed)
    2. an interior point to specify which region of the sphere the polygon represents; this is required for non-Euclidian closed geometry
    """

    def __init__(
        self,
        polygon: ArcString | tuple[ArcString, SphericalPoint],
    ):
        """
        Providing an interior point is recommended because a sphere is a finite space and the boundary of a polygon divides it into two regions.
        If not provided, smaller of the two spaces be inferred as "inside" the polygon.
        """
        ...

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
    def antipode(self) -> bool: ...
    @property
    def inverse(self) -> SphericalPolygon: ...
    @property
    def is_clockwise(self) -> bool:
        """whether the points in this polygon are in clockwise order"""
        ...

    def simplify(self):
        """remove redundant vertices that already lie along the boundary"""
        ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def area(self) -> float:
        """
        surface area of this polygon

        deconstructs into triangles using method described at https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
        """
        ...

    @property
    def length(self) -> float: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def boundary(self) -> ArcString: ...
    @property
    def convex_hull(self) -> SphericalPolygon: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class MultiSphericalPolygon:
    """collection of polygons on the sphere"""

    def __init__(self, polygons: list[SphericalPolygon]): ...
    @property
    def vertices(self) -> MultiSphericalPoint: ...
    @property
    def area(self) -> float: ...
    @property
    def length(self) -> float: ...
    @property
    def representative(self) -> SphericalPoint: ...
    @property
    def centroid(self) -> SphericalPoint: ...
    @property
    def boundary(self) -> MultiArcString: ...
    @property
    def convex_hull(self) -> SphericalPolygon: ...
    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...
    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def touches(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def crosses(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def equals(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...
    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...
    def symmetric_difference(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...
    @property
    def parts(self) -> List[SphericalPolygon]: ...
    def __len__(self) -> int:
        """number of polygons in this collection"""
        ...

    def __getitem__(self, index: int) -> SphericalPolygon: ...
    def append(self, other: SphericalPolygon): ...
    def extend(self, other: MultiSphericalPolygon): ...
    def __iadd__(self, other: MultiSphericalPolygon): ...
    def __add__(self, other: MultiSphericalPolygon) -> MultiSphericalPolygon: ...
    def __eq__(self, other) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class AnyGeometry(Enum):
    SphericalPoint = SphericalPoint
    MultiSphericalPoint = MultiSphericalPoint
    ArcString = ArcString
    MultiArcString = MultiArcString
    SphericalPolygon = SphericalPolygon
    MultiSphericalPolygon = MultiSphericalPolygon
