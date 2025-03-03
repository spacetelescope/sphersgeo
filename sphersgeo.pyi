from numpy import float64
from numpy.typing import NDArray


class VectorPoint:
    def __init__(
        self, point: NDArray[float64] | tuple[float, float, float] | list[float]
    ): ...

    @classmethod
    def normalize(
        cls, point: NDArray[float64] | tuple[float, float, float] | list[float]
    ) -> VectorPoint: ...

    @classmethod
    def from_lonlat(
        cls,
        coordinates: NDArray[float64] | tuple[float, float] | list[float],
        degrees: bool,
    ) -> VectorPoint: ...

    @property
    def xyz(self) -> NDArray[float64]: ...

    def to_lonlat(self, degrees: bool) -> NDArray[float64]: ...

    @property
    def normalized(self) -> VectorPoint: ...

    def angle(self, a: VectorPoint, b: VectorPoint, degrees: bool) -> float: ...

    def collinear(self, a: VectorPoint, b: VectorPoint) -> bool: ...

    @property
    def vector_length(self) -> float: ...

    def vector_rotate_around(
        self, other: VectorPoint, theta: float, degrees: bool
    ) -> VectorPoint: ...

    def combine(self, other: VectorPoint) -> MultiVectorPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...

    def __add__(self, other: VectorPoint) -> MultiVectorPoint: ...


class MultiVectorPoint:
    def __init__(
        self,
        points: NDArray[float64]
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float],
    ): ...

    @classmethod
    def normalize(
        cls,
        points: NDArray[float64]
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float],
    ) -> MultiVectorPoint: ...

    @classmethod
    def from_lonlats(
        cls,
        coordinates: NDArray[float64]
        | tuple[float, float]
        | list[list[float]]
        | list[float],
        degrees: bool,
    ) -> MultiVectorPoint: ...

    @property
    def xyz(self) -> NDArray[float64]: ...

    def to_lonlats(self, degrees: bool) -> NDArray[float64]: ...

    @property
    def normalized(self) -> MultiVectorPoint: ...

    def angles(
        self, a: MultiVectorPoint, b: MultiVectorPoint, degrees: bool
    ) -> NDArray[float64]: ...

    def collinear(self, a: MultiVectorPoint, b: MultiVectorPoint) -> NDArray[bool]: ...

    @property
    def vector_lengths(self) -> NDArray[float64]: ...

    def vector_rotate_around(
        self, other: MultiVectorPoint, theta: float, degrees: bool
    ) -> MultiVectorPoint: ...

    def extend(self, other: MultiVectorPoint): ...

    def append(self, other: VectorPoint): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...

    def __add__(self, other: MultiVectorPoint) -> MultiVectorPoint: ...

    def __iadd__(self, other: MultiVectorPoint): ...


class ArcString:
    def __init__(
        self,
        points: NDArray[float64]
        | MultiVectorPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float],
    ): ...

    @classmethod
    def normalize(
        cls,
        points: NDArray[float64]
        | MultiVectorPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float],
    ) -> ArcString: ...

    @classmethod
    def from_lonlats(
        cls,
        coordinates: NDArray[float64]
        | tuple[float, float]
        | list[list[float]]
        | list[float],
        degrees: bool,
    ) -> ArcString: ...

    @property
    def lengths(self) -> NDArray[float64]: ...

    @property
    def midpoints(self) -> MultiVectorPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...


class MultiArcString:
    def __init__(
        self,
        arcstrings: list[NDArray[float64]]
        | list[MultiVectorPoint]
        | list[list[tuple[float64, float64, float64]]]
        | list[list[list[float64]]],
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...


class AngularBounds:
    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        degrees: bool,
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...


class AngularPolygon:
    def __init__(
        self,
        arcstring: ArcString,
        interior: None | VectorPoint,
        holes: None | MultiArcString,
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...


class MultiAngularPolygon:
    def __init__(self, polygons: list[AngularPolygon]): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...

    def distance(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> float: ...

    def contains(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def within(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: VectorPoint
        | MultiVectorPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | AngularPolygon
        | MultiAngularPolygon,
    ) -> GeometryCollection: ...


class GeometryCollection:
    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool) -> AngularBounds: ...

    @property
    def convex_hull(self) -> AngularPolygon | None: ...

    @property
    def points(self) -> MultiVectorPoint: ...
