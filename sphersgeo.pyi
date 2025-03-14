from enum import Enum

from numpy import float64
from numpy.typing import NDArray


class SphericalPoint:
    def __init__(
        self,
        point: tuple[float, float, float] | list[float] | NDArray[float],
    ): ...

    @classmethod
    def normalize(
        cls, point: NDArray[float64] | tuple[float, float, float] | list[float]
    ) -> SphericalPoint: ...

    @classmethod
    def from_lonlat(
        cls,
        coordinates: NDArray[float64] | tuple[float, float] | list[float],
        degrees: bool = True,
    ) -> SphericalPoint: ...

    @property
    def xyz(self) -> NDArray[float64]: ...

    def to_lonlat(self, degrees: bool = True) -> NDArray[float64]: ...

    @property
    def normalized(self) -> SphericalPoint: ...

    def angle_between(
        self, a: SphericalPoint, b: SphericalPoint, degrees: bool = True
    ) -> float: ...

    def collinear(self, a: SphericalPoint, b: SphericalPoint) -> bool: ...

    @property
    def vector_length(self) -> float: ...

    def vector_rotate_around(
        self, other: SphericalPoint, theta: float, degrees: bool = True
    ) -> SphericalPoint: ...

    def combine(self, other: SphericalPoint) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...

    def __add__(
        self, other: SphericalPoint | NDArray[float64] | tuple[float, float, float]
    ) -> SphericalPoint: ...

    def __iadd__(
        self, other: SphericalPoint | NDArray[float64] | tuple[float, float, float]
    ): ...


class MultiSphericalPoint:
    def __init__(
        self,
        points: list[tuple[float, float, float]]
        | list[list[float]]
        | list[float]
        | NDArray[float],
    ): ...

    @classmethod
    def normalize(
        cls,
        points: NDArray[float64]
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float],
    ) -> MultiSphericalPoint: ...

    @classmethod
    def from_lonlats(
        cls,
        coordinates: NDArray[float64]
        | tuple[float, float]
        | list[list[float]]
        | list[float],
        degrees: bool = True,
    ) -> MultiSphericalPoint: ...

    @property
    def xyz(self) -> NDArray[float64]: ...

    def to_lonlats(self, degrees: bool = True) -> NDArray[float64]: ...

    @property
    def normalized(self) -> MultiSphericalPoint: ...

    def angles_between(
        self, a: MultiSphericalPoint, b: MultiSphericalPoint, degrees: bool = True
    ) -> NDArray[float64]: ...

    def collinear(
        self, a: MultiSphericalPoint, b: MultiSphericalPoint
    ) -> NDArray[bool]: ...

    @property
    def vector_lengths(self) -> NDArray[float64]: ...

    def vector_rotate_around(
        self, other: MultiSphericalPoint, theta: float, degrees: bool = True
    ) -> MultiSphericalPoint: ...

    def extend(self, other: MultiSphericalPoint): ...

    def append(self, other: SphericalPoint): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...

    def __add__(
        self, other: MultiSphericalPoint | NDArray[float64]
    ) -> MultiSphericalPoint: ...

    def __iadd__(self, other: MultiSphericalPoint | NDArray[float64]): ...


class ArcString:
    def __init__(
        self,
        points: MultiSphericalPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float]
        | NDArray[float],
    ): ...

    @classmethod
    def normalize(
        cls,
        points: NDArray[float64]
        | MultiSphericalPoint
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
        degrees: bool = True,
    ) -> ArcString: ...

    @property
    def lengths(self) -> NDArray[float64]: ...

    @property
    def midpoints(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...


class MultiArcString:
    def __init__(
        self,
        arcstrings: list[ArcString]
        | list[MultiSphericalPoint]
        | list[list[tuple[float, float, float]]]
        | list[list[list[float]]]
        | list[NDArray[float]],
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...


class AngularBounds:
    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        degrees: bool = True,
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...


class SphericalPolygon:
    def __init__(
        self,
        arcstring: ArcString
        | MultiSphericalPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float]
        | NDArray[float],
        interior: None
        | SphericalPoint
        | tuple[float, float, float]
        | list[float]
        | NDArray[float],
        holes: None
        | MultiArcString
        | list[MultiSphericalPoint]
        | list[list[tuple[float, float, float]]]
        | list[list[list[float]]]
        | list[NDArray[float]],
    ): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...


class MultiSphericalPolygon:
    def __init__(self, polygons: list[SphericalPolygon]): ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    def bounds(self, degrees: bool = True) -> AngularBounds: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    @property
    def points(self) -> MultiSphericalPoint: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> float: ...

    def contains(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def within(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersects(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> bool: ...

    def intersection(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | AngularBounds
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry | None: ...


class AnyGeometry(Enum):
    SphericalPoint = (SphericalPoint,)
    MultiSphericalPoint = (MultiSphericalPoint,)
    ArcString = (ArcString,)
    MultiArcString = (MultiArcString,)
    AngularBounds = (AngularBounds,)
    SphericalPolygon = (SphericalPolygon,)
    MultiSphericalPolygon = (MultiSphericalPolygon,)
