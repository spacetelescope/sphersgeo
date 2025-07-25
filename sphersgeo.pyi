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

    @property
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

    @property
    def centroid(self) -> SphericalPoint: ...

    @property
    def boundary(self) -> None: ...

    @property
    def convex_hull(self) -> None: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
        degrees: bool = True,
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

    def __add__(self, other: SphericalPoint) -> MultiSphericalPoint: ...


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
    def from_lonlat(
        cls,
        coordinates: NDArray[float64]
        | tuple[float, float]
        | list[list[float]]
        | list[float],
        degrees: bool = True,
    ) -> MultiSphericalPoint: ...

    @property
    def xyz(self) -> NDArray[float64]: ...

    def to_lonlat(self, degrees: bool = True) -> NDArray[float64]: ...

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
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

    @property
    def centroid(self) -> SphericalPoint: ...

    @property
    def boundary(self) -> None: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
        degrees: bool = True,
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

    @property
    def closed(self) -> bool: ...

    @property
    def lengths(self) -> NDArray[float64]: ...

    @property
    def midpoints(self) -> MultiSphericalPoint: ...

    @property
    def crosses_self(self) -> bool: ...

    @property
    def crossings_with_self(self) -> MultiSphericalPoint: ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

    @property
    def centroid(self) -> SphericalPoint: ...

    @property
    def boundary(self) -> MultiSphericalPoint: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
        degrees: bool = True,
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
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

    @property
    def centroid(self) -> SphericalPoint: ...

    @property
    def boundary(self) -> MultiSphericalPoint: ...

    @property
    def convex_hull(self) -> SphericalPolygon | None: ...

    def distance(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
        degrees: bool = True,
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


class SphericalPolygon:
    def __init__(
        self,
        exterior: ArcString
        | MultiSphericalPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | list[float]
        | NDArray[float],
        interior_point: None
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

    @classmethod
    def from_cone(
        self,
        center: SphericalPoint,
        radius: float,
        degrees: bool = True,
        steps: int = 16,
    ) -> SphericalPolygon: ...

    @property
    def antipode(self) -> bool: ...

    @property
    def inverse(self) -> SphericalPolygon: ...

    @property
    def is_clockwise(self) -> bool: ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

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
        degrees: bool = True,
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


class MultiSphericalPolygon:
    def __init__(self, polygons: list[SphericalPolygon]): ...

    @property
    def vertices(self) -> MultiSphericalPoint: ...

    @property
    def area(self) -> float: ...

    @property
    def length(self) -> float: ...

    @property
    def representative_point(self) -> SphericalPoint: ...

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
        degrees: bool = True,
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


class AnyGeometry(Enum):
    SphericalPoint = SphericalPoint
    MultiSphericalPoint = MultiSphericalPoint
    ArcString = ArcString
    MultiArcString = MultiArcString
    SphericalPolygon = SphericalPolygon
    MultiSphericalPolygon = MultiSphericalPolygon
