from enum import Enum
from typing import List, Tuple

from numpy import float64
from numpy.typing import NDArray


class SphericalPoint:
    def __init__(
        self,
        xyz: tuple[float, float, float] | NDArray[float64] | list[float],
    ): ...

    @classmethod
    def from_lonlat(
        cls,
        lonlat: tuple[float, float] | NDArray[float64] | list[float],
    ) -> SphericalPoint: ...

    @property
    def xyz(self) -> Tuple[float, float, float]: ...

    def to_lonlat(self) -> Tuple[float, float]: ...

    def two_arc_angle(self, a: SphericalPoint, b: SphericalPoint) -> float: ...

    def collinear(self, a: SphericalPoint, b: SphericalPoint) -> bool: ...

    def interpolate_between(
        self, other: SphericalPoint, n: int
    ) -> MultiSphericalPoint: ...

    @property
    def vector_length(self) -> float: ...

    def vector_rotate_around(
        self, other: SphericalPoint, theta: float
    ) -> SphericalPoint: ...

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

    def split(
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

    def __eq__(self, other: SphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class MultiSphericalPoint:
    def __init__(
        self,
        xyzs: list[tuple[float, float, float]] | NDArray[float] | list[list[float]],
    ): ...

    @classmethod
    def from_lonlats(
        cls, lonlats: tuple[float, float] | NDArray[float64] | list[list[float]]
    ) -> MultiSphericalPoint: ...

    @property
    def xyzs(self) -> NDArray[float64]: ...

    def to_lonlats(self) -> NDArray[float64]: ...

    def vectors_rotate_around(
        self, other: MultiSphericalPoint, theta: float
    ) -> MultiSphericalPoint: ...

    @property
    def vectors_lengths(self) -> NDArray[float64]: ...

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

    def split(
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

    def __concat__(self, other: MultiSphericalPoint) -> MultiSphericalPoint: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> SphericalPoint: ...

    def append(self, other: SphericalPoint): ...

    def extend(self, other: MultiSphericalPoint): ...

    def __add__(self, other: MultiSphericalPoint) -> MultiSphericalPoint: ...

    def __iadd__(self, other: MultiSphericalPoint): ...

    def __eq__(self, other: MultiSphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class ArcString:
    def __init__(
        self,
        points: MultiSphericalPoint
        | list[tuple[float, float, float]]
        | list[list[float]]
        | NDArray[float],
        closed: bool = False,
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

    def adjoins(self, other: ArcString) -> bool: ...

    def join(self, other: ArcString) -> ArcString | None: ...

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

    def split(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...

    def __eq__(self, other: SphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __len__(self) -> int: ...


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
    def representative(self) -> SphericalPoint: ...

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

    def split(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...

    def __eq__(self, other: SphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __len__(self) -> int: ...


class SphericalPolygon:
    def __init__(
        self,
        exterior: ArcString
        | MultiSphericalPoint
        | list[tuple[float, float, float]]
        | NDArray[float]
        | list[list[float]],
        interior_point: None
        | SphericalPoint
        | tuple[float, float, float]
        | NDArray[float] = None,
    ): ...

    @classmethod
    def from_cone(
        self,
        center: SphericalPoint,
        radius: float,
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

    def split(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...

    def __eq__(self, other: SphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class MultiSphericalPolygon:
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

    def split(
        self,
        other: SphericalPoint
        | MultiSphericalPoint
        | ArcString
        | MultiArcString
        | SphericalPolygon
        | MultiSphericalPolygon,
    ) -> AnyGeometry: ...

    def __eq__(self, other: SphericalPoint) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __len__(self) -> int: ...


class AnyGeometry(Enum):
    SphericalPoint = SphericalPoint
    MultiSphericalPoint = MultiSphericalPoint
    ArcString = ArcString
    MultiArcString = MultiArcString
    SphericalPolygon = SphericalPolygon
    MultiSphericalPolygon = MultiSphericalPolygon
