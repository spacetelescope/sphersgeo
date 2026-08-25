from typing import Any

import numpy as np

import sphersgeo

from .sphersgeo import (
    ArcString,
    MultiArcString,
    MultiSphericalPoint,
    MultiSphericalPolygon,
    SphericalPoint,
    SphericalPolygon,
    from_wkt,
)

type AnyGeometry = (
    SphericalPoint
    | MultiSphericalPoint
    | ArcString
    | MultiArcString
    | SphericalPolygon
    | MultiSphericalPolygon
)

type AnyGeometryInputs = (
    SphericalPointInputs
    | MultiSphericalPointInputs
    | ArcStringInputs
    | MultiArcStringInputs
    | SphericalPolygonInputs
    | MultiSphericalPolygonInputs
)

type GeometryCollection = tuple[
    MultiSphericalPoint, MultiArcString, MultiSphericalPolygon
]

type SphericalPointInputs = (
    tuple[float, float]
    | np.ndarray[tuple[np.typing.Literal[2]], np.dtype[np.float64]]
    | tuple[float, float, float]
    | np.ndarray[tuple[np.typing.Literal[3]], np.dtype[np.float64]]
    | list[float]
    | str
    | SphericalPoint
)

type MultiSphericalPointInputs = (
    list[SphericalPointInputs]
    | np.ndarray[tuple[Any, np.typing.Literal[2]], np.dtype[np.float64]]
    | np.ndarray[tuple[Any, np.typing.Literal[3]], np.dtype[np.float64]]
    | str
    | MultiSphericalPoint
)

type ArcStringInputs = MultiSphericalPointInputs | ArcString

type MultiArcStringInputs = list[ArcStringInputs] | str | MultiArcString

type SphericalPolygonInputs = (
    ArcStringInputs
    | tuple[ArcStringInputs, SphericalPointInputs]
    | str
    | SphericalPolygon
)

type MultiSphericalPolygonInputs = (
    list[SphericalPolygonInputs] | str | MultiSphericalPolygon
)

__all__ = [
    "AnyGeometry",
    "AnyGeometryInputs",
    "ArcString",
    "ArcStringInputs",
    "MultiArcString",
    "MultiArcStringInputs",
    "MultiSphericalPoint",
    "MultiSphericalPointInputs",
    "MultiSphericalPolygon",
    "MultiSphericalPolygonInputs",
    "SphericalPoint",
    "SphericalPointInputs",
    "SphericalPolygon",
    "SphericalPolygonInputs",
    "from_wkt",
]

__doc__ = sphersgeo.__doc__
if hasattr(sphersgeo, "__all__"):
    __all__ = sphersgeo.__all__
