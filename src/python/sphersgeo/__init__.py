from typing import Any, TypeAlias

import numpy as np

import sphersgeo

from .sphersgeo import (
    ArcString,
    GeometryCollection,
    MultiArcString,
    MultiSphericalPoint,
    MultiSphericalPolygon,
    SphericalPoint,
    SphericalPolygon,
    from_wkt,
)

AnyGeometry: TypeAlias = (  # noqa: UP040
    SphericalPoint
    | MultiSphericalPoint
    | ArcString
    | MultiArcString
    | SphericalPolygon
    | MultiSphericalPolygon
)

SphericalPointInputs: TypeAlias = (  # noqa: UP040
    tuple[float, float]
    | np.ndarray[tuple[2], np.dtype[np.float64]]
    | tuple[float, float, float]
    | np.ndarray[tuple[3], np.dtype[np.float64]]
    | list[float]
    | str
    | SphericalPoint
)

MultiSphericalPointInputs: TypeAlias = (  # noqa: UP040
    list[SphericalPointInputs]
    | np.ndarray[tuple[Any, 2], np.dtype[np.float64]]
    | np.ndarray[tuple[Any, 3], np.dtype[np.float64]]
    | str
    | MultiSphericalPoint
)

ArcStringInputs: TypeAlias = MultiSphericalPointInputs | ArcString  # noqa: UP040

MultiArcStringInputs: TypeAlias = list[ArcStringInputs] | str | MultiArcString  # noqa: UP040

SphericalPolygonInputs: TypeAlias = (  # noqa: UP040
    ArcStringInputs
    | tuple[ArcStringInputs, SphericalPointInputs]
    | str
    | SphericalPolygon
)

MultiSphericalPolygonInputs: TypeAlias = (  # noqa: UP040
    list[SphericalPolygonInputs] | str | MultiSphericalPolygon
)

AnyGeometryInputs: TypeAlias = (  # noqa: UP040
    SphericalPointInputs
    | MultiSphericalPointInputs
    | ArcStringInputs
    | MultiArcStringInputs
    | SphericalPolygonInputs
    | MultiSphericalPolygonInputs
)

MultiGeometry: TypeAlias = (  # noqa: UP040
    sphersgeo.MultiSphericalPoint
    | sphersgeo.MultiArcString
    | sphersgeo.MultiSphericalPolygon
)


__all__ = [
    "AnyGeometry",
    "AnyGeometryInputs",
    "ArcString",
    "ArcStringInputs",
    "GeometryCollection",
    "MultiArcString",
    "MultiArcStringInputs",
    "MultiGeometry",
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
