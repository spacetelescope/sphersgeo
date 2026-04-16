import sphersgeo

from .sphersgeo import (
    ArcString,
    MultiArcString,
    MultiSphericalPoint,
    MultiSphericalPolygon,
    SphericalPoint,
    SphericalPolygon,
)

__all__ = [
    "SphericalPoint",
    "MultiSphericalPoint",
    "ArcString",
    "MultiArcString",
    "SphericalPolygon",
    "MultiSphericalPolygon",
]

__doc__ = sphersgeo.__doc__
if hasattr(sphersgeo, "__all__"):
    __all__ = sphersgeo.__all__
