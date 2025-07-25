import warnings
from os import PathLike

from sphersgeo import MultiSphericalPoint, SphericalPoint, SphericalPolygon

try:
    from gwcs import WCS
except (ImportError, ModuleNotFoundError):
    warnings.warn("the `from_wcs` module requires `gwcs` installed")


def polygon_from_wcs(
    wcs: WCS | PathLike,
) -> SphericalPolygon:
    r"""
    Create a new ``SphericalPolygon`` from the footprint of a WCS specification.

    This method requires having ``gwcs`` installed.

    :param wcs: GWCS WCS object
    :returns: polygon representing the footprint of the provided WCS
    """

    if not hasattr(wcs, "bounding_box") or wcs.bounding_box is None:
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            # pixel_shape is in xy order, contrary to numpy convention
            wcs.bounding_box = (
                (-0.5, wcs.pixel_shape[1] - 0.5),
                (-0.5, wcs.pixel_shape[0] - 0.5),
            )
        else:
            raise ValueError(
                "cannot infer a bounding box from a GWCS object without `.pixel_shape`"
            )

    if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
        interior_point = SphericalPoint.from_lonlat(
            wcs.pixel_to_world(wcs.pixel_shape[1] / 2.0, wcs.pixel_shape[0] / 2.0)
        )
    else:
        interior_point = None

    return SphericalPolygon(
        MultiSphericalPoint.from_lonlat(wcs.footprint(center=False)),
        interior_point=interior_point,
    )
