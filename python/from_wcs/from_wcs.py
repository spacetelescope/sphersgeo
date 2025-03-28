from sphersgeo import SphericalPolygon, SphericalPoint, MultiSphericalPoint
import numpy as np
from os import PathLike
import warnings

try:
    import astropy.wcs
    import astropy.io.fits
except (ImportError, ModuleNotFoundError):
    warnings.warn("the `from_wcs` module requires `astropy` installed")


def polygon_from_wcs(
    wcs: astropy.wcs.WCS | astropy.io.fits.Header | PathLike,
    steps: int = 1,
    crval: tuple[float, float] | None = None,
) -> SphericalPolygon:
    r"""
    Create a new `SphericalPolygon` from the footprint of a FITS WCS specification.

    This method requires having `astropy <http://astropy.org>`__ installed.

    :param wcs: FITS header containing a WCS specification, or a path to a FITS file
    :param steps: The number of steps along each edge to convert into polygon edges.
    :returns: polygon representing the footprint of the provided FITS WCS
    """

    if not isinstance(wcs, astropy.wcs.WCS):
        wcs = astropy.wcs.WCS(wcs)

    if crval is not None:
        wcs.wcs.crval = crval

    try:
        xa, ya = wcs.pixel_shape
    except AttributeError:
        xa, ya = (wcs._naxis1, wcs._naxis2)

    length = steps * 4 + 1
    X = np.empty(length)
    Y = np.empty(length)

    # Now define each of the 4 edges of the quadrilateral
    X[0:steps] = np.linspace(1, xa, steps, False)
    Y[0:steps] = 1
    X[steps : steps * 2] = xa
    Y[steps : steps * 2] = np.linspace(1, ya, steps, False)
    X[steps * 2 : steps * 3] = np.linspace(xa, 1, steps, False)
    Y[steps * 2 : steps * 3] = ya
    X[steps * 3 : steps * 4] = 1
    Y[steps * 3 : steps * 4] = np.linspace(ya, 1, steps, False)
    X[-1] = 1
    Y[-1] = 1

    # Use wcslib to retrieve lonlats from pixels
    points = MultiSphericalPoint.from_lonlats(
        np.stack(wcs.all_pix2world(X, Y, 1), axis=1), degrees=True
    )

    # Calculate an inside point
    center = SphericalPoint.from_lonlat(
        wcs.all_pix2world(xa / 2.0, ya / 2.0, 1), degrees=True
    )

    return SphericalPolygon(points, center, holes=None)
