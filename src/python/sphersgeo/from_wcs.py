import numpy as np

from sphersgeo import SphericalPolygon

__all__ = ["polygon_from_wcs"]


def polygon_from_wcs(wcs, steps: int | None = None) -> SphericalPolygon:
    """
    Infer the image footprint of a world coordinate system (WCS) from `wcs.array_shape` (and its intersection with `wcs.bounding_box`, if available).

    Requires `astropy` to be installed.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS` | `astropy.io.fits.Header` | `gwcs.WCS` | str :
        any WCS object that implements the common WCS API
    steps: int :
        The number of edges along each side of the footprint.
        More than 1 step captures WCS distortion along the edges.
        (Default value = 1)

    Returns
    -------
    `sphersgeo.SphericalPolygon`
    """

    if steps is None:
        steps = 1

    import astropy

    if isinstance(wcs, astropy.io.fits.Header | str):
        wcs = astropy.wcs.WCS(wcs)

    if (bbox_attr := getattr(wcs, "bounding_box", None)) is not None:
        if isinstance(bbox_attr, astropy.modeling.bounding_box.ModelBoundingBox):
            bbox = bbox_attr.bounding_box(order="F")
        else:
            bbox = tuple(bbox_attr)

        xmin, xmax = bbox[0]
        ymin, ymax = bbox[1]
        width = xmax - xmin
        height = ymax - ymin

        if (shape := getattr(wcs, "array_shape", None)) is not None:
            # clip (intersect) the bounding box with the array shape
            # if it is available, to avoid having edge vertices outside
            # of the image footprint.
            xmin = max(xmin, -0.5)
            ymin = max(ymin, -0.5)
            xmax = min(xmax, shape[1] - 0.5)
            ymax = min(ymax, shape[0] - 0.5)

    elif (shape := getattr(wcs, "array_shape", None)) is not None:
        height, width = shape
        xmin = ymin = -0.5
        xmax = width - 0.5
        ymax = height - 0.5

    else:
        raise ValueError(
            "Unable to infer footprint from WCS: the WCS object must have "
            "either a 'bounding_box' or an 'array_shape' property set."
        )

    # constrain number of vertices to the maximum number
    # of pixels on an edge:
    steps = max(1, min(int(steps), int(width), int(height)))

    # build a list of pixel indices that represent
    # equally-spaced edge vertices:
    x0 = np.repeat(xmin, steps)
    y0 = np.repeat(ymin, steps)
    x1 = np.repeat(xmax, steps)
    y1 = np.repeat(ymax, steps)
    x = np.linspace(xmin, xmax, num=steps, endpoint=False)
    y = np.linspace(ymin, ymax, num=steps, endpoint=False)

    # define each of the 4 edges of the quadrilateral
    vertices = np.concatenate(
        [
            # south edge
            np.stack([x, y0], axis=1),
            # east edge
            np.stack([x1, y], axis=1),
            # north edge
            np.stack([x1 - x + x0, y1], axis=1),
            # west edge
            np.stack([x0, y1 - y + y0], axis=1),
        ],
        axis=0,
    )

    # ensure bounding box is None because, due to rounding errors,
    # edge vertices may be outside of the bounding box, which would
    # result in `NaN` values when converting to sky coordinates
    if hasattr(wcs, "bounding_box"):
        wcs.bounding_box = None

    # convert the pixel indices into sky coordinates using the WCS
    try:
        vertex_skycoords = wcs.pixel_to_world(*vertices.T)
        xc = xmin + width / 2
        yc = ymin + height / 2
        center_skycoord = wcs.pixel_to_world(xc, yc)
        center = center_skycoord.ra.degree, center_skycoord.dec.degree
    finally:
        # restore the original bounding box if it was set, since we do not
        # want to have side effects on the WCS object passed in by the user
        if bbox_attr is not None:
            wcs.bounding_box = bbox_attr

    # pass the sky coordinates to a new polygon object as degrees
    return SphericalPolygon(
        (
            np.stack([vertex_skycoords.ra.degree, vertex_skycoords.dec.degree], axis=1),
            center,
        )
    )
