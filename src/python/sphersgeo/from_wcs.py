import astropy.wcs
import gwcs
import numpy as np

from sphersgeo import SphericalPolygon

__all__ = ["polygon_from_wcs"]


def polygon_from_wcs(
    wcs: gwcs.WCS | astropy.wcs.WCS, edges_per_side: int = 1
) -> SphericalPolygon:
    """
    Create a `SphericalPolygon` from the footprint of a world coordinate system.

    If the number of edges per side is set to 1, the polygon will be rectangular.
    Otherwise, the polygon will capture WCS distortion along the edges of the footprint.

    This method requires `astropy <http://astropy.org>`__ installed.

    Parameters
    ----------
    wcs: gwcs.WCS | astropy.wcs.WCS :
        WCS object
    edges_per_side: int :
        number of edges to create along each side of the polygon (Default value = 1)

    Returns
    -------
    polygon representing the footprint of the provided WCS
    """

    if not isinstance(wcs, gwcs.WCS):
        wcs = astropy.wcs.WCS(wcs)

    array_shape = (
        wcs.array_shape
        if hasattr(wcs, "array_shape") and wcs.array_shape is not None
        else wcs.pixel_shape[::-1]
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None
        else tuple(
            wcs.bounding_box[index][1] - wcs.bounding_box[index][0]
            for index in range(len(wcs.bounding_box))
        )
    )
    # if (
    #     edges_per_side <= 1
    #     and hasattr(wcs, "bounding_box")
    #     and wcs.bounding_box is not None
    # ):
    #     lonlats = wcs.footprint(center=False).T
    #     center = np.mean(lonlats, axis=0)
    # else:
    vertices_per_side = edges_per_side + 1

    # constrain number of vertices to the maximum number of pixels on an edge
    if vertices_per_side > max(array_shape):
        vertices_per_side = max(array_shape)

    # build a list of pixel indices that represent equally-spaced edge vertices
    origin_indices = np.zeros(vertices_per_side) - 0.5
    x_end_indices = array_shape[0] - origin_indices
    y_end_indices = array_shape[1] - origin_indices
    vertices_x = np.linspace(0, array_shape[0], num=vertices_per_side, endpoint=False)
    vertices_y = np.linspace(0, array_shape[1], num=vertices_per_side, endpoint=False)
    vertex_indices = np.concatenate(
        [
            # north edge
            np.stack([origin_indices, vertices_y], axis=1),
            # east edge
            np.stack([vertices_x, y_end_indices], axis=1),
            # south edge
            np.stack([x_end_indices, y_end_indices - vertices_y], axis=1),
            # west edge
            np.stack([x_end_indices - vertices_x, origin_indices], axis=1),
        ],
        axis=0,
    )

    # ensure bounding box is None
    if hasattr(wcs, "bounding_box"):
        wcs.bounding_box = None

    # query the WCS for pixel indices at the edges
    vertex_skycoords = wcs.pixel_to_world(*vertex_indices.T)
    lonlats = np.stack(
        [vertex_skycoords.ra.degree, vertex_skycoords.dec.degree], axis=1
    )
    center_skycoord = wcs.pixel_to_world(
        *(origin_indices + (origin_indices + array_shape) / 2)
    )
    center = center_skycoord.ra.degree, center_skycoord.dec.degree

    return SphericalPolygon((lonlats, center))
