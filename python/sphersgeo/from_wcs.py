import warnings
from os import PathLike

from sphersgeo import MultiSphericalPoint, SphericalPoint, SphericalPolygon

try:
    from gwcs import WCS
    import numpy as np
except (ImportError, ModuleNotFoundError):
    warnings.warn("using `from_wcs` requires `gwcs` to be installed")


__all__ = ["polygon_from_wcs"]


def polygon_from_wcs(
    wcs: WCS | PathLike, extra_vertices_per_edge: int = 0
) -> SphericalPolygon:
    """
    Create a new `SphericalPolygon` from the footprint of a WCS specification.

    Parameters
    ----------
    wcs: WCS :
        WCS object
    extra_vertices_per_edge: int :
        extra vertices to create on each edge to capture distortion (Default value = 0)

    Returns
    -------
    polygon representing the footprint of the provided WCS
    """

    if not hasattr(wcs, "bounding_box") or wcs.bounding_box is None:
        raise ValueError(
            "Cannot infer image footprint from WCS without a bounding box."
        )

    array_shape = (
        wcs.array_shape
        if hasattr(wcs, "array_shape") and wcs.array_shape is not None
        else tuple(
            wcs.bounding_box[index][1] - wcs.bounding_box[index][0]
            for index in range(len(wcs.bounding_box))
        )
    )
    if extra_vertices_per_edge <= 0:
        vertex_points = wcs.footprint(center=False)
    else:
        # constrain number of vertices to the maximum number of pixels on an edge, excluding the corners
        if extra_vertices_per_edge > max(array_shape) - 2:
            extra_vertices_per_edge = max(array_shape) - 2

        # build a list of pixel indices that represent equally-spaced edge vertices
        vertices_per_edge = 2 + extra_vertices_per_edge
        origin_indices = np.zeros(vertices_per_edge - 1) - 0.5
        x_end_indices = array_shape[0] - origin_indices
        y_end_indices = array_shape[1] - origin_indices
        vertices_x = np.linspace(
            0, array_shape[0], num=vertices_per_edge - 1, endpoint=False
        )
        vertices_y = np.linspace(
            0, array_shape[1], num=vertices_per_edge - 1, endpoint=False
        )
        edge_indices = np.concatenate(
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

        # query the WCS for pixel indices at the edges
        vertex_points = np.stack(
            wcs(*edge_indices.T, with_bounding_box=False),
            axis=1,
        )

    return SphericalPolygon(
        (
            MultiSphericalPoint.from_lonlats(vertex_points),
            SphericalPoint.from_lonlat(wcs(array_shape[0] / 2.0, array_shape[1] / 2.0)),
        ),
    )
