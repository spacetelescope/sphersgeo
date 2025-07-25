import warnings
from os import PathLike

from sphersgeo import MultiSphericalPoint, SphericalPoint, SphericalPolygon

try:
    from gwcs import WCS
    import numpy as np
except (ImportError, ModuleNotFoundError):
    warnings.warn("using `from_wcs` requires `gwcs` to be installed")


def polygon_from_wcs(
    wcs: WCS | PathLike, extra_vertices_per_edge: int = 0
) -> SphericalPolygon:
    r"""
    Create a new ``SphericalPolygon`` from the footprint of a WCS specification.

    This method requires ``gwcs`` be installed.

    :param wcs: GWCS WCS object
    :param extra_vertices_per_edge: extra vertices to create on each edge to capture distortion
    :returns: polygon representing the footprint of the provided WCS
    """

    if not hasattr(wcs, "bounding_box") or wcs.bounding_box is None:
        raise ValueError(
            "Cannot infer image footprint from WCS without a bounding box."
        )

    image_shape = wcs.array_shape
    if extra_vertices_per_edge <= 0:
        vertex_points = wcs.footprint(center=False)
    else:
        # constrain number of vertices to the maximum number of pixels on an edge, excluding the corners
        if extra_vertices_per_edge > max(image_shape) - 2:
            extra_vertices_per_edge = max(image_shape) - 2

        # build a list of pixel indices that represent equally-spaced edge vertices
        edge_xs = np.linspace(
            0,
            image_shape[0],
            num=extra_vertices_per_edge + 1,
            endpoint=False,
        )
        edge_ys = np.linspace(
            0,
            image_shape[1],
            num=extra_vertices_per_edge + 1,
            endpoint=False,
        )
        edge_indices = np.round(
            np.concatenate(
                [
                    # north edge
                    np.stack(
                        [
                            np.repeat(0, repeats=extra_vertices_per_edge + 1),
                            edge_ys,
                        ],
                        axis=1,
                    ),
                    # east edge
                    np.stack(
                        [
                            edge_xs,
                            np.repeat(
                                image_shape[1] - 1, repeats=extra_vertices_per_edge + 1
                            ),
                        ],
                        axis=1,
                    ),
                    # south edge
                    np.stack(
                        [
                            np.repeat(
                                image_shape[0] - 1, repeats=extra_vertices_per_edge + 1
                            ),
                            edge_ys[::-1],
                        ],
                        axis=1,
                    ),
                    # west edge
                    np.stack(
                        [
                            edge_xs[::-1],
                            np.repeat(0, repeats=extra_vertices_per_edge + 1),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        )

        # query the WCS for pixel indices at the edges
        vertex_points = np.stack(
            wcs(*edge_indices.T, with_bounding_box=False),
            axis=1,
        )

    if hasattr(wcs, "array_shape") and wcs.array_shape is not None:
        interior_point = SphericalPoint.from_lonlat(
            wcs(wcs.array_shape[0] / 2.0, wcs.array_shape[1] / 2.0)
        )
    else:
        interior_point = None

    return SphericalPolygon(
        MultiSphericalPoint.from_lonlat(vertex_points),
        interior_point=interior_point,
    )
