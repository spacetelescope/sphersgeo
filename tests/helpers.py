import os
from pathlib import Path

import numpy as np

from sphersgeo import MultiSphericalPoint

__all__ = ["ROOT_DIR", "get_point_set", "resolve_imagename"]


ROOT_DIR = Path(__file__).parent / "data"


def get_point_set(density: int = 25):
    return MultiSphericalPoint.from_lonlats(
        [
            (j, i)
            for i in np.linspace(-85, 85, density, True)
            for j in np.linspace(-180, 180, int(np.cos(np.deg2rad(i)) * density))
        ],
        degrees=True,
    )


def resolve_imagename(root: os.PathLike, base_name: os.PathLike):
    """Resolve image name for tests."""

    image_name = Path(root) / base_name

    # Is it zipped?
    if not image_name.exists():
        image_name = image_name.with_suffix(".fits.gz")

    return image_name
