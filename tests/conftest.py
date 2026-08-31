from pathlib import Path

import pytest
import sphersgeo


@pytest.helpers.register
def read_geometry_wkt_txt(
    *filenames: Path,
) -> dict[
    str,
    (float, float, sphersgeo.AnyGeometry),
]:
    # TODO; figure out how to make this less dependent on specific CSV format
    lines = []
    for filename in filenames:
        with open(filename) as geometries_file:
            lines.extend(geometries_file.readlines()[1:])

    geometries = {}
    for line in lines:
        # this line needs to be changed if we add another field to the CSV
        name, area, length, wkt = line.split(",", 3)
        geometries[name] = (
            float(area),
            float(length),
            sphersgeo.from_wkt(wkt.strip().strip('"')),
        )
    return geometries
