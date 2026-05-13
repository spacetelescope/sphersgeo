<a href="https://stsci.edu">
  <img src="docs/_static/stsci_symbol_pantone279.png" alt="STScI Logo" width="15%" style="margin-left: auto;"/>
  <img src="docs/_static/stsci_wordmark_grey.png" alt="STScI Name" width="68%"/>
</a>

# sphersgeo

[![build](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml)
[![tests](https://github.com/spacetelescope/sphersgeo/actions/workflows/tests.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/tests.yml)
[![readthedocs](https://readthedocs.org/projects/sphersgeo/badge/?version=latest)](https://sphersgeo.readthedocs.io)
[![Powered by STScI](https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat)](https://www.stsci.edu)

[`sphersgeo`](https://sphersgeo.readthedocs.io) is an object-oriented spherical geometry package written in Rust with Python accessor classes and methods.

> [!IMPORTANT]
> `sphersgeo` is still in development and does NOT currently implement all of the functionality provided by other geo packages such as `geo` or Shapely.

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.

```shell
pip install sphersgeo
```

Planar geometry packages typically classify geometries into points, linestrings, and polygons
(along with multi-geometry collections: multi-points, multi-linestrings, and multi-polygons).
The spherical geometry analogues to these are spherical points, arcstrings, and spherical polygons.

| Planar     | Spherical          | Planar Collection | Spherical Collection    |
| ---------- | ------------------ | ----------------- | ----------------------- |
| Point      | `SphericalPoint`   | MultiPoint        | `MultiSphericalPoint`   |
| LineString | `ArcString`        | MultiLineString   | `MultiArcString`        |
| Polygon    | `SphericalPolygon` | MultiPolygon      | `MultiSphericalPolygon` |

### Other Spherical Geometry Packages
- [`spherical_geometry`](https://github.com/spacetelescope/spherical_geometry)
- [`s2geometry`](https://github.com/google/s2geometry)
