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
> `sphersgeo` is still in development
> and does not currently implement all the robust functionality provided by
> planar geometry packages such as [`geo`](https://docs.rs/geo/) or [Shapely](https://shapely.readthedocs.io/).

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.

```shell
pip install sphersgeo
```

Planar geometry packages typically classify geometries into points, linestrings (also called polylines), and polygons
(along with multi-geometry collections: multi-points, multi-linestrings / multi-polylines, and multi-polygons).
The spherical geometry analogues to these are spherical points, arcstrings, and spherical polygons.

| Planar     | Spherical          | Planar Collection | Spherical Collection    |
| ---------- | ------------------ | ----------------- | ----------------------- |
| Point      | `SphericalPoint`   | MultiPoint        | `MultiSphericalPoint`   |
| LineString | `ArcString`        | MultiLineString   | `MultiArcString`        |
| Polygon    | `SphericalPolygon` | MultiPolygon      | `MultiSphericalPolygon` |

### Sources and Further Reading

- Toddhunter, I. (1886). Article 99. In Spherical Trigonometry: For the Use of Colleges and Schools (pp. 73–74). [print](https://www.gutenberg.org/files/19770/19770-pdf.pdf).
- Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. p132. 1994. Academic Press. doi:10.5555/180895.180907. [print](https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover).
- Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
- M.A, Jayaram & Fleyeh, Hasan. (2016). Convex Hulls in Image Processing: A Scoping Review. American Journal of Intelligent Systems. 2016. 48-58. 10.5923/j.ajis.20160602.03. [pdf](https://www.researchgate.net/profile/Jayaram-Ma-2/publication/303522254).
- Klain, D. A. (2019). A probabilistic proof of the spherical excess formula (No. arXiv:1909.04505). arXiv. https://doi.org/10.48550/arXiv.1909.04505
- [`spherical_geometry`](https://github.com/spacetelescope/spherical_geometry)
- [`s2geometry`](https://github.com/google/s2geometry)
