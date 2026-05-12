<a href="https://stsci.edu">
  <img src="docs/assets/stsci_logo.png" alt="STScI Logo" width="15%" style="margin-left: auto;"/>
  <img src="docs/assets/stsci_name.png" alt="STScI Name" width="68%"/>
</a>

# sphersgeo

[![build](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml)
[![tests](https://github.com/spacetelescope/sphersgeo/actions/workflows/test.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/test.yml)
[![Powered by STScI](https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat)](https://www.stsci.edu)

#### object-oriented spherical geometry

> [!IMPORTANT]
> `sphersgeo` is still in development and does NOT currently implement all of the functionality provided by other geo packages such as `geo` or Shapely.

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.

### Installation

```shell
pip install sphersgeo
```

### Usage

Euclidean geometry packages classify geometries into points, linestrings, and polygons (along with multi-variations: multipoints, multilinestrings, and multipolygons).
Spherical geometry analogues are spherical points, arcstrings, and spherical polygons.

| Euclidean       | Spherical             |
| --------------- | --------------------- |
| Point           | SphericalPoint        |
| MultiPoint      | MultiSphericalPoint   |
| LineString      | ArcString             |
| MultiLineString | MultiArcString        |
| Polygon         | SphericalPolygon      |
| MultiPolygon    | MultiSphericalPolygon |

See [`src/python/sphersgeo/sphersgeo.pyi`](src/python/sphersgeo/sphersgeo.pyi) for Python class definitions.

#### SphericalPoint

Spherical points are represented internally as 3-dimensional Euclidean vectors (X, Y, Z) a unit distance (length of 1.0) from the center of the sphere.

```python
from sphersgeo import SphericalPoint

# define a point on the sphere in angular coordinates (longitude and latitude)
a = SphericalPoint((60.0, 30.0))
b = SphericalPoint((60.0, 0.0))
c = SphericalPoint((-30.0, -30.0))

# ... or in Euclidean coordinates (X, Y, Z)
a = SphericalPoint((0.43301270189221946, 0.75, 0.5))
b = SphericalPoint((0.5, 0.8660254037844386, 0.0))
c = SphericalPoint((0.75, -0.4330127018922193, -0.5))
d = SphericalPoint((0.0, 0.0, 1.0))
e = SphericalPoint((0.0, 0.0, -1.0))

# collate multiple points together by passing lists of angular or Euclidean coordinates
from sphersgeo import MultiSphericalPoint

ab = MultiSphericalPoint([(60.0, 30.0), (60.0, 0.0)])
ab = MultiSphericalPoint(
    [(0.43301270189221946, 0.75, 0.5), (0.5, 0.8660254037844386, 0.0)]
)
de = MultiSphericalPoint(
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
)

# ... or a Numpy array of coordinates
import numpy as np

ab = MultiSphericalPoint(np.array([(60.0, 30.0), (60.0, 0.0)]))
ab = MultiSphericalPoint(
    np.array([(0.43301270189221946, 0.75, 0.5), (0.5, 0.8660254037844386, 0.0)])
)

# ... or a list of SphericalPoint objects
abcde = MultiSphericalPoint([a, b, c, d, e])
```

#### ArcString

A great circle arcs is the shortest geodesic distance over the surface of the sphere between any two points.
Arcstrings are comprised of an **ordered** collection of spherical points.
Arcstrings can also be **closed**, in which case the final point is considered as connected back to the first point.

```python
from sphersgeo import ArcString

# define an arcstring on the sphere by passing angular or Euclidean coordinates
abc = ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
de = ArcString(
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
)

# ... or a list of SphericalPoints
from sphersgeo import SphericalPoint

abc = ArcString(
    [
        SphericalPoint((60.0, 0.0)),
        SphericalPoint((60.0, 30.0)),
        SphericalPoint((-30.0, -30.0)),
    ],
    closed=True,
)

# ... or a single MultiSphericalPoint
from sphersgeo import MultiSphericalPoint

de = ArcString(
    MultiSphericalPoint(
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
    )
)

# collate arcstrings into a MultiArcString by passing the same inputs you would to the individual objects
from sphersgeo import MultiArcString

abc_de = MultiArcString(
    [
        [(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)],
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
    ]
)

# ... or a list of ArcString objects
abc_de = MultiArcString([abc, de])
```

#### SphericalPolygon

Spherical polygons are comprised of

1. closed arcstring that represents the outer boundary, and
2. a sample point that defines which side of the closed spherical region is "inside" the boundary.

If the "inside point" is not given, the **smaller** of the two regions split by the boundary will be assigned to be the "inside".

> [!NOTE]
> Polygons in `sphersgeo` do NOT have holes.

```python
from sphersgeo import SphericalPolygon

# define a spherical polygon by passing angular or Euclidean coordinates to form the boundary
abc = SphericalPolygon([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
abc = SphericalPolygon(
    [
        (0.43301270189221946, 0.75, 0.5),
        (0.5, 0.8660254037844386, 0.0),
        (0.75, -0.4330127018922193, -0.5),
    ]
)

# ... or pass an ArcString object (the arcstring is assumed to be closed)
from sphersgeo import ArcString

abc = SphericalPolygon(
    ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
)

# collate multiple polygons into a MultiSphericalPolygon by passing the same inputs you would to the individual objects
from sphersgeo import MultiSphericalPolygon

abc_def = MultiSphericalPolygon(
    [
        [(60.0, 30.0), (60.0, 0.0), (-30.0, -30.0)],
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
        ],
    ]
)

# ... or a list of SphericalPolygon objects
abc_def = MultiSphericalPolygon(
    [
        abc,
        SphericalPolygon(
            [
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (1.0, 1.0, 0.0),
            ]
        ),
    ]
)
```
