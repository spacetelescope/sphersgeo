# sphersgeo

#### object-oriented spherical geometry in Rust (and Python bindings)

> [!IMPORTANT]
> `sphersgeo` is still in development and does NOT currently implement all of the functionality provided by other geo packages such as `geo` or Shapely.

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.

### Installation

```shell
pip install git+https://github.com/spacetelescope/sphersgeo.git
```

### Usage

Euclidean geometry packages classify geometries into points, linestrings, and polygons (along with multi-variations: multipoints, multilinestrings, and multipolygons).
Spherical geometry analogues are spherical points, arcstrings, and spherical polygons.

Full class definitions can be found in ``sphersgeo.pyi``.

#### Points on a sphere

Spherical points are represented internally as 3-dimensional Euclidean points (X, Y, Z) relative to the origin of the unit sphere.

```python
import sphersgeo

# define a point on the sphere in angular coordinates (longitude and latitude)
a = sphersgeo.SphericalPoint((60.0, 30.0))
b = sphersgeo.SphericalPoint((60.0, 0.0))
c = sphersgeo.SphericalPoint((-30.0, -30.0))

# ... or in Euclidean coordinates (X, Y, Z)
a = sphersgeo.SphericalPoint((0.43301270189221946, 0.75, 0.5))
b = sphersgeo.SphericalPoint((0.5, 0.8660254037844386, 0.0))
c = sphersgeo.SphericalPoint((0.75, -0.4330127018922193, -0.5))
d = sphersgeo.SphericalPoint((0.0, 0.0, 1.0))
e = sphersgeo.SphericalPoint((0.0, 0.0, -1.0))

# collate multiple points together by passing lists of angular or Euclidean coordinates
ab = sphersgeo.MultiSphericalPoint([(60.0, 30.0), (60.0, 0.0)])
ab = sphersgeo.MultiSphericalPoint(
    [(0.43301270189221946, 0.75, 0.5), (0.5, 0.8660254037844386, 0.0)]
)
de = sphersgeo.MultiSphericalPoint(
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
)

# ... or a Numpy array of coordinates
import numpy as np
ab = sphersgeo.MultiSphericalPoint(np.array([(60.0, 30.0), (60.0, 0.0)]))
ab = sphersgeo.MultiSphericalPoint(
    np.array([(0.43301270189221946, 0.75, 0.5), (0.5, 0.8660254037844386, 0.0)])
)

# ... or a list of SphericalPoint objects
abcde = sphersgeo.MultiSphericalPoint([a, b, c, d, e])
```

#### Strings of great circle arcs on a sphere

A great circle arcs is the shortest geodesic distance over the surface of the sphere between any two points.
Arcstrings are comprised of an **ordered** collection of spherical points.
Arcstrings can also be **closed**, in which case the final point is considered as connected back to the first point.

```python
import sphersgeo

# define an arcstring on the sphere by passing angular or Euclidean coordinates
abc = sphersgeo.ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
de = sphersgeo.ArcString(
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
)

# ... or a list of SphericalPoint objects
abc = sphersgeo.ArcString(
    [
        sphersgeo.SphericalPoint((60.0, 0.0)),
        sphersgeo.SphericalPoint((60.0, 30.0)),
        sphersgeo.SphericalPoint((-30.0, -30.0)),
    ],
    closed=True,
)
de = sphersgeo.ArcString(
    [
        sphersgeo.SphericalPoint((0.0, 0.0, 1.0)),
        sphersgeo.SphericalPoint((0.0, 0.0, -1.0)),
    ]
)

# collate arcstrings into a MultiArcString by passing the same inputs you would to the individual objects
abc_de = sphersgeo.MultiArcString(
    [
        [(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)],
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
    ]
)

# ... or a list of ArcString objects
abc_de = sphersgeo.MultiArcString([abc, de])
```

#### Polygons on a sphere

Spherical polygons are comprised of
1. closed arcstring that represents the outer boundary, and
2. a sample point that defines which side of the closed spherical region is "inside" the boundary.

If the "inside point" is not given, the **smaller** of the two regions split by the boundary will be assigned to be the "inside".

> [!NOTE]
> Polygons in ``sphersgeo`` do NOT have holes.

```python
import sphersgeo

# define a spherical polygon by passing angular or Euclidean coordinates to form the boundary
abc = sphersgeo.SphericalPolygon([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
abc = sphersgeo.SphericalPolygon(
    [
        (0.43301270189221946, 0.75, 0.5),
        (0.5, 0.8660254037844386, 0.0),
        (0.75, -0.4330127018922193, -0.5),
    ]
)

# ... or pass an ArcString object (the arcstring is assumed to be closed)
abc = sphersgeo.SphericalPolygon(
    sphersgeo.ArcString([(60.0, 0.0), (60.0, 30.0), (-30.0, -30.0)])
)

# collate multiple polygons into a MultiSphericalPolygon by passing the same inputs you would to the individual objects
abc_def = sphersgeo.MultiSphericalPolygon(
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
abc_def = sphersgeo.MultiSphericalPolygon(
    [
        sphersgeo.SphericalPolygon([(60.0, 30.0), (60.0, 0.0), (-30.0, -30.0)]),
        sphersgeo.SphericalPolygon(
            [
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (1.0, 1.0, 0.0),
            ]
        ),
    ]
)
```
