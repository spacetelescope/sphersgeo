Geometries
==========

Planar geometry packages typically classify geometries into points, linestrings (also called polylines), and polygons
(along with multi-geometry collections: multi-points, multi-linestrings / multi-polylines, and multi-polygons).
The spherical geometry analogues to these are spherical points, arcstrings, and spherical polygons.

====================  ============================  =========================  =================================
Planar                Spherical                     Planar Collection          Spherical Collection
====================  ============================  =========================  =================================
`shapely.Point`       `sphersgeo.SphericalPoint`    `shapely.MultiPoint`       `sphersgeo.MultiSphericalPoint`
`shapely.LineString`  `sphersgeo.ArcString`         `shapely.MultiLineString`  `sphersgeo.MultiArcString`
`shapely.Polygon`     `sphersgeo.SphericalPolygon`  `shapely.MultiPolygon`     `sphersgeo.MultiSphericalPolygon`
====================  ============================  =========================  =================================
