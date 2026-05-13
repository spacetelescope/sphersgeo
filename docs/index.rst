sphersgeo
=========

`sphersgeo <https://github.com/spacetelescope/sphersgeo>`_
is an object-oriented spherical geometry package written in Rust with Python accessor classes and methods.

.. code-block:: shell

   pip install sphersgeo

Planar geometry packages typically classify geometries into points, linestrings, and polygons
(along with multi-geometry collections: multi-points, multi-linestrings, and multi-polygons).
The spherical geometry analogues to these are spherical points, arcstrings, and spherical polygons.

==========  ==================  =================  =======================
Planar      Spherical           Planar Collection  Spherical Collection
==========  ==================  =================  =======================
Point       `SphericalPoint`    MultiPoint         `MultiSphericalPoint`
LineString  `ArcString`         MultiLineString    `MultiArcString`
Polygon     `SphericalPolygon`  MultiPolygon       `MultiSphericalPolygon`
==========  ==================  =================  =======================

.. attention::
   `sphersgeo` is still in development
   and does not currently implement all of the robust functionality provided by
   planar geometry packages such as `geo <https://docs.rs/geo/>`_ or `Shapely <https://shapely.readthedocs.io/>`_.

.. toctree::
   :maxdepth: 1
   :caption: API

   sphericalpoint.rst
   arcstring.rst
   sphericalpolygon.rst

.. toctree::
   :maxdepth: 1

   changes.rst
