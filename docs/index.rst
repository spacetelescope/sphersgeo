sphersgeo
=========

`sphersgeo <https://github.com/spacetelescope/sphersgeo>`_
is an object-oriented spherical geometry package written in Rust with Python accessor classes and methods.

.. tab:: Python

   .. code-block:: shell

      pip install sphersgeo

.. tab:: Rust

   .. code-block:: shell

      cargo install --git https://github.com/spacetelescope/sphersgeo

.. attention::
   `sphersgeo` is still in development
   and does not currently implement all the robust functionality provided by
   planar geometry packages such as `geo <https://docs.rs/geo/>`_ or `Shapely <https://shapely.readthedocs.io/>`_.

.. toctree::
   :maxdepth: 2

   geometry.rst

.. toctree::
   :maxdepth: 2
   :caption: Python API

   python/sphericalpoint.rst
   python/arcstring.rst
   python/sphericalpolygon.rst

.. toctree::
   :maxdepth: 1
   :caption: Other

   changes.rst
