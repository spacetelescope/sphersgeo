0.2.2 (2026-09-08)
==================

Fixes
-----

- support reading WKT with ``Z``, and drop ``M`` values (`#24
  <https://github.com/spacetelescope/sphersgeo/issues/24>`_)
- invert negative area of clockwise polygons (`#25
  <https://github.com/spacetelescope/sphersgeo/issues/25>`_)
- support writing WKT with ``Z`` (`#26
  <https://github.com/spacetelescope/sphersgeo/issues/26>`_)

0.2.1 (2026-09-02)
==================

Other Changes
-------------

- add common geometries and parametrize tests (`#22
  <https://github.com/spacetelescope/sphersgeo/issues/22>`_)
- update ``kiddo`` to ``6.1.0`` (`#23
  <https://github.com/spacetelescope/sphersgeo/issues/23>`_)


0.2.0 (2026-07-20)
==================

Documentation Changes
---------------------

- support serializing / deserializing geometries into well-known text (`#19
  <https://github.com/spacetelescope/sphersgeo/issues/19>`_)
- restructure self-operations into unary (`#17
  <https://github.com/spacetelescope/sphersgeo/issues/17>`_)

Other Changes
-------------

- allow slicing and negative indexing with ``__getitem__`` (`#18
  <https://github.com/spacetelescope/sphersgeo/issues/18>`_)


0.1.1 (2026-05-19)
==================

Documentation Changes
---------------------

- remove ``license`` from ``Cargo.toml`` in favor of keeping ``license-file`` (`#15
  <https://github.com/spacetelescope/sphersgeo/issues/15>`_)
- update branding and switch theme to ``furo`` (`#16
  <https://github.com/spacetelescope/sphersgeo/issues/16>`_)


0.1.0 (2026-05-12)
==================

Breaking Changes
----------------

- orientation of polygons must now be counterclockwise, such that the inside of
  the polygon is always to the left of the boundary (`#10
  <https://github.com/spacetelescope/sphersgeo/issues/10>`_)


Documentation Changes
---------------------

- build documentation with Sphinx and ``autoapi``, and set up ReadTheDocs
  configuration (`#12
  <https://github.com/spacetelescope/sphersgeo/issues/12>`_)
- move change log to its own ``toctree`` and distinguish Rust from Python install
  options (`#14 <https://github.com/spacetelescope/sphersgeo/issues/14>`_)


0.0.3 (2026-05-04)
==================

Documentation Changes
---------------------

- add docstrings to Python classes and methods (`#9
  <https://github.com/spacetelescope/sphersgeo/issues/9>`_)


0.0.2 (2026-04-30)
==================

Fixes
-----

- fix typing by moving ``sphersgeo.pyi`` stub file into ``src/python/sphersgeo/``
  (see https://pyo3.rs/main/python-typing-hints and
  https://github.com/PyO3/maturin/blob/0dee40510083c03607834c821eea76964140a126/Readme.md#mixed-rustpython-projects)
  (`#8 <https://github.com/spacetelescope/sphersgeo/issues/8>`_)

Documentation Changes
---------------------

- fix reference to ``src/sphersgeo.pyi`` in README (`#5
  <https://github.com/spacetelescope/sphersgeo/issues/5>`_)
