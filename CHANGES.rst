0.1.0 (2026-05-12)
==================

Breaking Changes
----------------

- orientation of polygons must now be counterclockwise, such that the inside of
  the polygon is always to the left of the boundary (`#10
  <https://github.com/spacetelescope/sphersgeo/issues/10>`_)


Documentation Changes
---------------------

- build documentation with Sphinx and `autoapi`, and set up ReadTheDocs
  configuration (`#12
  <https://github.com/spacetelescope/sphersgeo/issues/12>`_)
- move change log to its own `toctree` and distinguish Rust from Python install
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

- fix typing by moving `sphersgeo.pyi` stub file into `src/python/sphersgeo/`
  (see https://pyo3.rs/main/python-typing-hints and
  https://github.com/PyO3/maturin/blob/0dee40510083c03607834c821eea76964140a126/Readme.md#mixed-rustpython-projects)
  (`#8 <https://github.com/spacetelescope/sphersgeo/issues/8>`_)

Documentation Changes
---------------------

- fix reference to `src/sphersgeo.pyi` in README (`#5
  <https://github.com/spacetelescope/sphersgeo/issues/5>`_)

