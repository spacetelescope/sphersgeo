<a href="https://stsci.edu">
  <img src="docs/_static/stsci_logo.png" alt="STScI Logo" width="15%" style="margin-left: auto;"/>
  <img src="docs/_static/stsci_name.png" alt="STScI Name" width="68%"/>
</a>

# sphersgeo

[![build](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/build.yml)
[![tests](https://github.com/spacetelescope/sphersgeo/actions/workflows/test.yml/badge.svg)](https://github.com/spacetelescope/sphersgeo/actions/workflows/test.yml)
[![Powered by STScI](https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat)](https://www.stsci.edu)

`sphersgeo` is an object-oriented spherical geometry package written in Rust with Python accessor classes and methods.

> [!IMPORTANT]
> `sphersgeo` is still in development and does NOT currently implement all of the functionality provided by other geo packages such as `geo` or Shapely.

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.

```shell
pip install sphersgeo
```
