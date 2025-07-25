# sphersgeo

#### object-oriented spherical geometry in Rust (and Python bindings)

> [!IMPORTANT]
> `sphersgeo` is still in development and does NOT currently implement all of the functionality provided by other geo packages such as `geo` or Shapely.

> [!NOTE]
> Intersections between geometries are NOT rigorous; the `.intersection()` function will ONLY return the lower order of geometry being compared, and does NOT handle degenerate cases / touching geometries.
