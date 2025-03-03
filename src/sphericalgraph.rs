use crate::sphericalpolygon::SphericalPolygon;
use pyo3::prelude::*;

#[pyclass]
pub struct SphericalGraph {
    polygons: Vec<SphericalPolygon>,
}
