use crate::angularpolygon::AngularPolygon;
use pyo3::prelude::*;

#[pyclass]
pub struct SphericalGraph {
    polygons: Vec<AngularPolygon>,
}
