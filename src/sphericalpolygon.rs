use crate::{arcstring::ArcString, vectorpoint::VectorPoint};
use pyo3::prelude::*;

#[pyclass]
pub struct SphericalPolygon {
    arcstring: ArcString,
    interior: VectorPoint,
}
