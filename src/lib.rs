mod arcstring;
mod geometry;
mod sphericalgraph;
mod sphericalpolygon;
mod vectorpoint;

#[macro_use]
extern crate impl_ops;
extern crate numpy;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[pymodule(name = "sphersgeo")]
pub mod sphersgeo {
    use super::*;
    use crate::arcstring::{angle, angles, arc_length, interpolate};
    use crate::geometry::{BoundingBox, Distance};

    #[pymodule_export]
    use super::vectorpoint::VectorPoint;

    #[pymethods]
    impl VectorPoint {
        #[new]
        fn __new__<'py>(xyz: PyReadonlyArray1<'py, f64>) -> PyResult<Self> {
            Self::try_from(xyz.as_array().to_owned())
                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        /// from the given coordinates, build an xyz vector representing a point on the sphere
        #[classmethod]
        #[pyo3(name="from_lonlat", signature=(coordinates,degrees=true))]
        fn py_from_lonlat<'py>(
            cls: &Bound<'_, PyType>,
            coordinates: PyReadonlyArray1<'py, f64>,
            degrees: bool,
        ) -> Self {
            Self::from_lonlat(&coordinates.as_array(), degrees)
        }

        /// xyz vector as a 1-dimensional array of 3 floats
        #[getter]
        #[pyo3(name = "xyz")]
        fn py_xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        /// convert this point on the sphere to angular coordinates
        #[pyo3(name="to_lonlat", signature=(degrees=true))]
        fn py_to_lonlat<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray1<f64>> {
            self.to_lonlat(degrees).into_pyarray(py)
        }

        /// angular distance on the sphere between this point and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: &Self) -> f64 {
            self.distance(other)
        }

        /// normalize this vector to length 1 (the unit sphere) while preserving direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// angle on the sphere between this point and two other points
        #[pyo3(name="angle", signature=(a,b,degrees=true))]
        fn py_angle(&self, a: &VectorPoint, b: &VectorPoint, degrees: bool) -> f64 {
            self.angle(a, b, degrees)
        }

        /// whether this point lies exactly between the given points
        #[pyo3(name = "collinear")]
        fn py_collinear(&self, a: &VectorPoint, b: &VectorPoint) -> bool {
            self.collinear(a, b)
        }

        /// length of the underlying xyz vector
        #[getter]
        #[pyo3(name = "vector_length")]
        fn py_vector_length(&self) -> f64 {
            self.vector_length()
        }

        /// rotate this xyz vector by theta angle around another xyz vector
        #[pyo3(name="vector_rotate_around", signature=(other,theta,degrees=true))]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        fn __add__(&self, other: &Self) -> MultiVectorPoint {
            self + other
        }

        fn __eq__(&self, other: &Self) -> bool {
            &self.xyz == &other.xyz
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use super::vectorpoint::MultiVectorPoint;

    #[pymethods]
    impl MultiVectorPoint {
        #[new]
        fn __new__<'py>(xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Self> {
            Self::try_from(xyz.as_array().to_owned())
                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        /// from the given coordinates, build xyz vectors representing points on the sphere
        #[classmethod]
        #[pyo3(name = "from_lonlats", signature=(coordinates, degrees = true))]
        fn py_from_lonlats<'py>(
            cls: &Bound<'_, PyType>,
            coordinates: PyReadonlyArray2<'py, f64>,
            degrees: bool,
        ) -> Self {
            Self::from_lonlats(&coordinates.as_array(), degrees)
        }

        /// xyz vector as a 2-dimensional array of Nx3 floats
        #[getter]
        #[pyo3(name = "xyz")]
        fn py_xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        /// bounding box [minX,minY,maxX,maxY]
        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> [f64; 4] {
            self.bounds(degrees)
        }

        /// whether the given point is one of these points
        #[pyo3(name = "contains")]
        fn py_contains(&self, point: &VectorPoint) -> bool {
            self.contains(point)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name = "to_lonlats", signature=(degrees=true))]
        fn py_to_lonlats<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray2<f64>> {
            self.to_lonlats(degrees).into_pyarray(py)
        }

        /// angular distance on the sphere between these points and another set; returns None if overlaps
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: &Self) -> f64 {
            self.distance(other)
        }

        /// normalize the underlying vectors to length 1 (the unit sphere) while preserving direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// rotate the underlying vector by theta angle around other vectors
        #[pyo3(name="rotate_around", signature=(other,theta,degrees=true))]
        fn py_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        fn __concat__(&self, other: &Self) -> Self {
            self + other
        }

        /// angles on the sphere between these points and other sets of points
        #[pyo3(name="angles",signature=(a,b,degrees=true))]
        fn py_angles<'py>(
            &self,
            py: Python<'py>,
            a: &MultiVectorPoint,
            b: &MultiVectorPoint,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            self.angles(a, b, degrees).to_pyarray(py)
        }

        /// whether these points lie exactly between the given points
        #[pyo3(name = "collinear")]
        fn py_collinear<'py>(
            &self,
            py: Python<'py>,
            a: &VectorPoint,
            b: &VectorPoint,
        ) -> Bound<'py, PyArray1<bool>> {
            self.collinear(a, b).to_pyarray(py)
        }

        /// list of vector points
        #[getter]
        #[pyo3(name = "vector_points")]
        fn py_vector_points(&self) -> Vec<VectorPoint> {
            self.to_owned().into()
        }

        /// lengths of the underlying xyz vectors
        #[getter]
        #[pyo3(name = "vector_lengths")]
        fn py_vector_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.vector_lengths().to_pyarray(py)
        }

        fn __len__(&self) -> usize {
            self.length()
        }

        fn __add__(&self, other: &Self) -> Self {
            self + other
        }

        fn __eq__(&self, other: &Self) -> bool {
            &self.xyz == &other.xyz
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    /// radians subtended by this arc on the sphere
    #[pyfunction]
    #[pyo3(name = "arc_length")]
    fn py_arc_length(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
        arc_length(&a.as_array(), &b.as_array())
    }

    /// generate the given number of points at equal intervals between vectorpoints A and B
    #[pyfunction]
    #[pyo3(name="interpolate", signature=(a,b,n=50))]
    fn py_interpolate<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        n: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        interpolate(&a.as_array(), &b.as_array(), n).to_pyarray(py)
    }

    /// generate the given number of points at equal intervals between vectorpoints A and B
    #[pyfunction]
    #[pyo3(name="angle", signature=(a,b,c,degrees=true))]
    fn py_angle(
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        c: PyReadonlyArray1<f64>,
        degrees: bool,
    ) -> f64 {
        angle(&a.as_array(), &b.as_array(), &c.as_array(), degrees)
    }

    /// generate the given number of points at equal intervals between vectorpoints A and B
    #[pyfunction]
    #[pyo3(name="angles", signature=(a,b,c,degrees=true))]
    fn py_angles<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
        c: PyReadonlyArray2<f64>,
        degrees: bool,
    ) -> Bound<'py, PyArray1<f64>> {
        angles(&a.as_array(), &b.as_array(), &c.as_array(), degrees).to_pyarray(py)
    }

    #[pymodule_export]
    use super::arcstring::ArcString;

    #[pymethods]
    impl ArcString {
        #[new]
        fn __new__(points: &MultiVectorPoint) -> Self {
            points.to_owned().into()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points.to_owned()
        }

        /// radians subtended by each arc on the sphere
        #[getter]
        #[pyo3(name = "lengths")]
        fn py_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.lengths().to_pyarray(py)
        }

        /// total radians subtended by this arcstring on the sphere
        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        /// individual arcs along this string
        #[getter]
        #[pyo3(name = "arcs")]
        fn py_arcs(&self) -> Vec<ArcString> {
            self.to_owned().into()
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiVectorPoint {
            self.midpoints()
        }

        /// bounding box [minX,minY,maxX,maxY]
        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> [f64; 4] {
            self.bounds(degrees)
        }

        /// whether this arcstring contains the given point
        #[pyo3(name = "contains")]
        fn py_contains(&self, point: &VectorPoint) -> bool {
            self.contains(point)
        }

        /// whether this arcstring and another given arcstring intersect
        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: &Self) -> bool {
            self.intersects(other)
        }

        /// point(s) at which this arcstring and another given arcstring intersect
        #[pyo3(name = "intersection", signature=(other))]
        fn py_intersection(&self, other: &Self) -> Option<MultiVectorPoint> {
            self.intersection(other)
        }

        fn __add__(&self, other: &Self) -> Self {
            self + other
        }

        fn __eq__(&self, other: &Self) -> bool {
            &self.points == &other.points
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use crate::sphericalpolygon::SphericalPolygon;

    #[pymethods]
    impl SphericalPolygon {}

    #[pymodule_export]
    use crate::sphericalgraph::SphericalGraph;

    #[pymethods]
    impl SphericalGraph {}
}
