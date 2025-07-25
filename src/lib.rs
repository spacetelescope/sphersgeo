mod arcstring;
mod geometry;
mod sphericalgraph;
mod sphericalpolygon;
mod vectorpoint;

#[macro_use]
extern crate impl_ops;
extern crate numpy;

use numpy::ndarray::{s, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyType};

#[pymodule(name = "sphersgeo")]
pub mod sphersgeo {
    use super::*;
    use crate::arcstring::{angle, arc_length, collinear, interpolate};
    use crate::vectorpoint::{normalize_vector, normalize_vectors};

    /// xyz vector representing a point on the sphere
    #[pymodule_export]
    use super::vectorpoint::VectorPoint;

    #[pymethods]
    impl VectorPoint {
        #[new]
        fn py_init<'py>(xyz: PyReadonlyArray1<'py, f64>) -> Self {
            Self::new(&xyz.as_array())
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
        fn xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        /// convert this point on the sphere to angular coordinates
        #[pyo3(name="to_lonlat", signature=(degrees=true))]
        fn py_to_lonlat<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray1<f64>> {
            self.to_lonlat(degrees).into_pyarray(py)
        }

        /// length of the underlying xyz vector
        #[getter]
        pub fn vector_length(&self) -> f64 {
            self.xyz.pow2().sum().sqrt()
        }

        pub fn distance(&self, other: &Self) -> f64 {
            arc_length(&self.xyz.view(), &other.xyz.view())
        }

        /// normalize this vector to length 1 (the unit sphere) while preserving direction
        #[getter]
        pub fn normalized(&self) -> Self {
            Self {
                xyz: normalize_vector(&self.xyz.view()),
            }
        }

        /// rotate this xyz vector by theta angle around another xyz vector
        #[pyo3(name = "vector_rotate_around", signature=(other,theta,degrees=true))]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        /// angle on the sphere between this point and two other points
        #[pyo3(signature=(a,b,degrees=true))]
        pub fn angle(&self, a: &VectorPoint, b: &VectorPoint, degrees: bool) -> f64 {
            angle(&a.into(), &self.into(), &b.into(), degrees)
        }

        /// whether this point lies exactly between the given points
        fn collinear(&self, a: &VectorPoint, b: &VectorPoint) -> bool {
            collinear(&a.xyz.view(), &self.xyz.view(), &b.xyz.view())
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
    }

    #[pymodule_export]
    use super::vectorpoint::MultiVectorPoint;

    #[pymethods]
    impl MultiVectorPoint {
        #[new]
        fn py_init<'py>(xyz: PyReadonlyArray2<'py, f64>) -> Self {
            Self::new(&xyz.as_array())
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
        fn xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        /// list of vector points
        #[getter]
        pub fn vectorpoints(&self) -> Vec<VectorPoint> {
            self.xyz
                .rows()
                .into_iter()
                .map(|row| VectorPoint {
                    xyz: row.to_owned(),
                })
                .collect::<Vec<VectorPoint>>()
        }

        /// whether the given point is one of these points
        pub fn contains(&self, point: &VectorPoint) -> bool {
            Zip::from((&self.xyz - &point.xyz).rows()).any(|diff| diff.sum() < 3e-11)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name = "to_lonlats", signature=(degrees=true))]
        fn py_to_lonlats<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray2<f64>> {
            self.to_lonlats(degrees).into_pyarray(py)
        }

        /// lengths of the underlying xyz vectors
        #[getter]
        #[pyo3(name = "vectorlengths")]
        pub fn py_vectorlengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.vector_lengths().to_pyarray(py)
        }

        /// normalize the underlying vectors to length 1 (the unit sphere) while preserving direction
        #[getter]
        pub fn normalized(&self) -> Self {
            Self {
                xyz: normalize_vectors(&self.xyz.view()),
            }
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
        pub fn py_angles<'py>(
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
        pub fn py_collinear<'py>(
            &self,
            py: Python<'py>,
            a: &VectorPoint,
            b: &VectorPoint,
        ) -> Bound<'py, PyArray1<bool>> {
            self.collinear(a, b).to_pyarray(py)
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
    }

    #[pymodule_export]
    use super::arcstring::ArcString;

    #[pymethods]
    impl ArcString {
        #[new]
        fn new(points: &MultiVectorPoint) -> Self {
            Self {
                points: points.to_owned(),
            }
        }

        #[getter]
        pub fn points(&self) -> MultiVectorPoint {
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
        pub fn length(&self) -> f64 {
            self.lengths().sum()
        }

        /// individual arcs along this string
        #[getter]
        pub fn arcs(&self) -> Vec<ArcString> {
            let vectors = &self.points.xyz;
            let mut arcs = vec![];
            for index in 0..vectors.nrows() - 1 {
                arcs.push(ArcString {
                    points: MultiVectorPoint {
                        xyz: vectors.slice(s![index..index + 1, ..]).to_owned(),
                    },
                })
            }

            arcs
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiVectorPoint {
            self.midpoints()
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
}
