#![allow(refining_impl_trait)]
mod arcstring;
mod edgegraph;
mod geometry;
mod sphericalpoint;
mod sphericalpolygon;

#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg(feature = "py")]
extern crate numpy;

#[cfg(feature = "py")]
#[pymodule(name = "sphersgeo")]
mod py_sphersgeo {
    #[cfg(feature = "py")]
    #[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq)]
    enum AnyGeometry {
        #[pyo3(transparent)]
        SphericalPoint(crate::sphericalpoint::SphericalPoint),
        #[pyo3(transparent)]
        MultiSphericalPoint(crate::sphericalpoint::MultiSphericalPoint),
        #[pyo3(transparent)]
        ArcString(crate::arcstring::ArcString),
        #[pyo3(transparent)]
        MultiArcString(crate::arcstring::MultiArcString),
        #[pyo3(transparent)]
        SphericalPolygon(crate::sphericalpolygon::SphericalPolygon),
        #[pyo3(transparent)]
        MultiSphericalPolygon(crate::sphericalpolygon::MultiSphericalPolygon),
    }

    use super::*;
    use crate::geometry::{GeometricOperations, GeometricPredicates, Geometry, MultiGeometry};
    use numpy::{
        ndarray::{Array1, Array2},
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    };
    use pyo3::{
        exceptions::{PyKeyError, PyValueError},
        types::PyType,
    };

    #[pymodule_export]
    use super::sphericalpoint::SphericalPoint;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PySphericalPointInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        Array([f64; 3]),
        Tuple((f64, f64, f64)),
        NumpyArray(PyReadonlyArray1<'py, f64>),
        List(Vec<f64>),
    }

    #[derive(FromPyObject)]
    enum PySphericalPointLonLatInputs<'py> {
        Array([f64; 2]),
        Tuple((f64, f64)),
        NumpyArray(PyReadonlyArray1<'py, f64>),
        List(Vec<f64>),
    }

    #[pymethods]
    impl SphericalPoint {
        #[new]
        fn py_new(xyz: PySphericalPointInputs) -> PyResult<Self> {
            Ok(match xyz {
                PySphericalPointInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::SphericalPoint(point) => Ok(point),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive point from {geometry:?}",
                    ))),
                }?,
                PySphericalPointInputs::Array(xyz) => Self::from(xyz),
                PySphericalPointInputs::Tuple(xyz) => Self::from(&xyz),
                PySphericalPointInputs::NumpyArray(xyz) => {
                    Self::try_from(&xyz.as_array()).map_err(PyValueError::new_err)?
                }
                PySphericalPointInputs::List(xyz) => {
                    Self::try_from(&xyz).map_err(PyValueError::new_err)?
                }
            })
        }

        #[classmethod]
        #[pyo3(name = "from_lonlat")]
        /// from the given coordinates, build an xyz vector representing a point on the sphere
        fn py_from_lonlat<'py>(
            _: &Bound<'py, PyType>,
            lonlat: PySphericalPointLonLatInputs,
        ) -> Self {
            let lonlat = match lonlat {
                PySphericalPointLonLatInputs::Array(lonlat) => lonlat,
                PySphericalPointLonLatInputs::NumpyArray(lonlat) => {
                    let lonlat = lonlat.as_array();
                    [lonlat[0], lonlat[1]]
                }
                PySphericalPointLonLatInputs::Tuple((lon, lat)) => [lon, lat],
                PySphericalPointLonLatInputs::List(list) => [list[0], list[1]],
            };

            Self::from_lonlat(&lonlat)
        }

        #[getter]
        /// xyz vector as a 1-dimensional array of 3 floats
        fn get_xyz(&self) -> [f64; 3] {
            self.xyz
        }

        #[getter]
        /// convert this point on the sphere to angular coordinates
        fn get_lonlat(&self) -> [f64; 2] {
            self.to_lonlat()
        }

        #[pyo3(name = "two_arc_angle")]
        /// angle on the sphere between this point and two other points
        fn py_two_arc_angle(
            &self,
            start: PySphericalPointInputs,
            end: PySphericalPointInputs,
        ) -> PyResult<f64> {
            Ok(self.two_arc_angle(&Self::py_new(start)?, &Self::py_new(end)?))
        }

        #[pyo3(name = "collinear")]
        /// whether this point shares a line with two other points
        fn py_collinear(
            &self,
            a: PySphericalPointInputs,
            b: PySphericalPointInputs,
        ) -> PyResult<bool> {
            Ok(self.collinear(&Self::py_new(a)?, &Self::py_new(b)?))
        }

        #[pyo3(name = "is_clockwise_turn")]
        /// whether the angle formed between this point and two other points is a clockwise turn
        fn py_is_clockwise_turn(
            &self,
            start: PySphericalPointInputs,
            end: PySphericalPointInputs,
        ) -> PyResult<bool> {
            Ok(self.is_clockwise_turn(&Self::py_new(start)?, &Self::py_new(end)?))
        }

        #[pyo3(name = "interpolate_points")]
        /// create n number of points equally spaced on an arc between this point and another point
        fn py_interpolate_points(
            &self,
            end: PySphericalPointInputs,
            n: usize,
        ) -> PyResult<MultiSphericalPoint> {
            self.interpolate_points(&Self::py_new(end)?, n)
                .map_err(PyValueError::new_err)
        }

        #[getter]
        /// length of the underlying xyz vector
        fn get_vector_length(&self) -> f64 {
            self.vector_length()
        }

        #[pyo3(name = "vector_cross")]
        /// cross product of this xyz vector with another xyz vector
        fn py_vector_cross(&self, other: PySphericalPointInputs) -> PyResult<Self> {
            Ok(self.vector_cross(&Self::py_new(other)?))
        }

        #[pyo3(name = "vector_dot")]
        /// dot product of this xyz vector with another xyz vector
        fn py_vector_dot(&self, other: PySphericalPointInputs) -> PyResult<f64> {
            Ok(self.vector_dot(&Self::py_new(other)?))
        }

        #[pyo3(name = "vector_rotate_around")]
        /// rotate this xyz vector by theta angle around another xyz vector
        fn py_vector_rotate_around(
            &self,
            other: PySphericalPointInputs,
            theta: f64,
        ) -> PyResult<Self> {
            Ok(self.vector_rotate_around(&Self::py_new(other)?, &theta))
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<SphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .union(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => {
                    self.union(&arcstring).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .union(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .union(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPoint),
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.intersection(&polygon).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::SphericalPoint),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multipolygon))
                }
            }
        }

        fn __add__(&self, other: PySphericalPointInputs) -> PyResult<SphericalPoint> {
            Ok(self + &Self::py_new(other)?)
        }

        fn __sub__(&self, other: PySphericalPointInputs) -> PyResult<SphericalPoint> {
            Ok(self - &Self::py_new(other)?)
        }

        fn __mul__(&self, other: PySphericalPointInputs) -> PyResult<SphericalPoint> {
            Ok(self * &Self::py_new(other)?)
        }

        fn __div__(&self, other: PySphericalPointInputs) -> PyResult<SphericalPoint> {
            Ok(self / &Self::py_new(other)?)
        }

        fn __eq__(&self, other: PySphericalPointInputs) -> PyResult<bool> {
            Ok(self == &Self::py_new(other)?)
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use super::sphericalpoint::MultiSphericalPoint;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiSphericalPointInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        ListOfPoints(Vec<SphericalPoint>),
        ListOfArrays(Vec<[f64; 3]>),
        ListOfTuples(Vec<(f64, f64, f64)>),
        NumpyArray(PyReadonlyArray2<'py, f64>),
        NestedList(Vec<Vec<f64>>),
    }

    #[derive(FromPyObject)]
    enum PyMultiSphericalPointLonLatInputs<'py> {
        ListOfArrays(Vec<[f64; 2]>),
        ListOfTuples(Vec<(f64, f64)>),
        NumpyArray(PyReadonlyArray2<'py, f64>),
        NestedList(Vec<Vec<f64>>),
    }

    #[pymethods]
    impl MultiSphericalPoint {
        #[new]
        fn py_new(xyzs: PyMultiSphericalPointInputs) -> PyResult<Self> {
            match xyzs {
                PyMultiSphericalPointInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::MultiSphericalPoint(multipoint) => Ok(multipoint),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive multipoint from {geometry:?}",
                    ))),
                },
                PyMultiSphericalPointInputs::ListOfPoints(points) => {
                    Self::try_from(points).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfArrays(xyzs) => {
                    Self::try_from(xyzs).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfTuples(xyzs) => {
                    Self::try_from(&xyzs).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::NumpyArray(xyzs) => {
                    Self::try_from(xyzs.as_array().to_owned()).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::NestedList(xyzs) => {
                    Self::try_from(&xyzs).map_err(PyValueError::new_err)
                }
            }
        }

        #[classmethod]
        #[pyo3(name = "from_lonlats")]
        /// from the given coordinates, build xyz vectors representing points on the sphere
        fn py_from_lonlats<'py>(
            _: &Bound<'py, PyType>,
            lonlats: PyMultiSphericalPointLonLatInputs,
        ) -> PyResult<Self> {
            let lonlats = match lonlats {
                PyMultiSphericalPointLonLatInputs::ListOfArrays(lonlats) => lonlats,
                PyMultiSphericalPointLonLatInputs::ListOfTuples(lonlats) => {
                    lonlats.iter().map(|lonlat| [lonlat.0, lonlat.1]).collect()
                }
                PyMultiSphericalPointLonLatInputs::NumpyArray(lonlats) => lonlats
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|lonlat| [lonlat[0], lonlat[1]])
                    .collect(),
                PyMultiSphericalPointLonLatInputs::NestedList(lonlats) => lonlats
                    .iter()
                    .map(|lonlat| [lonlat[0], lonlat[1]])
                    .collect(),
            };

            match Self::try_from_lonlats(&lonlats) {
                Ok(result) => Ok(result),
                Err(err) => Err(PyValueError::new_err(err)),
            }
        }

        #[getter]
        /// xyz vectors as a 2-dimensional array of Nx3 floats
        fn get_xyzs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            Array2::<f64>::from(self).into_pyarray(py)
        }

        #[getter]
        /// convert to angle coordinates along the sphere
        fn get_lonlats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            Array2::<f64>::from(self.to_lonlats()).into_pyarray(py)
        }

        #[pyo3(name = "nearest")]
        /// retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point across the unit sphere
        fn py_nearest(&self, other: PySphericalPointInputs) -> PyResult<(SphericalPoint, f64)> {
            Ok(self.nearest(&SphericalPoint::py_new(other)?))
        }

        #[pyo3(name = "vectors_rotate_around")]
        /// rotate the underlying vector by theta angle around other vectors
        fn py_vectors_rotate_around(
            &self,
            other: PyMultiSphericalPointInputs,
            theta: f64,
        ) -> PyResult<Self> {
            Ok(self.vectors_rotate_around(&Self::py_new(other)?, theta))
        }

        #[getter]
        /// lengths of the underlying xyz vectors
        fn get_vectors_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            Array1::<f64>::from(self.vectors_lengths()).into_pyarray(py)
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .union(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => {
                    self.union(&arcstring).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .union(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiSphericalPoint)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .union(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPoint),
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPoint),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiSphericalPoint(self.symmetric_difference(&multipolygon))
                }
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<SphericalPoint> {
            self.to_owned().into()
        }

        /// number of points in this collection
        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: usize) -> SphericalPoint {
            SphericalPoint {
                xyz: self.xyzs[index],
            }
        }

        #[pyo3(name = "append")]
        fn py_append(&mut self, point: PySphericalPointInputs) -> PyResult<()> {
            self.push(SphericalPoint::py_new(point)?);
            Ok(())
        }

        #[pyo3(name = "extend")]
        fn py_extend(&mut self, points: PyMultiSphericalPointInputs) -> PyResult<()> {
            self.extend(Self::py_new(points)?);
            Ok(())
        }

        fn __iadd__(&mut self, points: PyMultiSphericalPointInputs) -> PyResult<()> {
            *self += &Self::py_new(points)?;
            Ok(())
        }

        fn __add__(&self, points: PyMultiSphericalPointInputs) -> PyResult<Self> {
            Ok(self + &Self::py_new(points)?)
        }

        fn __eq__(&self, other: PyMultiSphericalPointInputs) -> PyResult<bool> {
            Ok(self == &Self::py_new(other)?)
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use super::arcstring::ArcString;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyArcStringInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        MultiPointInput(PyMultiSphericalPointInputs<'py>),
    }

    #[pymethods]
    impl ArcString {
        #[new]
        #[pyo3(signature=(arcstring, closed=None))]
        fn py_new(arcstring: PyArcStringInputs, closed: Option<bool>) -> PyResult<Self> {
            match arcstring {
                PyArcStringInputs::AnyGeometry(geometry) => {
                    let mut instance = match geometry {
                        AnyGeometry::MultiSphericalPoint(multipoint) => {
                            ArcString::try_from(multipoint).map_err(PyValueError::new_err)
                        }
                        AnyGeometry::ArcString(arcstring) => Ok(arcstring),
                        AnyGeometry::SphericalPolygon(polygon) => Ok(polygon.boundary),
                        _ => Err(PyValueError::new_err(format!(
                            "cannot derive arcstring from {geometry:?}",
                        ))),
                    }?;

                    if let Some(is_closed) = closed {
                        if is_closed != instance.closed {
                            instance.closed = is_closed;
                        }
                    }

                    Ok(instance)
                }
                PyArcStringInputs::MultiPointInput(points) => {
                    Self::try_new(MultiSphericalPoint::py_new(points)?, closed)
                        .map_err(PyValueError::new_err)
                }
            }
        }

        /// number of arcs in this arcstring
        fn __len__(&self) -> usize {
            self.points.len() - 1
        }

        #[getter]
        /// radians subtended by each arc on the sphere
        fn get_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            Array1::<f64>::from(self.lengths()).into_pyarray(py)
        }

        #[getter]
        /// midpoints of each arc
        fn get_midpoints(&self) -> MultiSphericalPoint {
            self.midpoints()
        }

        #[getter]
        /// whether this arcstring is "closed" (the last vertex is connected to the first)
        fn get_closed(&self) -> bool {
            self.closed
        }

        #[setter]
        /// "close" this arcstring (connect the last vertex to the first)
        fn set_closed(&mut self, closed: bool) {
            self.closed = closed
        }

        #[getter]
        /// whether this arcstring crosses itself
        fn get_crosses_self(&self) -> bool {
            self.crosses_self()
        }

        #[getter]
        /// points at which this arcstring crosses itself
        fn get_crossings_with_self(&self) -> Option<MultiSphericalPoint> {
            self.crossings_with_self()
        }

        #[pyo3(name = "adjoins")]
        /// if this arcstring's endpoints touch another's
        fn py_adjoins(&self, other: PyArcStringInputs) -> PyResult<bool> {
            Ok(self.adjoins(&Self::py_new(other, None)?))
        }

        #[pyo3(name = "join")]
        /// join this arcstring's endpoint(s) to another
        fn py_join(&self, other: PyArcStringInputs) -> PyResult<Option<ArcString>> {
            Ok(self.join(&Self::py_new(other, None)?))
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    self.union(&multipoint).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::ArcString(arcstring) => {
                    self.union(&arcstring).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    self.union(&multiarcstring).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.union(&multipolygon).map(AnyGeometry::MultiArcString)
                }
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.intersection(&polygon).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiArcString),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multipolygon))
                }
            }
        }

        fn __eq__(&self, other: PyArcStringInputs) -> PyResult<bool> {
            Ok(self == &Self::py_new(other, None)?)
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use super::arcstring::MultiArcString;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiArcStringInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        ListOfArcStrings(Vec<PyArcStringInputs<'py>>),
    }

    #[pymethods]
    impl MultiArcString {
        #[new]
        fn py_new(arcstrings: PyMultiArcStringInputs) -> PyResult<Self> {
            match arcstrings {
                PyMultiArcStringInputs::AnyGeometry(geometry) => {
                    match geometry {
                        AnyGeometry::MultiSphericalPoint(multipoint) => Self::try_from(vec![
                            ArcString::try_from(multipoint).map_err(PyValueError::new_err)?,
                        ])
                        .map_err(PyValueError::new_err),
                        AnyGeometry::ArcString(arcstring) => {
                            Ok(Self::try_from(vec![arcstring]).expect("invalid arcstring"))
                        }
                        AnyGeometry::MultiArcString(multiarcstring) => Ok(multiarcstring),
                        AnyGeometry::SphericalPolygon(polygon) => {
                            Ok(Self::try_from(vec![polygon.boundary])
                                .expect("polygon has no boundary"))
                        }
                        AnyGeometry::MultiSphericalPolygon(multipolygon) => Ok(multipolygon
                            .boundary()
                            .expect("multipolygon has no boundary")),
                        _ => Err(PyValueError::new_err(format!(
                            "cannot derive multiarcstring from {geometry:?}"
                        ))),
                    }
                }
                PyMultiArcStringInputs::ListOfArcStrings(arcstring_inputs) => {
                    let mut arcstrings = vec![];
                    for arcstring_input in arcstring_inputs {
                        arcstrings.push(ArcString::py_new(arcstring_input, None)?);
                    }
                    Self::try_from(arcstrings).map_err(PyValueError::new_err)
                }
            }
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    self.union(&multipoint).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::ArcString(arcstring) => {
                    self.union(&arcstring).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    self.union(&multiarcstring).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.union(&multipolygon).map(AnyGeometry::MultiArcString)
                }
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.intersection(&polygon).map(AnyGeometry::MultiArcString)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiArcString),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiArcString(self.symmetric_difference(&multipolygon))
                }
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<ArcString> {
            self.arcstrings.to_owned()
        }

        /// number of arcstrings in this collection
        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: usize) -> PyResult<ArcString> {
            self.arcstrings
                .get(index)
                .map(|arcstring| arcstring.to_owned())
                .ok_or(PyKeyError::new_err(format!(
                    "invalid arcstring index for list of length {}",
                    self.arcstrings.len()
                )))
        }

        #[pyo3(name = "append")]
        fn py_append(&mut self, arcstring: PyArcStringInputs) -> PyResult<()> {
            self.push(ArcString::py_new(arcstring, None)?);
            Ok(())
        }

        #[pyo3(name = "extend")]
        fn py_extend(&mut self, arcstrings: PyMultiArcStringInputs) -> PyResult<()> {
            self.extend(Self::py_new(arcstrings)?);
            Ok(())
        }

        fn __iadd__(&mut self, arcstrings: PyMultiArcStringInputs) -> PyResult<()> {
            *self += &Self::py_new(arcstrings)?;
            Ok(())
        }

        fn __add__(&self, arcstrings: PyMultiArcStringInputs) -> PyResult<Self> {
            Ok(self + &Self::py_new(arcstrings)?)
        }

        fn __eq__(&self, other: PyMultiArcStringInputs) -> PyResult<bool> {
            Ok(self == &Self::py_new(other)?)
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

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PySphericalPolygonInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        ArcStringInput(PyArcStringInputs<'py>),
        ArcStringInputWithInteriorPoint(PyArcStringInputs<'py>, PySphericalPointInputs<'py>),
    }

    #[pymethods]
    impl SphericalPolygon {
        #[new]
        /// an interior point is required because an arcstring divides a sphere into two regions
        fn py_new<'py>(polygon: PySphericalPolygonInputs<'py>) -> PyResult<Self> {
            match polygon {
                PySphericalPolygonInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::MultiSphericalPoint(points) => Self::try_new(
                        ArcString::try_new(points, Some(true)).map_err(PyValueError::new_err)?,
                        None,
                    )
                    .map_err(PyValueError::new_err),
                    AnyGeometry::ArcString(boundary) => {
                        Self::try_new(boundary, None).map_err(PyValueError::new_err)
                    }
                    AnyGeometry::SphericalPolygon(polygon) => Ok(polygon),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive polygon from {geometry:?}"
                    ))),
                },
                PySphericalPolygonInputs::ArcStringInput(boundary) => {
                    Self::try_new(ArcString::py_new(boundary, Some(true))?, None)
                        .map_err(PyValueError::new_err)
                }
                PySphericalPolygonInputs::ArcStringInputWithInteriorPoint(
                    boundary,
                    interior_point,
                ) => Self::try_new(
                    ArcString::py_new(boundary, Some(true))?,
                    Some(SphericalPoint::py_new(interior_point)?),
                )
                .map_err(PyValueError::new_err),
            }
        }

        #[classmethod]
        #[pyo3(name="from_cone", signature=(center, radius, steps=16))]
        fn py_from_cone<'py>(
            _: &Bound<'py, PyType>,
            center: PySphericalPointInputs,
            radius: f64,
            steps: usize,
        ) -> PyResult<Self> {
            Ok(Self::from_cone(
                &SphericalPoint::py_new(center)?,
                &radius,
                steps,
            ))
        }

        #[getter]
        fn get_is_convex(&self) -> bool {
            self.is_convex()
        }

        #[getter]
        fn get_is_clockwise(&self) -> bool {
            self.is_clockwise()
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<ArcString> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiSphericalPolygon)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .union(&multipoint)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::ArcString(arcstring) => self
                    .union(&arcstring)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .union(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiSphericalPolygon)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .union(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multipolygon))
                }
            }
        }

        fn __eq__(&self, other: &Self) -> bool {
            self == other
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use crate::sphericalpolygon::MultiSphericalPolygon;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiSphericalPolygonInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        AnyGeometry(AnyGeometry),
        ListOfPolygons(Vec<PySphericalPolygonInputs<'py>>),
    }

    #[pymethods]
    impl MultiSphericalPolygon {
        #[new]
        fn py_new(polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            let polygons = match polygons {
                PyMultiSphericalPolygonInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::MultiArcString(boundaries) => {
                        let mut polygons = vec![];
                        for boundary in boundaries.arcstrings {
                            polygons.push(
                                SphericalPolygon::try_new(boundary, None)
                                    .map_err(PyValueError::new_err)?,
                            );
                        }
                        polygons
                    }
                    AnyGeometry::MultiSphericalPolygon(polygons) => {
                        return Ok(polygons);
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "cannot derive multipolygon from {geometry:?}",
                        )));
                    }
                },
                PyMultiSphericalPolygonInputs::ListOfPolygons(boundaries) => {
                    let mut polygons: Vec<SphericalPolygon> = vec![];
                    for boundary in boundaries {
                        polygons.push(
                            SphericalPolygon::py_new(boundary).map_err(PyValueError::new_err)?,
                        );
                    }
                    polygons
                }
            };

            MultiSphericalPolygon::try_from(polygons).map_err(PyValueError::new_err)
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<MultiArcString> {
            self.boundary()
        }

        #[getter]
        fn get_representative(&self) -> SphericalPoint {
            self.representative()
        }

        #[getter]
        fn get_centroid(&self) -> SphericalPoint {
            self.centroid()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[pyo3(name = "disjoint")]
        fn py_disjoint(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.disjoint(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.disjoint(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.disjoint(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.disjoint(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.disjoint(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.disjoint(&multipolygon),
            }
        }

        #[pyo3(name = "crosses")]
        fn py_crosses(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.crosses(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.crosses(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.crosses(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.crosses(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.crosses(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.crosses(&multipolygon),
            }
        }

        #[pyo3(name = "within")]
        fn py_within(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.within(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.within(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.within(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.within(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "contains")]
        fn py_contains(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.contains(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.contains(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.contains(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.contains(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.contains(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.contains(&multipolygon),
            }
        }

        #[pyo3(name = "overlaps")]
        fn py_overlaps(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.overlaps(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.overlaps(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.overlaps(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.overlaps(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.overlaps(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.overlaps(&multipolygon),
            }
        }

        #[pyo3(name = "covers")]
        fn py_covers(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.covers(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.covers(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.covers(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.covers(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.covers(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.covers(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.union(&point).map(AnyGeometry::MultiSphericalPolygon)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .union(&multipoint)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::ArcString(arcstring) => self
                    .union(&arcstring)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .union(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::SphericalPolygon(polygon) => {
                    self.union(&polygon).map(AnyGeometry::MultiSphericalPolygon)
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .union(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.distance(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.distance(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    self.intersection(&point).map(AnyGeometry::SphericalPoint)
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "symmetric_difference")]
        fn py_symmetric_difference(&self, other: AnyGeometry) -> AnyGeometry {
            match other {
                AnyGeometry::SphericalPoint(point) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&point))
                }
                AnyGeometry::MultiSphericalPoint(multipoint) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multipoint))
                }
                AnyGeometry::ArcString(arcstring) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&arcstring))
                }
                AnyGeometry::MultiArcString(multiarcstring) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multiarcstring))
                }
                AnyGeometry::SphericalPolygon(polygon) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&polygon))
                }
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    AnyGeometry::MultiSphericalPolygon(self.symmetric_difference(&multipolygon))
                }
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<SphericalPolygon> {
            self.polygons.to_owned()
        }

        /// number of polygons in this collection
        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: usize) -> PyResult<SphericalPolygon> {
            self.polygons
                .get(index)
                .map(|polygon| polygon.to_owned())
                .ok_or(PyKeyError::new_err(format!(
                    "invalid polygon index for list of length {}",
                    self.polygons.len()
                )))
        }

        #[pyo3(name = "append")]
        fn py_append(&mut self, polygon: PySphericalPolygonInputs) -> PyResult<()> {
            self.push(SphericalPolygon::py_new(polygon)?);
            Ok(())
        }

        #[pyo3(name = "extend")]
        fn py_extend(&mut self, polygons: PyMultiSphericalPolygonInputs) -> PyResult<()> {
            self.extend(Self::py_new(polygons)?);
            Ok(())
        }

        fn __iadd__(&mut self, polygons: PyMultiSphericalPolygonInputs) -> PyResult<()> {
            *self += &Self::py_new(polygons)?;
            Ok(())
        }

        fn __add__(&self, polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            Ok(self + &Self::py_new(polygons)?)
        }

        fn __eq__(&self, other: PyMultiSphericalPolygonInputs) -> PyResult<bool> {
            Ok(self == &Self::py_new(other)?)
        }

        fn __str__(&self) -> String {
            self.to_string()
        }

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }
}
