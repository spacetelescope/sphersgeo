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
    use pyo3::{exceptions::PyValueError, types::PyType};

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
                        "cannot derive vector point from {geometry:?}",
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

        #[pyo3(name = "to_lonlat")]
        /// convert this point on the sphere to angular coordinates
        fn py_to_lonlat(&self) -> [f64; 2] {
            self.to_lonlat()
        }

        #[pyo3(name = "two_arc_angle")]
        /// angle on the sphere between this point and two other points
        fn py_two_arc_angle(
            &self,
            a: PySphericalPointInputs,
            b: PySphericalPointInputs,
        ) -> PyResult<f64> {
            Ok(self.two_arc_angle(&Self::py_new(a)?, &Self::py_new(b)?))
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

        #[pyo3(name = "interpolate_between", signature=(other, n=16))]
        /// create n number of points equally spaced on an arc between this point and another point
        fn py_interpolate_between(
            &self,
            other: PySphericalPointInputs,
            n: usize,
        ) -> PyResult<MultiSphericalPoint> {
            self.interpolate_between(&Self::py_new(other)?, n)
                .map_err(PyValueError::new_err)
        }

        #[getter]
        /// length of the underlying xyz vector
        fn get_vector_length(&self) -> f64 {
            self.vector_length()
        }

        #[pyo3(name = "vector_dot")]
        /// dot product of this xyz vector with another xyz vector
        fn py_vector_dot(&self, other: PySphericalPointInputs) -> PyResult<f64> {
            Ok(self.vector_dot(&Self::py_new(other)?))
        }

        #[pyo3(name = "vector_cross")]
        /// cross product of this xyz vector with another xyz vector
        fn py_vector_cross(&self, other: PySphericalPointInputs) -> PyResult<Self> {
            Ok(self.vector_cross(&Self::py_new(other)?))
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
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<SphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
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
                    AnyGeometry::SphericalPoint(point) => Ok(point.vertices()),
                    AnyGeometry::MultiSphericalPoint(multipoint) => Ok(multipoint),
                    AnyGeometry::ArcString(arcstring) => Ok(arcstring.vertices()),
                    AnyGeometry::MultiArcString(multiarcstring) => Ok(multiarcstring.vertices()),
                    AnyGeometry::SphericalPolygon(polygon) => Ok(polygon.vertices()),
                    AnyGeometry::MultiSphericalPolygon(multipolygon) => Ok(multipolygon.vertices()),
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

        #[pyo3(name = "to_lonlats")]
        /// convert to angle coordinates along the sphere
        fn py_to_lonlats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            Array2::<f64>::from(self.to_lonlats()).into_pyarray(py)
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
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[pyo3(name = "distance")]
        /// closest angular distance on the sphere between this geometry and another
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

        fn __concat__(&self, other: PyMultiSphericalPointInputs) -> PyResult<Self> {
            Ok(self + &Self::py_new(other)?)
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
            let mut instance = match arcstring {
                PyArcStringInputs::MultiPointInput(points) => {
                    Self::try_new(MultiSphericalPoint::py_new(points)?, closed)
                        .map_err(PyValueError::new_err)
                }
                PyArcStringInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::MultiSphericalPoint(multipoint) => {
                        ArcString::try_from(multipoint).map_err(PyValueError::new_err)
                    }
                    AnyGeometry::ArcString(arcstring) => Ok(arcstring),
                    AnyGeometry::SphericalPolygon(polygon) => Ok(polygon.boundary),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive arcstring from {geometry:?}",
                    ))),
                },
            }?;

            if let Some(is_closed) = closed {
                if is_closed != instance.closed {
                    instance.closed = is_closed;
                }
            }

            Ok(instance)
        }

        /// number of arcs in this string
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

        #[setter]
        /// "close" this arcstring (connect the last vertex to the first)
        fn set_closed(&mut self, closed: bool) {
            self.closed = closed
        }

        #[getter]
        /// whether this arcstring is "closed" (the last vertex is connected to the first)
        fn get_closed(&self) -> bool {
            self.closed
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
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[pyo3(name = "distance")]
        /// closest angular distance on the sphere between this geometry and another
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
                PyMultiArcStringInputs::ListOfArcStrings(arcstring_inputs) => {
                    let mut arcstrings = vec![];
                    for arcstring_input in arcstring_inputs {
                        arcstrings.push(ArcString::py_new(arcstring_input, None)?);
                    }
                    Self::try_from(arcstrings).map_err(PyValueError::new_err)
                }
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
            }
        }

        #[getter]
        /// radians subtended by each arcstring on the sphere
        fn get_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            Array1::<f64>::from(self.lengths()).into_pyarray(py)
        }

        #[getter]
        /// midpoints of each arc
        fn get_midpoints(&self) -> MultiSphericalPoint {
            self.midpoints()
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[pyo3(name = "distance")]
        /// closest angular distance on the sphere between this geometry and another
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

        fn __len__(&self) -> usize {
            self.len()
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

    #[pymethods]
    impl SphericalPolygon {
        #[new]
        #[pyo3(signature=(boundary, interior_point=None))]
        /// an interior point is required because an arcstring divides a sphere into two regions
        fn py_new<'py>(
            boundary: PyArcStringInputs<'py>,
            interior_point: Option<PySphericalPointInputs<'py>>,
        ) -> PyResult<Self> {
            let boundary = ArcString::py_new(boundary, Some(true))?;
            let interior_point = if let Some(interior_point) = interior_point {
                Some(SphericalPoint::py_new(interior_point)?)
            } else {
                None
            };
            Self::try_new(boundary, interior_point).map_err(PyValueError::new_err)
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
        fn get_is_clockwise(&self) -> bool {
            self.is_clockwise()
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<ArcString> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[pyo3(name = "distance")]
        /// closest angular distance on the sphere between this geometry and another
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
        ListOfArcStrings(Vec<PyArcStringInputs<'py>>),
        ListOfPolygons(Vec<SphericalPolygon>),
    }

    #[pymethods]
    impl MultiSphericalPolygon {
        #[new]
        fn py_new(polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            let polygons = match polygons {
                PyMultiSphericalPolygonInputs::ListOfArcStrings(arcstrings) => {
                    let mut polygons: Vec<SphericalPolygon> = vec![];
                    for arcstring in arcstrings {
                        polygons.push(
                            SphericalPolygon::py_new(arcstring, None)
                                .map_err(PyValueError::new_err)?,
                        );
                    }
                    polygons
                }
                PyMultiSphericalPolygonInputs::ListOfPolygons(polygons) => polygons,
                PyMultiSphericalPolygonInputs::AnyGeometry(geometry) => match geometry {
                    AnyGeometry::SphericalPolygon(polygon) => vec![polygon],
                    AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                        return Ok(multipolygon);
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "cannot derive multi-sphericalpolygon from {geometry:?}",
                        )));
                    }
                },
            };

            MultiSphericalPolygon::try_from(polygons).map_err(PyValueError::new_err)
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        fn get_length(&self) -> f64 {
            self.length()
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
        fn get_boundary(&self) -> Option<MultiArcString> {
            self.boundary()
        }

        #[getter]
        fn get_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
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

        fn __len__(&self) -> usize {
            self.len()
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

    #[pymodule(name = "array")]
    pub mod py_array {
        use super::*;

        #[pyfunction]
        #[pyo3(name = "arc_distance_over_sphere_radians")]
        /// radians subtended between two points on the sphere
        ///
        /// Notes
        /// -----
        /// The length is computed using the following:
        ///
        ///     l = arccos(A ⋅ B) / r^2
        ///
        /// References
        /// ----------
        /// - https://www.mathforengineers.com/math-calculators/angle-between-two-vectors-in-spherical-coordinates.html
        fn py_arc_distance_over_sphere_radians(arc: ([f64; 3], [f64; 3])) -> f64 {
            crate::sphericalpoint::xyzs_distance_over_sphere_radians(&arc.0, &arc.1)
        }

        #[pyfunction]
        #[pyo3(name="interpolate_points_along_arc", signature=(arc, n=50))]
        fn py_interpolate_points_along_arc(
            arc: ([f64; 3], [f64; 3]),
            n: usize,
        ) -> PyResult<Vec<[f64; 3]>> {
            crate::arcstring::interpolate_points_along_arc((&arc.0, &arc.1), n)
                .map_err(PyValueError::new_err)
        }

        #[pyfunction]
        #[pyo3(name = "xyz_two_arc_angle_radians")]
        /// given three XYZ vector points on the sphere (`a`, `b`, and `c`), retrieve the angle at `b` formed by arcs `ab` and `bc`
        ///
        ///     cos(ca) = cos(bc) * cos(ab) + sin(bc) * sin(ab) * cos(b)
        ///
        /// References:
        /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. p132. 1994. Academic Press. doi:10.5555/180895.180907
        ///   `pdf <https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover>`_
        fn py_xyz_two_arc_angle_radians(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
            crate::sphericalpoint::xyz_two_arc_angle_radians(&a, &b, &c)
        }

        #[pyfunction]
        #[pyo3(name = "spherical_triangle_area_steradians")]
        /// surface area of a triangle on the sphere via Girard's theorum
        ///
        ///     θ_1 + θ_2 + θ_3 − π
        ///
        /// References
        /// ----------
        /// - Klain, D. A. (2019). A probabilistic proof of the spherical excess formula (No. arXiv:1909.04505). arXiv. https://doi.org/10.48550/arXiv.1909.04505
        /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
        ///   `pdf <https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover>`_
        fn py_spherical_triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
            // xyz_two_arc_angle_radians(c, a, b)
            //     + xyz_two_arc_angle_radians(a, b, c)
            //     + xyz_two_arc_angle_radians(b, c, a)
            //     - std::f64::consts::PI

            // redefine Girard's theorum to avoid domain errors
            let ab = crate::sphericalpoint::xyzs_distance_over_sphere_radians(&a, &b);
            let bc = crate::sphericalpoint::xyzs_distance_over_sphere_radians(&b, &c);
            let ca = crate::sphericalpoint::xyzs_distance_over_sphere_radians(&c, &a);
            let s = (ab + bc + ca) / 2.0;

            4.0 * ((s / 2.0).tan()
                * ((s - ab) / 2.0).tan()
                * ((s - bc) / 2.0).tan()
                * ((s - ca) / 2.0).tan())
            .sqrt()
            .atan()
        }
    }
}
