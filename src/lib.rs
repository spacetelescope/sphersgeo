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
    use super::*;
    use crate::geometry::{
        AnyGeometry, GeometricOperations, GeometricRelationships, Geometry, GeometryCollection,
        MultiGeometry, MultiGeometryUnaryOperations,
    };
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
        ndarray::{Array1, Array2},
    };
    use pyo3::{
        exceptions::{PyIndexError, PyRuntimeError, PyValueError},
        types::{PySlice, PyType},
    };

    #[derive(FromPyObject)]
    enum PyIndex<'py> {
        Int(isize),
        Slice(Bound<'py, PySlice>),
    }

    #[pymodule_export]
    use crate::sphericalpoint::SphericalPoint;

    #[derive(IntoPyObject)]
    enum PySingleOrMultiPoint {
        Single(SphericalPoint),
        Multi(MultiSphericalPoint),
    }

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PySphericalPointInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        Geometry(AnyGeometry),
        XYZArray([f64; 3]),
        XYZTuple((f64, f64, f64)),
        LonLatArray([f64; 2]),
        LonLatTuple((f64, f64)),
        NumpyArray(PyReadonlyArray1<'py, f64>),
        List(Vec<f64>),
        WellKnownText(String),
    }

    #[pymethods]
    impl SphericalPoint {
        #[new]
        fn py_new(point: PySphericalPointInputs) -> PyResult<Self> {
            match point {
                PySphericalPointInputs::Geometry(geometry) => match geometry {
                    AnyGeometry::SphericalPoint(point) => Ok(point),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive point from {geometry:?}",
                    ))),
                },
                PySphericalPointInputs::XYZArray(xyz) => Ok(Self::from(xyz)),
                PySphericalPointInputs::XYZTuple(xyz) => Ok(Self::from(&xyz)),
                PySphericalPointInputs::LonLatArray(lonlat) => Ok(Self::from(&lonlat)),
                PySphericalPointInputs::LonLatTuple(lonlat) => {
                    Ok(Self::from(&[lonlat.0, lonlat.1]))
                }
                PySphericalPointInputs::NumpyArray(xyz) => {
                    Self::try_from(&xyz.as_array()).map_err(PyValueError::new_err)
                }
                PySphericalPointInputs::List(xyz) => {
                    Self::try_from(&xyz).map_err(PyValueError::new_err)
                }
                PySphericalPointInputs::WellKnownText(wkt) => {
                    Self::py_new(PySphericalPointInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ))
                }
            }
        }

        #[getter]
        fn get_xyz(&self) -> [f64; 3] {
            self.xyz
        }

        #[getter]
        fn get_lonlat(&self) -> [f64; 2] {
            self.into()
        }

        #[getter]
        fn get_antipode(&self) -> SphericalPoint {
            self.antipode()
        }

        #[pyo3(name = "two_arc_angle")]
        fn py_two_arc_angle(
            &self,
            start: PySphericalPointInputs,
            end: PySphericalPointInputs,
        ) -> PyResult<f64> {
            Ok(self.two_arc_angle(&Self::py_new(start)?, &Self::py_new(end)?))
        }

        #[pyo3(name = "colinear")]
        fn py_colinear(
            &self,
            a: PySphericalPointInputs,
            b: PySphericalPointInputs,
        ) -> PyResult<bool> {
            Ok(self.colinear(&Self::py_new(a)?, &Self::py_new(b)?))
        }

        #[pyo3(name = "is_clockwise_turn")]
        fn py_is_clockwise_turn(
            &self,
            start: PySphericalPointInputs,
            end: PySphericalPointInputs,
        ) -> PyResult<bool> {
            Ok(self.is_clockwise_turn(&Self::py_new(start)?, &Self::py_new(end)?))
        }

        #[pyo3(name = "interpolate_points")]
        fn py_interpolate_points(
            &self,
            end: PySphericalPointInputs,
            n: usize,
        ) -> PyResult<MultiSphericalPoint> {
            self.interpolate_points(&Self::py_new(end)?, n)
                .map_err(PyValueError::new_err)
        }

        #[getter]
        fn get_vector_length(&self) -> f64 {
            self.vector_length()
        }

        #[pyo3(name = "vector_cross")]
        fn py_vector_cross(&self, other: PySphericalPointInputs) -> PyResult<Self> {
            Ok(self.vector_cross(&Self::py_new(other)?))
        }

        #[pyo3(name = "vector_dot")]
        fn py_vector_dot(&self, other: PySphericalPointInputs) -> PyResult<f64> {
            Ok(self.vector_dot(&Self::py_new(other)?))
        }

        #[pyo3(name = "vector_rotate_around")]
        fn py_vector_rotate_around(
            &self,
            other: PySphericalPointInputs,
            theta: f64,
        ) -> PyResult<Self> {
            Ok(self.vector_rotate_around(&Self::py_new(other)?, &theta))
        }

        #[pyo3(name = "to")]
        fn py_to(&self, other: PySphericalPointInputs) -> PyResult<ArcString> {
            Ok(self.to(&Self::py_new(other)?))
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<Self> {
            None
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<MultiSphericalPoint> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
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

        fn __repr__(&self) -> String {
            self.to_string()
        }
    }

    #[pymodule_export]
    use crate::sphericalpoint::MultiSphericalPoint;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiSphericalPointInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        Geometry(AnyGeometry),
        ListOfPoints(Vec<SphericalPoint>),
        ListOfXYZArrays(Vec<[f64; 3]>),
        ListOfXYZTuples(Vec<(f64, f64, f64)>),
        ListOfLonLatArrays(Vec<[f64; 2]>),
        ListOfLonLatTuples(Vec<(f64, f64)>),
        NumpyArray(PyReadonlyArray2<'py, f64>),
        NestedList(Vec<Vec<f64>>),
        WellKnownText(String),
    }

    #[pymethods]
    impl MultiSphericalPoint {
        #[new]
        fn py_new(points: PyMultiSphericalPointInputs) -> PyResult<Self> {
            match points {
                PyMultiSphericalPointInputs::Geometry(geometry) => match geometry {
                    AnyGeometry::MultiSphericalPoint(multipoint) => Ok(multipoint),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive multipoint from {geometry:?}",
                    ))),
                },
                PyMultiSphericalPointInputs::ListOfPoints(points) => {
                    Self::try_from(points).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfXYZArrays(xyzs) => {
                    Self::try_from(xyzs).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfXYZTuples(xyzs) => {
                    Self::try_from(&xyzs).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfLonLatArrays(lonlats) => {
                    Self::try_from(&lonlats).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::ListOfLonLatTuples(lonlats) => {
                    Self::try_from(&lonlats).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::NumpyArray(xyzs) => {
                    Self::try_from(xyzs.as_array().to_owned()).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::NestedList(xyzs) => {
                    Self::try_from(&xyzs).map_err(PyValueError::new_err)
                }
                PyMultiSphericalPointInputs::WellKnownText(wkt) => {
                    Self::py_new(PyMultiSphericalPointInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ))
                }
            }
        }

        #[getter]
        fn get_xyzs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            Array2::<f64>::from(self).into_pyarray(py)
        }

        #[getter]
        fn get_lonlats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            let lonlats: Vec<[f64; 2]> = self.into();
            Array2::<f64>::from(lonlats).into_pyarray(py)
        }

        #[pyo3(name = "nearest")]
        fn py_nearest(&self, other: PySphericalPointInputs) -> PyResult<(SphericalPoint, f64)> {
            Ok(self.nearest(&SphericalPoint::py_new(other)?))
        }

        #[getter]
        fn get_vectors_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            Array1::<f64>::from(self.vectors_lengths()).into_pyarray(py)
        }

        #[getter]
        fn get_vertices(&self) -> MultiSphericalPoint {
            self.vertices()
        }

        #[getter]
        fn get_boundary(&self) -> Option<Self> {
            None
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<MultiSphericalPoint> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<SphericalPoint> {
            self.to_owned().into()
        }

        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: PyIndex) -> PyResult<Option<PySingleOrMultiPoint>> {
            let length = self.xyzs.len() as isize;
            match index {
                PyIndex::Int(index) => {
                    // wrap negative index
                    let index = if index < 0 { index + length } else { index };

                    if index < length {
                        Ok(Some(PySingleOrMultiPoint::Single(SphericalPoint {
                            xyz: self.xyzs[index as usize],
                        })))
                    } else {
                        Err(PyIndexError::new_err(String::from("index out of range")))
                    }
                }
                PyIndex::Slice(slice) => {
                    let indices = slice.indices(length).map_err(PyIndexError::new_err)?;

                    Ok(if indices.slicelength > 0 {
                        Some(PySingleOrMultiPoint::Multi(
                            if indices.slicelength as isize == length {
                                self.to_owned()
                            } else {
                                let mut xyzs = vec![];
                                let mut index = indices.start;
                                while index < indices.stop {
                                    xyzs.push(self.xyzs[index as usize]);
                                    index += indices.step;
                                }

                                Self::try_from(xyzs).map_err(PyRuntimeError::new_err)?
                            },
                        ))
                    } else {
                        None
                    })
                }
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

        #[getter]
        fn get_unary_union(&self) -> Self {
            self.unary_union()
        }

        #[getter]
        fn get_unary_intersection(&self) -> Option<Self> {
            self.unary_intersection()
        }

        #[getter]
        fn get_unary_symmetric_difference(&self) -> Option<Self> {
            self.unary_symmetric_difference()
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
            format!("SphericalPoint({:?})", self.xyzs)
        }
    }

    #[pymodule_export]
    use crate::arcstring::ArcString;

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyArcStringInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        Geometry(AnyGeometry),
        MultiPointInput(PyMultiSphericalPointInputs<'py>),
        WellKnownText(String),
    }

    #[pymethods]
    impl ArcString {
        #[new]
        #[pyo3(signature=(arcstring, closed=None))]
        fn py_new(arcstring: PyArcStringInputs, closed: Option<bool>) -> PyResult<Self> {
            match arcstring {
                PyArcStringInputs::Geometry(geometry) => {
                    let mut instance = match geometry {
                        AnyGeometry::MultiSphericalPoint(multipoint) => {
                            ArcString::try_from(multipoint).map_err(PyValueError::new_err)
                        }
                        AnyGeometry::ArcString(arcstring) => Ok(arcstring),
                        _ => Err(PyValueError::new_err(format!(
                            "cannot derive arcstring from {geometry:?}",
                        ))),
                    }?;

                    if let Some(is_closed) = closed
                        && is_closed != instance.closed
                    {
                        instance.closed = is_closed;
                    }

                    Ok(instance)
                }
                PyArcStringInputs::MultiPointInput(points) => {
                    Self::try_new(MultiSphericalPoint::py_new(points)?, closed)
                        .map_err(PyValueError::new_err)
                }
                PyArcStringInputs::WellKnownText(wkt) => Self::py_new(
                    PyArcStringInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ),
                    closed,
                ),
            }
        }

        fn __len__(&self) -> usize {
            self.points.len() - 1
        }

        #[getter]
        fn get_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            Array1::<f64>::from(self.lengths()).into_pyarray(py)
        }

        #[getter]
        fn get_midpoints(&self) -> MultiSphericalPoint {
            self.midpoints()
        }

        #[getter]
        fn get_arcs(&self) -> Vec<ArcString> {
            self.arcs()
        }

        #[getter]
        fn get_closed(&self) -> bool {
            self.closed
        }

        #[setter]
        fn set_closed(&mut self, closed: bool) {
            self.closed = closed
        }

        #[getter]
        fn get_crosses_self(&self) -> bool {
            self.crosses_self()
        }

        #[getter]
        fn get_crossings_with_self(&self) -> Option<MultiSphericalPoint> {
            self.crossings_with_self()
        }

        #[pyo3(name = "adjoins")]
        fn py_adjoins(&self, other: PyArcStringInputs) -> PyResult<bool> {
            Ok(self.adjoins(&Self::py_new(other, None)?))
        }

        #[pyo3(name = "simplify")]
        fn py_simplify(&mut self) {
            self.simplify()
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<MultiArcString> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
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
    use crate::arcstring::MultiArcString;

    #[derive(IntoPyObject)]
    enum PySingleOrMultiArcString {
        Single(ArcString),
        Multi(MultiArcString),
    }

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiArcStringInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        Geometry(AnyGeometry),
        ListOfArcStrings(Vec<PyArcStringInputs<'py>>),
        WellKnownText(String),
    }

    #[pymethods]
    impl MultiArcString {
        #[new]
        fn py_new(arcstrings: PyMultiArcStringInputs) -> PyResult<Self> {
            match arcstrings {
                PyMultiArcStringInputs::Geometry(geometry) => match geometry {
                    AnyGeometry::MultiSphericalPoint(multipoint) => Self::try_from(vec![
                        ArcString::try_from(multipoint).map_err(PyValueError::new_err)?,
                    ])
                    .map_err(PyValueError::new_err),
                    AnyGeometry::ArcString(arcstring) => {
                        Self::try_from(vec![arcstring]).map_err(PyValueError::new_err)
                    }
                    AnyGeometry::MultiArcString(multiarcstring) => Ok(multiarcstring),
                    _ => Err(PyValueError::new_err(format!(
                        "cannot derive multiarcstring from {geometry:?}"
                    ))),
                },
                PyMultiArcStringInputs::ListOfArcStrings(arcstring_inputs) => {
                    let mut arcstrings = vec![];
                    for arcstring_input in arcstring_inputs {
                        arcstrings.push(ArcString::py_new(arcstring_input, None)?);
                    }
                    Self::try_from(arcstrings).map_err(PyValueError::new_err)
                }
                PyMultiArcStringInputs::WellKnownText(wkt) => {
                    Self::py_new(PyMultiArcStringInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ))
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<Self> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<ArcString> {
            self.arcstrings.to_owned()
        }

        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: PyIndex) -> PyResult<Option<PySingleOrMultiArcString>> {
            let length = self.arcstrings.len() as isize;
            match index {
                PyIndex::Int(index) => {
                    let length = length as isize;
                    // wrap negative index
                    let index = if index < 0 { index + length } else { index };

                    if index < length {
                        Ok(Some(PySingleOrMultiArcString::Single(
                            self.arcstrings[index as usize].to_owned(),
                        )))
                    } else {
                        Err(PyIndexError::new_err(String::from("index out of range")))
                    }
                }
                PyIndex::Slice(slice) => {
                    let indices = slice.indices(length).map_err(PyIndexError::new_err)?;

                    Ok(if indices.slicelength > 0 {
                        Some(PySingleOrMultiArcString::Multi(
                            if indices.slicelength as isize == length {
                                self.to_owned()
                            } else {
                                let mut arcstrings = vec![];
                                let mut index = indices.start;
                                while index < indices.stop {
                                    arcstrings.push(self.arcstrings[index as usize].to_owned());
                                    index += indices.step;
                                }

                                Self::try_from(arcstrings).map_err(PyRuntimeError::new_err)?
                            },
                        ))
                    } else {
                        None
                    })
                }
            }
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

        #[getter]
        fn get_unary_union(&self) -> Self {
            self.unary_union()
        }

        #[getter]
        fn get_unary_intersection(&self) -> Option<Self> {
            self.unary_intersection()
        }

        #[getter]
        fn get_unary_symmetric_difference(&self) -> Option<Self> {
            self.unary_symmetric_difference()
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
        Geometry(AnyGeometry),
        ArcStringInput(PyArcStringInputs<'py>),
        ArcStringInputWithInteriorPoint(PyArcStringInputs<'py>, PySphericalPointInputs<'py>),
        WellKnownText(String),
    }

    #[pymethods]
    impl SphericalPolygon {
        #[new]
        fn py_new<'py>(polygon: PySphericalPolygonInputs<'py>) -> PyResult<Self> {
            match polygon {
                PySphericalPolygonInputs::Geometry(geometry) => match geometry {
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
                PySphericalPolygonInputs::WellKnownText(wkt) => {
                    Self::py_new(PySphericalPolygonInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ))
                }
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

        #[pyo3(name = "simplify")]
        fn py_simplify(&mut self) {
            self.simplify()
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<MultiSphericalPolygon> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
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

    #[derive(IntoPyObject)]
    enum PySingleOrMultiPolygon {
        Single(SphericalPolygon),
        Multi(MultiSphericalPolygon),
    }

    #[derive(FromPyObject)]
    #[allow(clippy::large_enum_variant)]
    enum PyMultiSphericalPolygonInputs<'py> {
        // NOTE: AnyGeometry MUST be the first option in this enum, otherwise it will attempt to match another pattern
        Geometry(AnyGeometry),
        ListOfPolygons(Vec<PySphericalPolygonInputs<'py>>),
        WellKnownText(String),
    }

    #[pymethods]
    impl MultiSphericalPolygon {
        #[new]
        fn py_new(polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            let polygons = match polygons {
                PyMultiSphericalPolygonInputs::Geometry(geometry) => match geometry {
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
                PyMultiSphericalPolygonInputs::WellKnownText(wkt) => {
                    return Self::py_new(PyMultiSphericalPolygonInputs::Geometry(
                        crate::geometry::try_from_wkt(wkt.as_str())
                            .map_err(PyValueError::new_err)?,
                    ));
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

        #[getter]
        fn get_wkt(&self) -> String {
            self.to_wkt(true)
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

        #[pyo3(name = "equals")]
        fn py_equals(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.equals(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.equals(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.equals(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.equals(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.equals(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.equals(&multipolygon),
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

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersection(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "difference")]
        fn py_difference(&self, other: AnyGeometry) -> Option<Self> {
            match other {
                AnyGeometry::SphericalPoint(point) => self.difference(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.difference(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.difference(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.difference(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.difference(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.difference(&multipolygon),
            }
        }

        #[pyo3(name = "union")]
        fn py_union(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::SphericalPoint(point) => self.union(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.union(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.union(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.union(&multiarcstring),
                AnyGeometry::SphericalPolygon(polygon) => self.union(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.union(&multipolygon),
            }
        }

        #[getter]
        fn get_parts(&self) -> Vec<SphericalPolygon> {
            self.polygons.to_owned()
        }

        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: PyIndex) -> PyResult<Option<PySingleOrMultiPolygon>> {
            let length = self.polygons.len() as isize;
            match index {
                PyIndex::Int(index) => {
                    let length = length as isize;
                    // wrap negative index
                    let index = if index < 0 { index + length } else { index };

                    if index < length {
                        Ok(Some(PySingleOrMultiPolygon::Single(
                            self.polygons[index as usize].to_owned(),
                        )))
                    } else {
                        Err(PyIndexError::new_err(String::from("index out of range")))
                    }
                }
                PyIndex::Slice(slice) => {
                    let indices = slice.indices(length).map_err(PyIndexError::new_err)?;

                    Ok(if indices.slicelength > 0 {
                        Some(PySingleOrMultiPolygon::Multi(
                            if indices.slicelength as isize == length {
                                self.to_owned()
                            } else {
                                let mut polygons = vec![];
                                let mut index = indices.start;
                                while index < indices.stop {
                                    polygons.push(self.polygons[index as usize].to_owned());
                                    index += indices.step;
                                }

                                Self::try_from(polygons).map_err(PyRuntimeError::new_err)?
                            },
                        ))
                    } else {
                        None
                    })
                }
            }
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

        #[getter]
        fn get_unary_union(&self) -> Self {
            self.unary_union()
        }

        #[getter]
        fn get_unary_intersection(&self) -> Option<Self> {
            self.unary_intersection()
        }

        #[getter]
        fn get_unary_symmetric_difference(&self) -> Option<Self> {
            self.unary_symmetric_difference()
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

    #[pyfunction]
    fn from_wkt(wkt: &str) -> PyResult<AnyGeometry> {
        let wkt = wkt.trim();
        crate::geometry::try_from_wkt(wkt)
            .map_err(|err| PyValueError::new_err(format!("{err} when parsing `{wkt}`")))
    }
}
