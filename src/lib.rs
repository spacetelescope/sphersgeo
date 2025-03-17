#![allow(unused_variables, refining_impl_trait)]
mod angularbounds;
mod arcstring;
mod geometry;
mod geometrycollection;
mod sphericalgraph;
mod sphericalpoint;
mod sphericalpolygon;

extern crate numpy;

use pyo3::prelude::*;

#[pymodule(name = "sphersgeo")]
mod py_sphersgeo {
    use super::*;
    use crate::geometry::{
        AnyGeometry, ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
    };
    use numpy::{
        ndarray::{array, s, Array, Array2, Axis},
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
    };
    use pyo3::{exceptions::PyValueError, types::PyType};

    #[pymodule_export]
    use super::sphericalpoint::SphericalPoint;

    #[derive(FromPyObject)]
    enum PySphericalPointInputs<'py> {
        NumpyArray(PyReadonlyArray1<'py, f64>),
        Tuple((f64, f64, f64)),
        List(Vec<f64>),
        Point(SphericalPoint),
    }

    #[derive(FromPyObject)]
    enum PySphericalPointLonLatInputs<'py> {
        NumpyArray(PyReadonlyArray1<'py, f64>),
        Tuple((f64, f64)),
        List(Vec<f64>),
    }

    #[pymethods]
    impl SphericalPoint {
        #[new]
        fn py_new(point: PySphericalPointInputs) -> PyResult<Self> {
            let xyz = match point {
                PySphericalPointInputs::NumpyArray(xyz) => xyz.as_array().to_owned(),
                PySphericalPointInputs::Tuple((x, y, z)) => array![x, y, z],
                PySphericalPointInputs::List(list) => Array::from_vec(list),
                PySphericalPointInputs::Point(point) => {
                    return Ok(point.to_owned());
                }
            };

            Self::try_from(xyz).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        #[classmethod]
        #[pyo3(name = "normalize")]
        fn py_normalize<'py>(
            _: &Bound<'_, PyType>,
            point: PySphericalPointInputs,
        ) -> PyResult<Self> {
            // TODO: normalize vector before passing to constructor
            Ok(Self::py_new(point)?.py_normalized())
        }

        /// from the given coordinates, build an xyz vector representing a point on the sphere
        #[classmethod]
        #[pyo3(name = "from_lonlat", signature=(coordinates, degrees=true))]
        fn py_from_lonlat<'py>(
            _: &Bound<'_, PyType>,
            coordinates: PySphericalPointLonLatInputs,
            degrees: bool,
        ) -> PyResult<Self> {
            let coordinates = match coordinates {
                PySphericalPointLonLatInputs::NumpyArray(coordinates) => {
                    coordinates.as_array().to_owned()
                }
                PySphericalPointLonLatInputs::Tuple((lon, lat)) => array![lon, lat],
                PySphericalPointLonLatInputs::List(list) => Array::from_vec(list),
            };

            match Self::try_from_lonlat(&coordinates.view(), degrees) {
                Ok(result) => Ok(result),
                Err(err) => Err(PyValueError::new_err(err)),
            }
        }

        /// xyz vector as a 1-dimensional array of 3 floats
        #[getter]
        #[pyo3(name = "xyz")]
        fn py_xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.xyz.to_pyarray(py)
        }

        /// convert this point on the sphere to angular coordinates
        #[pyo3(name = "to_lonlat", signature=(degrees=true))]
        fn py_to_lonlat<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray1<f64>> {
            self.to_lonlat(degrees).into_pyarray(py)
        }

        /// normalize this vector to length 1 (the unit sphere) while preserving direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// angle on the sphere between this point and two other points
        #[pyo3(name = "angle_between", signature=(a, b, degrees=true))]
        fn py_angle_between(&self, a: &SphericalPoint, b: &SphericalPoint, degrees: bool) -> f64 {
            self.angle_between(a, b, degrees)
        }

        /// create n number of points equally spaced on an arc between this point and another point
        #[pyo3(name = "interpolate_between", signature=(other, n=16))]
        fn py_interpolate_between(
            &self,
            other: &SphericalPoint,
            n: usize,
        ) -> PyResult<MultiSphericalPoint> {
            self.interpolate_between(other, n)
                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        /// whether this point lies exactly between the given points
        #[pyo3(name = "collinear")]
        fn py_collinear(&self, a: &SphericalPoint, b: &SphericalPoint) -> bool {
            self.collinear(a, b)
        }

        /// length of the underlying xyz vector
        #[getter]
        #[pyo3(name = "vector_length")]
        fn py_vector_length(&self) -> f64 {
            self.vector_length()
        }

        /// rotate this xyz vector by theta angle around another xyz vector
        #[pyo3(name = "vector_rotate_around", signature=(other, theta, degrees=true))]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, &theta, degrees)
        }

        #[pyo3(name = "combine")]
        fn py_combine(&self, other: &Self) -> MultiSphericalPoint {
            self.combine(other)
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<SphericalPoint> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::SphericalPoint),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        fn __add__(&self, other: &Self) -> MultiSphericalPoint {
            self + other
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.xyz == other.xyz
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
    enum PyMultiSphericalPointInputs<'py> {
        NumpyArray(PyReadonlyArray2<'py, f64>),
        ListOfTuples(Vec<(f64, f64, f64)>),
        NestedList(Vec<Vec<f64>>),
        FlatList(Vec<f64>),
        PointList(Vec<SphericalPoint>),
        // TODO: fix error with collapse_axis: Index 4 must be less than axis length 4 for array with shape [4, 3]
        // MultiPoint(MultiSphericalPoint),
    }

    #[derive(FromPyObject)]
    enum PyMultiSphericalPointLonLatInputs<'py> {
        NumpyArray(PyReadonlyArray2<'py, f64>),
        ListOfTuples(Vec<(f64, f64)>),
        NestedList(Vec<Vec<f64>>),
        FlatList(Vec<f64>),
    }

    #[pymethods]
    impl MultiSphericalPoint {
        #[new]
        fn py_new<'py>(points: PyMultiSphericalPointInputs) -> PyResult<Self> {
            match points {
                PyMultiSphericalPointInputs::NumpyArray(xyz) => {
                    Self::try_from(xyz.as_array().to_owned())
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiSphericalPointInputs::ListOfTuples(list) => Ok(Self::from(&list)),
                PyMultiSphericalPointInputs::NestedList(list) => {
                    Self::try_from(&list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiSphericalPointInputs::FlatList(list) => {
                    Self::try_from(list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiSphericalPointInputs::PointList(list) => {
                    Self::try_from(&list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                } // TODO: fix error with collapse_axis: Index 4 must be less than axis length 4 for array with shape [4, 3]
                  // PyMultiSphericalPointInputs::MultiPoint(multipoint) => Ok(multipoint.to_owned()),
            }
        }

        #[classmethod]
        #[pyo3(name = "normalize")]
        fn py_normalize<'py>(
            _: &Bound<'_, PyType>,
            points: PyMultiSphericalPointInputs,
        ) -> PyResult<Self> {
            // TODO: normalize vectors before passing to constructor
            Ok(Self::py_new(points)?.py_normalized())
        }

        /// from the given coordinates, build xyz vectors representing points on the sphere
        #[classmethod]
        #[pyo3(name = "from_lonlats", signature=(coordinates, degrees=true))]
        fn py_from_lonlats<'py>(
            _: &Bound<'_, PyType>,
            coordinates: PyMultiSphericalPointLonLatInputs,
            degrees: bool,
        ) -> PyResult<Self> {
            let coordinates = match coordinates {
                PyMultiSphericalPointLonLatInputs::NumpyArray(coordinates) => {
                    coordinates.as_array().to_owned()
                }
                PyMultiSphericalPointLonLatInputs::ListOfTuples(list) => {
                    let mut xyz = Array2::uninit((list.len(), 2));
                    for (index, tuple) in list.iter().enumerate() {
                        array![tuple.0, tuple.1].assign_to(xyz.index_axis_mut(Axis(0), index));
                    }
                    unsafe { xyz.assume_init() }
                }
                PyMultiSphericalPointLonLatInputs::NestedList(list) => {
                    let mut xyz = Array2::<f64>::default((list.len(), 3));
                    for (i, mut point) in xyz.axis_iter_mut(Axis(0)).enumerate() {
                        for (j, value) in point.iter_mut().enumerate() {
                            *value = list[i][j];
                        }
                    }

                    xyz
                }
                PyMultiSphericalPointLonLatInputs::FlatList(list) => {
                    Array2::from_shape_vec((list.len(), 3), list)
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?
                }
            };

            match Self::try_from_lonlats(&coordinates.view(), degrees) {
                Ok(result) => Ok(result),
                Err(err) => Err(PyValueError::new_err(err)),
            }
        }

        /// xyz vectors as a 2-dimensional array of Nx3 floats
        #[getter]
        #[pyo3(name = "xyz")]
        fn py_xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            self.xyz.to_pyarray(py)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name = "to_lonlats", signature=(degrees=true))]
        fn py_to_lonlats<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray2<f64>> {
            self.to_lonlats(degrees).into_pyarray(py)
        }

        /// normalize the underlying vectors to length 1 (the unit sphere) while preserving direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// rotate the underlying vector by theta angle around other vectors
        #[pyo3(name = "vector_rotate_around", signature=(other, theta, degrees=true))]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        /// lengths of the underlying xyz vectors
        #[getter]
        #[pyo3(name = "vector_lengths")]
        fn py_vector_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.vector_lengths().into_pyarray(py)
        }

        /// angles on the sphere between these points and other sets of points
        #[pyo3(name="angles_between", signature=(a, b, degrees=true))]
        fn py_angles_between<'py>(
            &self,
            py: Python<'py>,
            a: &MultiSphericalPoint,
            b: &MultiSphericalPoint,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            self.angles_between(a, b, degrees).into_pyarray(py)
        }

        /// whether these points share a line with the given points
        #[pyo3(name = "collinear")]
        fn py_collinear<'py>(
            &self,
            py: Python<'py>,
            a: &SphericalPoint,
            b: &SphericalPoint,
        ) -> Bound<'py, PyArray1<bool>> {
            self.collinear(a, b).into_pyarray(py)
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPoint),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[getter]
        #[pyo3(name = "parts")]
        fn py_parts(&self) -> Vec<SphericalPoint> {
            self.into()
        }

        fn __concat__(&self, other: &Self) -> Self {
            self + other
        }

        /// number of points in this collection
        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: usize) -> SphericalPoint {
            SphericalPoint {
                xyz: self.xyz.slice(s![index, ..]).to_owned(),
            }
        }

        #[pyo3(name = "append")]
        fn py_append(&mut self, other: SphericalPoint) {
            self.push(other);
        }

        #[pyo3(name = "extend")]
        fn py_extend(&mut self, other: Self) {
            self.extend(other);
        }

        fn __iadd__(&mut self, other: MultiSphericalPoint) {
            *self += &other
        }

        fn __add__(&self, other: MultiSphericalPoint) -> Self {
            self + &other
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
    use super::arcstring::ArcString;

    #[derive(FromPyObject)]
    enum PyArcStringInputs<'py> {
        MultiPointInput(PyMultiSphericalPointInputs<'py>),
        ArcString(ArcString),
    }

    #[pymethods]
    impl ArcString {
        #[new]
        fn py_new(points: PyArcStringInputs) -> PyResult<Self> {
            let points = match points {
                PyArcStringInputs::MultiPointInput(multipoint_input) => {
                    MultiSphericalPoint::py_new(multipoint_input)?
                }
                PyArcStringInputs::ArcString(arcstring) => {
                    return Ok(arcstring.to_owned());
                }
            };

            Self::try_from(points).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        /// number of arcs in this string
        fn __len__(&self) -> usize {
            self.points.len() - 1
        }

        /// radians subtended by each arc on the sphere
        #[getter]
        #[pyo3(name = "lengths")]
        fn py_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.lengths().into_pyarray(py)
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiSphericalPoint {
            self.midpoints()
        }

        /// "close" this arcstring (connect the last vertex to the first)
        #[pyo3(name = "close")]
        fn py_close(&mut self) {
            self.close()
        }

        /// whether this arcstring is "closed" (the last vertex is connected to the first)
        #[getter]
        #[pyo3(name = "closed")]
        fn py_closed(&self) -> bool {
            self.closed()
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiArcString),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
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
    use super::arcstring::MultiArcString;

    #[derive(FromPyObject)]
    enum PyMultiArcStringInputs<'py> {
        ListOfArcStrings(Vec<PyArcStringInputs<'py>>),
        MultiArcString(MultiArcString),
    }

    #[pymethods]
    impl MultiArcString {
        #[new]
        fn py_new(arcstrings: PyMultiArcStringInputs) -> PyResult<Self> {
            match arcstrings {
                PyMultiArcStringInputs::ListOfArcStrings(arcstring_inputs) => {
                    let mut arcstrings = vec![];
                    for arcstring_input in arcstring_inputs {
                        arcstrings.push(ArcString::py_new(arcstring_input)?);
                    }
                    Self::try_from(arcstrings)
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiArcStringInputs::MultiArcString(multiarcstring) => {
                    Ok(multiarcstring.to_owned())
                }
            }
        }

        /// radians subtended by each arcstring on the sphere
        #[getter]
        #[pyo3(name = "lengths")]
        fn py_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.lengths().into_pyarray(py)
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiSphericalPoint {
            self.midpoints()
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<MultiSphericalPoint> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiArcString),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[getter]
        #[pyo3(name = "parts")]
        fn py_parts(&self) -> Vec<ArcString> {
            self.arcstrings.to_owned().into()
        }

        fn __len__(&self) -> usize {
            self.len()
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
    use crate::angularbounds::AngularBounds;

    #[pymethods]
    impl AngularBounds {
        #[new]
        #[pyo3(signature=(min_x, min_y, max_x, max_y, degrees=true))]
        fn py_new(min_x: f64, min_y: f64, max_x: f64, max_y: f64, degrees: bool) -> Self {
            Self {
                min_x,
                min_y,
                max_x,
                max_y,
                degrees,
            }
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<ArcString> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::AngularBounds),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
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
    use crate::sphericalpolygon::SphericalPolygon;

    #[derive(FromPyObject)]
    enum PySphericalPolygonInputs<'py> {
        Construction {
            exterior: PyArcStringInputs<'py>,
            interior_point: PySphericalPointInputs<'py>,
            holes: Option<PyMultiArcStringInputs<'py>>,
        },
        Polygon(SphericalPolygon),
    }

    #[pymethods]
    impl SphericalPolygon {
        /// an interior point is required because an arcstring divides a sphere into two regions
        #[new]
        fn py_new(polygon: PySphericalPolygonInputs) -> PyResult<Self> {
            match polygon {
                PySphericalPolygonInputs::Construction {
                    exterior,
                    interior_point,
                    holes,
                } => {
                    let exterior = ArcString::py_new(exterior)?;
                    let interior_point = SphericalPoint::py_new(interior_point)?;
                    let holes = if let Some(holes) = holes {
                        Some(MultiArcString::py_new(holes)?)
                    } else {
                        None
                    };
                    Self::new(exterior, interior_point, holes)
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PySphericalPolygonInputs::Polygon(polygon) => Ok(polygon.to_owned()),
            }
        }

        #[classmethod]
        #[pyo3(signature=(center, radius, degrees=true, steps=16))]
        fn py_from_cone(
            _: &Bound<'_, PyType>,
            center: SphericalPoint,
            radius: f64,
            degrees: bool,
            steps: usize,
        ) -> Self {
            Self::from_cone(&center, &radius, degrees, steps)
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<ArcString> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
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
    enum PyMultiSphericalPolygonInputs<'py> {
        ListOfPolygons(Vec<PySphericalPolygonInputs<'py>>),
        MultiPolygon(MultiSphericalPolygon),
    }

    #[pymethods]
    impl MultiSphericalPolygon {
        #[new]
        fn py_new(polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            Ok(match polygons {
                PyMultiSphericalPolygonInputs::ListOfPolygons(polygon_inputs) => {
                    let mut polygons: Vec<SphericalPolygon> = vec![];
                    for polygon_input in polygon_inputs {
                        polygons.push(SphericalPolygon::py_new(polygon_input)?);
                    }
                    Self::try_from(polygons)
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?
                }
                PyMultiSphericalPolygonInputs::MultiPolygon(multipolygon) => {
                    multipolygon.to_owned()
                }
            })
        }

        #[getter]
        #[pyo3(name = "area")]
        fn py_area(&self) -> f64 {
            self.area()
        }

        #[getter]
        #[pyo3(name = "length")]
        fn py_length(&self) -> f64 {
            self.length()
        }

        #[pyo3(name = "bounds", signature=(degrees=true))]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "coords")]
        fn py_coords(&self) -> MultiSphericalPoint {
            self.coords()
        }

        #[getter]
        #[pyo3(name = "boundary")]
        fn py_boundary(&self) -> Option<MultiArcString> {
            self.boundary()
        }

        #[getter]
        #[pyo3(name = "representative_point")]
        fn py_representative_point(&self) -> SphericalPoint {
            self.representative_point()
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::SphericalPoint(point) => self.distance(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.distance(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.distance(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.distance(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.distance(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.contains(&bounds),
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
                AnyGeometry::AngularBounds(bounds) => self.within(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.within(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.within(&multipolygon),
            }
        }

        #[pyo3(name = "intersects")]
        fn py_intersects(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.intersects(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> Option<AnyGeometry> {
            match other {
                AnyGeometry::SphericalPoint(point) => self
                    .intersection(&point)
                    .map(AnyGeometry::SphericalPoint),
                AnyGeometry::MultiSphericalPoint(multipoint) => self
                    .intersection(&multipoint)
                    .map(AnyGeometry::MultiSphericalPoint),
                AnyGeometry::ArcString(arcstring) => self
                    .intersection(&arcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::MultiArcString(multiarcstring) => self
                    .intersection(&multiarcstring)
                    .map(AnyGeometry::MultiArcString),
                AnyGeometry::AngularBounds(bounds) => self
                    .intersection(&bounds)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::SphericalPolygon(polygon) => self
                    .intersection(&polygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self
                    .intersection(&multipolygon)
                    .map(AnyGeometry::MultiSphericalPolygon),
            }
        }

        #[pyo3(name = "touches")]
        fn py_touches(&self, other: AnyGeometry) -> bool {
            match other {
                AnyGeometry::SphericalPoint(point) => self.touches(&point),
                AnyGeometry::MultiSphericalPoint(multipoint) => self.touches(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.touches(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.touches(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.touches(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.touches(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.touches(&multipolygon),
            }
        }

        #[getter]
        #[pyo3(name = "parts")]
        fn py_parts(&self) -> Vec<SphericalPolygon> {
            self.polygons.to_owned().into()
        }

        fn __len__(&self) -> usize {
            self.len()
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
    use crate::sphericalgraph::SphericalGraph;

    #[pymethods]
    impl SphericalGraph {}

    #[pymodule(name = "array")]
    pub mod py_array {
        use super::*;

        #[pyfunction]
        #[pyo3(name = "normalize_vector")]
        fn py_normalize_vector<'py>(
            py: Python<'py>,
            xyz: PyReadonlyArray1<f64>,
        ) -> Bound<'py, PyArray1<f64>> {
            crate::sphericalpoint::normalize_vector(&xyz.as_array()).into_pyarray(py)
        }

        #[pyfunction]
        #[pyo3(name = "normalize_vectors")]
        fn py_normalize_vectors<'py>(
            py: Python<'py>,
            xyz: PyReadonlyArray2<f64>,
        ) -> Bound<'py, PyArray2<f64>> {
            crate::sphericalpoint::normalize_vectors(&xyz.as_array()).into_pyarray(py)
        }

        #[pyfunction]
        #[pyo3(name = "vector_arc_length")]
        fn py_vector_arc_length(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
            crate::sphericalpoint::vector_arc_length(&a.as_array(), &b.as_array())
        }

        #[pyfunction]
        #[pyo3(name="interpolate_points_along_vector_arc", signature=(a, b, n=50))]
        fn py_interpolate_points_along_vector_arc<'py>(
            py: Python<'py>,
            a: PyReadonlyArray1<f64>,
            b: PyReadonlyArray1<f64>,
            n: usize,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            match crate::arcstring::interpolate_points_along_vector_arc(
                &a.as_array(),
                &b.as_array(),
                n,
            ) {
                Ok(result) => Ok(result.into_pyarray(py)),
                Err(err) => Err(PyValueError::new_err(err)),
            }
        }

        #[pyfunction]
        #[pyo3(name = "vector_arc_angle", signature=(a, b, c, degrees=true))]
        fn py_vector_arc_angle(
            a: PyReadonlyArray1<f64>,
            b: PyReadonlyArray1<f64>,
            c: PyReadonlyArray1<f64>,
            degrees: bool,
        ) -> f64 {
            crate::arcstring::vector_arcs_angle_between(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                degrees,
            )
        }

        #[pyfunction]
        #[pyo3(name = "vector_arc_angles", signature=(a, b, c, degrees=true))]
        fn py_vector_arc_angles<'py>(
            py: Python<'py>,
            a: PyReadonlyArray2<f64>,
            b: PyReadonlyArray2<f64>,
            c: PyReadonlyArray2<f64>,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            crate::arcstring::vector_arcs_angles_between(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                degrees,
            )
            .into_pyarray(py)
        }

        #[pyfunction]
        #[pyo3(name = "spherical_triangle_area")]
        fn spherical_triangle_area<'py>(
            a: PyReadonlyArray1<f64>,
            b: PyReadonlyArray1<f64>,
            c: PyReadonlyArray1<f64>,
        ) -> f64 {
            crate::sphericalpolygon::spherical_triangle_area(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
            )
        }

        #[pyfunction]
        #[pyo3(name = "spherical_polygon_area")]
        fn spherical_polygon_area<'py>(points: PyReadonlyArray2<f64>) -> f64 {
            crate::sphericalpolygon::spherical_polygon_area(&points.as_array())
        }
    }
}
