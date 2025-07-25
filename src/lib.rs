mod angularbounds;
mod arcstring;
mod geometry;
mod geometrycollection;
mod sphericalgraph;
mod sphericalpolygon;
mod vectorpoint;

extern crate numpy;

use pyo3::prelude::*;

#[pymodule(name = "sphersgeo")]
mod py_sphersgeo {
    use super::*;
    use crate::{
        geometry::{
            AnyGeometry, ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        },
        geometrycollection::GeometryCollection,
    };
    use numpy::{
        ndarray::{array, s, Array, Array2, Axis},
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
    };
    use pyo3::{exceptions::PyValueError, types::PyType};

    #[pymodule_export]
    use super::vectorpoint::VectorPoint;

    #[derive(FromPyObject)]
    enum PyVectorPointInputs<'py> {
        NumpyArray(PyReadonlyArray1<'py, f64>),
        Tuple((f64, f64, f64)),
        List(Vec<f64>),
    }

    #[derive(FromPyObject)]
    enum PyVectorPointLonLatInputs<'py> {
        NumpyArray(PyReadonlyArray1<'py, f64>),
        Tuple((f64, f64)),
        List(Vec<f64>),
    }

    #[pymethods]
    impl VectorPoint {
        #[new]
        fn py_new(point: PyVectorPointInputs) -> PyResult<Self> {
            let xyz = match point {
                PyVectorPointInputs::NumpyArray(xyz) => xyz.as_array().to_owned(),
                PyVectorPointInputs::Tuple((x, y, z)) => array![x, y, z],
                PyVectorPointInputs::List(list) => Array::from_vec(list),
            };

            Self::try_from(xyz).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        #[classmethod]
        #[pyo3(name = "normalize")]
        fn py_normalize<'py>(_: &Bound<'_, PyType>, point: PyVectorPointInputs) -> PyResult<Self> {
            Ok(Self::py_new(point)?.normalized())
        }

        /// from the given coordinates, build an xyz vector representing a point on the sphere
        #[classmethod]
        #[pyo3(name = "from_lonlat")]
        fn py_from_lonlat<'py>(
            _: &Bound<'_, PyType>,
            coordinates: PyVectorPointLonLatInputs,
            degrees: bool,
        ) -> PyResult<Self> {
            let coordinates = match coordinates {
                PyVectorPointLonLatInputs::NumpyArray(coordinates) => {
                    coordinates.as_array().to_owned()
                }
                PyVectorPointLonLatInputs::Tuple((lon, lat)) => array![lon, lat],
                PyVectorPointLonLatInputs::List(list) => Array::from_vec(list),
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
            self.xyz.to_owned().into_pyarray(py)
        }

        /// convert this point on the sphere to angular coordinates
        #[pyo3(name = "to_lonlat")]
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
        #[pyo3(name = "angle")]
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
        #[pyo3(name = "vector_rotate_around")]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        #[pyo3(name = "combine")]
        fn py_combine(&self, other: &Self) -> MultiVectorPoint {
            self + other
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        fn __add__(&self, other: &Self) -> MultiVectorPoint {
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
    use super::vectorpoint::MultiVectorPoint;

    #[derive(FromPyObject)]
    enum PyMultiVectorPointInputs<'py> {
        NumpyArray(PyReadonlyArray2<'py, f64>),
        ListOfTuples(Vec<(f64, f64, f64)>),
        NestedList(Vec<Vec<f64>>),
        FlatList(Vec<f64>),
        PointList(Vec<VectorPoint>),
    }

    #[derive(FromPyObject)]
    enum PyMultiVectorPointLonLatInputs<'py> {
        NumpyArray(PyReadonlyArray2<'py, f64>),
        ListOfTuples(Vec<(f64, f64)>),
        NestedList(Vec<Vec<f64>>),
        FlatList(Vec<f64>),
    }

    #[pymethods]
    impl MultiVectorPoint {
        #[new]
        fn py_new<'py>(points: PyMultiVectorPointInputs) -> PyResult<Self> {
            match points {
                PyMultiVectorPointInputs::NumpyArray(xyz) => {
                    Self::try_from(xyz.as_array().to_owned())
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiVectorPointInputs::ListOfTuples(list) => Ok(Self::from(&list)),
                PyMultiVectorPointInputs::NestedList(list) => {
                    Self::try_from(&list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiVectorPointInputs::FlatList(list) => {
                    Self::try_from(list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
                PyMultiVectorPointInputs::PointList(list) => {
                    Self::try_from(&list).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
                }
            }
        }

        /// from the given coordinates, build xyz vectors representing points on the sphere
        #[classmethod]
        #[pyo3(name = "from_lonlats")]
        fn py_from_lonlats<'py>(
            _: &Bound<'_, PyType>,
            coordinates: PyMultiVectorPointLonLatInputs,
            degrees: bool,
        ) -> PyResult<Self> {
            let coordinates = match coordinates {
                PyMultiVectorPointLonLatInputs::NumpyArray(coordinates) => {
                    coordinates.as_array().to_owned()
                }
                PyMultiVectorPointLonLatInputs::ListOfTuples(list) => {
                    let mut xyz = Array2::uninit((list.len(), 2));
                    for (index, tuple) in list.iter().enumerate() {
                        array![tuple.0, tuple.1].assign_to(xyz.index_axis_mut(Axis(0), index));
                    }
                    unsafe { xyz.assume_init() }
                }
                PyMultiVectorPointLonLatInputs::NestedList(list) => {
                    let mut xyz = Array2::<f64>::default((list.len(), 3));
                    for (i, mut point) in xyz.axis_iter_mut(Axis(0)).enumerate() {
                        for (j, value) in point.iter_mut().enumerate() {
                            *value = list[i][j];
                        }
                    }

                    xyz
                }
                PyMultiVectorPointLonLatInputs::FlatList(list) => {
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
            self.xyz.to_owned().into_pyarray(py)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name = "to_lonlats")]
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
        #[pyo3(name = "vector_rotate_around")]
        fn py_vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.vector_rotate_around(other, theta, degrees)
        }

        /// lengths of the underlying xyz vectors
        #[getter]
        #[pyo3(name = "vector_lengths")]
        fn py_vector_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.vector_lengths().to_pyarray(py)
        }

        /// angles on the sphere between these points and other sets of points
        #[pyo3(name="angles",signature=(a,b,degrees))]
        fn py_angles<'py>(
            &self,
            py: Python<'py>,
            a: &MultiVectorPoint,
            b: &MultiVectorPoint,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            self.angles(a, b, degrees).to_pyarray(py)
        }

        /// whether these points share a line with the given points
        #[pyo3(name = "collinear")]
        fn py_collinear<'py>(
            &self,
            py: Python<'py>,
            a: &VectorPoint,
            b: &VectorPoint,
        ) -> Bound<'py, PyArray1<bool>> {
            self.collinear(a, b).to_pyarray(py)
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

        #[pyo3(name = "parts")]
        fn py_parts(&self) -> Vec<VectorPoint> {
            self.into()
        }

        fn __concat__(&self, other: &Self) -> Self {
            self + other
        }

        /// number of points in this collection
        fn __len__(&self) -> usize {
            self.len()
        }

        fn __getitem__(&self, index: usize) -> VectorPoint {
            VectorPoint {
                xyz: self.xyz.slice(s![index, ..]).to_owned(),
            }
        }

        #[pyo3(name = "append")]
        fn py_append(&mut self, other: VectorPoint) {
            self.push(other);
        }

        #[pyo3(name = "extend")]
        fn py_extend(&mut self, other: Self) {
            self.extend(other);
        }

        fn __iadd__(&mut self, other: &Self) {
            *self += other;
        }

        fn __add__(&self, other: &Self) -> Self {
            self + other
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
        NumpyArray(PyReadonlyArray2<'py, f64>),
        MultiPoint(MultiVectorPoint),
        ListOfTuples(Vec<(f64, f64, f64)>),
        NestedList(Vec<Vec<f64>>),
        FlatList(Vec<f64>),
    }

    #[pymethods]
    impl ArcString {
        #[new]
        fn py_new(points: PyArcStringInputs) -> PyResult<Self> {
            let points = match points {
                PyArcStringInputs::NumpyArray(xyz) => {
                    MultiVectorPoint::try_from(xyz.as_array().to_owned())
                        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?
                }
                PyArcStringInputs::MultiPoint(points) => points.to_owned(),
                PyArcStringInputs::ListOfTuples(list) => MultiVectorPoint::from(&list),
                PyArcStringInputs::NestedList(list) => MultiVectorPoint::try_from(&list)
                    .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?,
                PyArcStringInputs::FlatList(list) => MultiVectorPoint::try_from(list)
                    .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?,
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
            self.lengths().to_pyarray(py)
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiVectorPoint {
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
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
    use super::arcstring::MultiArcString;

    #[derive(FromPyObject)]
    enum PyMultiArcStringInputs<'py> {
        NumpyArrayList(Vec<PyReadonlyArray2<'py, f64>>),
        ListOfMultiPoints(Vec<MultiVectorPoint>),
        NestedListOfTuples(Vec<Vec<(f64, f64, f64)>>),
        NestedList(Vec<Vec<Vec<f64>>>),
    }

    #[pymethods]
    impl MultiArcString {
        #[new]
        fn py_new(arcstrings: PyMultiArcStringInputs) -> PyResult<Self> {
            let points = match arcstrings {
                PyMultiArcStringInputs::NumpyArrayList(xyzs) => {
                    let mut multipoints = vec![];
                    for xyz in xyzs {
                        multipoints.push(
                            MultiVectorPoint::try_from(xyz.as_array().to_owned())
                                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?,
                        );
                    }
                    multipoints
                }
                PyMultiArcStringInputs::ListOfMultiPoints(points) => points.to_owned(),
                PyMultiArcStringInputs::NestedListOfTuples(list) => list
                    .into_iter()
                    .map(|points| MultiVectorPoint::from(&points))
                    .collect(),
                PyMultiArcStringInputs::NestedList(list) => {
                    let mut multipoints = vec![];
                    for points in list {
                        multipoints.push(
                            MultiVectorPoint::try_from(&points)
                                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?,
                        )
                    }
                    multipoints
                }
            };

            Self::try_from(points).map_err(|err| PyValueError::new_err(format!("{:?}", err)))
        }

        /// radians subtended by each arcstring on the sphere
        #[getter]
        #[pyo3(name = "lengths")]
        fn py_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.lengths().to_pyarray(py)
        }

        /// midpoints of each arc
        #[getter]
        #[pyo3(name = "midpoints")]
        fn py_midpoints(&self) -> MultiVectorPoint {
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

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
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
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

        #[pyo3(name = "bounds")]
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
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
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
    use crate::sphericalpolygon::SphericalPolygon;

    #[pymethods]
    impl SphericalPolygon {
        /// interior point is required because a sphere is a finite space
        #[new]
        #[pyo3(signature=(exterior, interior, holes=None))]
        fn py_new(
            exterior: ArcString,
            interior: VectorPoint,
            holes: Option<MultiArcString>,
        ) -> PyResult<Self> {
            Self::new(exterior, interior, holes)
                .map_err(|err| PyValueError::new_err(format!("{:?}", err)))
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        /// closest angular distance on the sphere between this geometry and another
        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
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
    enum PyMultiSphericalPolygonInputs {
        ListOfPolygons(Vec<SphericalPolygon>),
    }

    #[pymethods]
    impl MultiSphericalPolygon {
        #[new]
        fn py_new(polygons: PyMultiSphericalPolygonInputs) -> PyResult<Self> {
            Ok(match polygons {
                PyMultiSphericalPolygonInputs::ListOfPolygons(polygons) => Self::try_from(polygons)
                    .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?,
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

        #[pyo3(name = "bounds")]
        fn py_bounds(&self, degrees: bool) -> AngularBounds {
            self.bounds(degrees)
        }

        #[getter]
        #[pyo3(name = "convex_hull")]
        fn py_convex_hull(&self) -> Option<SphericalPolygon> {
            self.convex_hull()
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> MultiVectorPoint {
            self.points()
        }

        #[pyo3(name = "distance")]
        fn py_distance(&self, other: AnyGeometry) -> f64 {
            match other {
                AnyGeometry::VectorPoint(point) => self.distance(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.distance(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.contains(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.contains(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.within(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.within(&multipoint),
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
                AnyGeometry::VectorPoint(point) => self.intersects(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersects(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersects(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersects(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersects(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersects(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => self.intersects(&multipolygon),
            }
        }

        #[pyo3(name = "intersection")]
        fn py_intersection(&self, other: AnyGeometry) -> GeometryCollection {
            match other {
                AnyGeometry::VectorPoint(point) => self.intersection(&point),
                AnyGeometry::MultiVectorPoint(multipoint) => self.intersection(&multipoint),
                AnyGeometry::ArcString(arcstring) => self.intersection(&arcstring),
                AnyGeometry::MultiArcString(multiarcstring) => self.intersection(&multiarcstring),
                AnyGeometry::AngularBounds(bounds) => self.intersection(&bounds),
                AnyGeometry::SphericalPolygon(polygon) => self.intersection(&polygon),
                AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                    self.intersection(&multipolygon)
                }
            }
        }

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
        #[pyo3(name = "arc_length_from_vectors")]
        fn py_arc_length_from_vectors(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
            crate::arcstring::arc_length_from_vectors(&a.as_array(), &b.as_array())
        }

        #[pyfunction]
        #[pyo3(name="arc_interpolate_points", signature=(a, b, n=50))]
        fn py_arc_interpolate_points<'py>(
            py: Python<'py>,
            a: PyReadonlyArray1<f64>,
            b: PyReadonlyArray1<f64>,
            n: usize,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            match crate::arcstring::arc_interpolate_points(&a.as_array(), &b.as_array(), n) {
                Ok(result) => Ok(result.to_pyarray(py)),
                Err(err) => Err(PyValueError::new_err(err)),
            }
        }

        #[pyfunction]
        #[pyo3(name = "arc_angle")]
        fn py_arc_angle(
            a: PyReadonlyArray1<f64>,
            b: PyReadonlyArray1<f64>,
            c: PyReadonlyArray1<f64>,
            degrees: bool,
        ) -> f64 {
            crate::arcstring::arc_angle(&a.as_array(), &b.as_array(), &c.as_array(), degrees)
        }

        #[pyfunction]
        #[pyo3(name = "arc_angles")]
        fn py_arc_angles<'py>(
            py: Python<'py>,
            a: PyReadonlyArray2<f64>,
            b: PyReadonlyArray2<f64>,
            c: PyReadonlyArray2<f64>,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            crate::arcstring::arc_angles(&a.as_array(), &b.as_array(), &c.as_array(), degrees)
                .to_pyarray(py)
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
