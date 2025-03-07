use crate::{
    angularbounds::AngularBounds,
    arcstring::{
        vector_arc_angle, vector_arc_length, vectors_collinear, ArcString, MultiArcString,
    },
    geometry::{
        ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        MultiGeometryIntoIterator,
    },
    sphericalpolygon::{MultiSphericalPolygon, SphericalPolygon},
};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use rayon::prelude::*;

use pyo3::prelude::*;
use std::{
    iter::Sum,
    num::NonZero,
    ops::{Add, AddAssign},
};

#[inline(always)]
pub fn min_1darray(arr: &ArrayView1<f64>) -> Option<f64> {
    if arr.is_any_nan() || arr.is_any_infinite() {
        None
    } else {
        Some(unsafe {
            *(arr
                .into_par_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
                .unwrap_unchecked())
        })
    }
}

#[inline(always)]
pub fn max_1darray(arr: &ArrayView1<f64>) -> Option<f64> {
    if arr.is_any_nan() || arr.is_any_infinite() {
        None
    } else {
        Some(unsafe {
            *(arr
                .into_par_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
                .unwrap_unchecked())
        })
    }
}

pub fn shift_rows(from: &ArrayView2<f64>, by: i32) -> Array2<f64> {
    let mut to = Array2::uninit(from.dim());
    from.slice(s![-by.., ..])
        .assign_to(to.slice_mut(s![..by, ..]));
    from.slice(s![..-by, ..])
        .assign_to(to.slice_mut(s![by.., ..]));

    unsafe { to.assume_init() }
}

/// normalize the given vector to length 1 (the unit sphere) while preserving direction
pub fn normalize_vector(xyz: &ArrayView1<f64>) -> Array1<f64> {
    xyz / xyz.pow2().sum().sqrt()
}

/// normalize the given vectors to length 1 (the unit sphere) while preserving direction
pub fn normalize_vectors(xyz: &ArrayView2<f64>) -> Array2<f64> {
    xyz / xyz
        .pow2()
        .sum_axis(Axis(1))
        .sqrt()
        .to_shape((xyz.shape()[0], 1))
        .unwrap()
        .to_owned()
}

fn vector_kdtree(xyz: &ArrayView2<f64>) -> ImmutableKdTree<f64, 3> {
    Zip::from(xyz.rows())
        .par_map_collect(|row| [row[0], row[1], row[2]])
        .as_slice()
        .unwrap()
        .into()
}

/// length of the given xyz vector
///
///     r = sqrt(x^2 + y^2 + z^2)
pub fn vector_length(vector: &ArrayView1<f64>) -> f64 {
    vector.pow2().sum().sqrt()
}

/// lengths of the given xyz vectors
///
///     r = sqrt(x^2 + y^2 + z^2)
pub fn vector_lengths(vectors: &ArrayView2<f64>) -> Array1<f64> {
    vectors.pow2().sum_axis(Axis(1)).sqrt()
}

pub fn cross_vector(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let bx = b[0];
    let by = b[1];
    let bz = b[2];

    array![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
}

pub fn cross_vectors(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let ax = a.slice(s![.., 0]);
    let ay = a.slice(s![.., 1]);
    let az = a.slice(s![.., 2]);
    let bx = b.slice(s![.., 0]);
    let by = b.slice(s![.., 1]);
    let bz = b.slice(s![.., 2]);

    let result = stack(
        Axis(0),
        &[
            (&ay * &bz - &az - &by).view(),
            (&az * &bx - &ax * &bz).view(),
            (&ax * &by - &ay * &ax).view(),
        ],
    );
    result.unwrap()
}

/// convert the given xyz vector to angular coordinates on the sphere
///
/// With radius *r*, longitude *l*, and latitude *b*:
///
///     r = sqrt(x^2 + y^2 + z^2)
///     l = arctan(y / x)
///     b = arcsin(z / r)
///
/// References
/// ----------
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn vector_to_lonlat(vector: &ArrayView1<f64>, degrees: bool) -> Array1<f64> {
    let mut lon = vector[1].atan2(vector[0]);
    let full_rotation = 2.0 * std::f64::consts::PI;
    if lon < 0.0 {
        lon += full_rotation;
    }
    if lon > full_rotation {
        lon -= full_rotation;
    }
    let radians = array![
        lon,
        vector[2].atan2((vector[0].powi(2) + vector[1].powi(2)).sqrt())
    ];
    return if degrees {
        radians.to_degrees()
    } else {
        radians
    };
}

pub fn vectors_to_lonlats(vectors: &ArrayView2<f64>, degrees: bool) -> Array2<f64> {
    let mut lons = Zip::from(vectors.rows()).par_map_collect(|xyz| xyz[1].atan2(xyz[0]));

    // mod longitudes past a full rotation
    let full_rotation = 2.0 * std::f64::consts::PI;
    lons.par_mapv_inplace(|lon| {
        if lon < 0.0 {
            lon + full_rotation
        } else if lon > full_rotation {
            lon - full_rotation
        } else {
            lon
        }
    });

    let lats = Zip::from(vectors.slice(s![.., 2]))
        .and(&vector_lengths(vectors))
        .par_map_collect(|z, r| (z / r).asin());
    // let lats = Zip::from(self.xyz.rows())
    //     .par_map_collect(|xyz| xyz[2].atan2((xyz[0].powi(2) + xyz[1].powi(2)).sqrt()));

    let radians = stack(Axis(1), &[lons.view(), lats.view()]).unwrap();
    if degrees {
        radians.to_degrees()
    } else {
        radians
    }
}

/// xyz vector representing a point on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct VectorPoint {
    pub xyz: Array1<f64>,
}

impl TryFrom<Array1<f64>> for VectorPoint {
    type Error = String;

    fn try_from(xyz: Array1<f64>) -> Result<Self, Self::Error> {
        if xyz.len() != 3 {
            Err(format!("array should have length 3, not {:?}", xyz.len()))
        } else {
            Ok(Self { xyz })
        }
    }
}

impl Into<Array1<f64>> for VectorPoint {
    fn into(self) -> Array1<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView1<'p, f64>> for &'p VectorPoint {
    fn into(self) -> ArrayView1<'p, f64> {
        self.xyz.view()
    }
}

impl TryFrom<Vec<f64>> for VectorPoint {
    type Error = String;

    fn try_from(xyz: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(Array1::<f64>::from_vec(xyz))
    }
}

impl Into<Vec<f64>> for VectorPoint {
    fn into(self) -> Vec<f64> {
        self.xyz.to_vec()
    }
}

impl From<[f64; 3]> for VectorPoint {
    fn from(xyz: [f64; 3]) -> Self {
        Self::try_from(xyz.to_vec()).unwrap()
    }
}

impl Into<[f64; 3]> for VectorPoint {
    fn into(self) -> [f64; 3] {
        self.xyz.to_vec().try_into().unwrap()
    }
}

impl Into<MultiVectorPoint> for &VectorPoint {
    fn into(self) -> MultiVectorPoint {
        MultiVectorPoint::try_from(self.xyz.to_shape((1, 3)).unwrap().to_owned()).unwrap()
    }
}

impl VectorPoint {
    /// normalize the given xyz vector
    pub fn normalize(xyz: &ArrayView1<f64>) -> Self {
        Self::try_from(normalize_vector(xyz)).unwrap()
    }

    /// from the given coordinates, build an xyz vector representing a point on the sphere
    ///
    /// With radius *r*, longitude *l*, and latitude *b*:
    ///
    ///     x = r * cos(l) * cos(b)
    ///     y = r * sin(l) * cos(b)
    ///     z = r * sin(b)
    ///
    /// References
    /// ----------
    /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
    pub fn try_from_lonlat(coordinates: &ArrayView1<f64>, degrees: bool) -> Result<Self, String> {
        if coordinates.len() == 2 {
            let coordinates = if degrees {
                coordinates.to_radians()
            } else {
                coordinates.to_owned()
            };

            let lon = coordinates[0];
            let lat = coordinates[1];
            let (lon_sin, lon_cos) = lon.sin_cos();
            let (lat_sin, lat_cos) = lat.sin_cos();

            Ok(Self::try_from(array![lon_cos * lat_cos, lon_sin * lat_cos, lat_sin]).unwrap())
        } else {
            Err(String::from("invalid shape"))
        }
    }

    /// convert this point on the sphere to angular coordinates
    ///
    /// With radius *r*, longitude *l*, and latitude *b*:
    ///
    ///     r = sqrt(x^2 + y^2 + z^2)
    ///     l = arctan(y / x)
    ///     b = arcsin(z / r)
    ///
    /// References
    /// ----------
    /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
    pub fn to_lonlat(&self, degrees: bool) -> Array1<f64> {
        vector_to_lonlat(&self.xyz.view(), degrees)
    }

    /// normalize this vector to length 1 (the unit sphere) while preserving direction
    pub fn normalized(&self) -> Self {
        Self::try_from(normalize_vector(&self.xyz.view())).unwrap()
    }

    /// angle on the sphere between this point and two other points
    pub fn angle(&self, a: &VectorPoint, b: &VectorPoint, degrees: bool) -> f64 {
        vector_arc_angle(&a.into(), &self.into(), &b.into(), degrees)
    }

    /// whether this point lies exactly between the given points
    pub fn collinear(&self, a: &VectorPoint, b: &VectorPoint) -> bool {
        vectors_collinear(&a.xyz.view(), &self.xyz.view(), &b.xyz.view())
    }

    /// length of the underlying xyz vector
    ///
    ///     r = sqrt(x^2 + y^2 + z^2)
    pub fn vector_length(&self) -> f64 {
        vector_length(&self.xyz.view())
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vector(&self.into(), &other.into());
        crossed.try_into().unwrap()
    }

    /// rotate this xyz vector by theta angle around another xyz vector
    pub fn vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
        let theta = if degrees { theta.to_radians() } else { theta };

        let a = &self.normalized().xyz;
        let ax = a[0];
        let ay = a[1];
        let az = a[2];

        let b = &other.normalized().xyz;
        let bx = b[0];
        let by = b[1];
        let bz = b[2];

        Self::try_from(
            -b * -a * b * (1.0 - theta.cos())
                + a * theta.cos()
                + array![-bz * ay + by * az, bz * ax - bx * az, -by * ax - bx * ay,] * theta.sin(),
        )
        .unwrap()
    }
}

impl ToString for VectorPoint {
    fn to_string(&self) -> String {
        format!("VectorPoint({})", self.xyz)
    }
}

impl PartialEq for VectorPoint {
    fn eq(&self, other: &VectorPoint) -> bool {
        let tolerance = 3e-11;
        (&self.xyz - &other.xyz).abs().sum() < tolerance
    }
}

impl Add<&VectorPoint> for &VectorPoint {
    type Output = MultiVectorPoint;

    fn add(self, rhs: &VectorPoint) -> Self::Output {
        Self::Output::try_from(stack(Axis(0), &[self.xyz.view(), rhs.xyz.view()]).unwrap()).unwrap()
    }
}

impl Add<&MultiVectorPoint> for &VectorPoint {
    type Output = MultiVectorPoint;

    fn add(self, rhs: &MultiVectorPoint) -> MultiVectorPoint {
        let mut owned = rhs.to_owned();
        owned.push(self.to_owned());
        owned
    }
}

impl Geometry for &VectorPoint {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        0.
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        let lonlat = self.to_lonlat(degrees);
        [lonlat[0], lonlat[1], lonlat[0], lonlat[1]].into()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        None
    }

    fn points(&self) -> MultiVectorPoint {
        self.to_owned().into()
    }
}

impl Geometry for VectorPoint {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> MultiVectorPoint {
        self.into()
    }
}

impl GeometricOperations<&VectorPoint> for &VectorPoint {
    fn distance(self, other: &VectorPoint) -> f64 {
        if self.xyz == other.xyz {
            0.
        } else {
            vector_arc_length(&self.xyz.view(), &other.xyz.view())
        }
    }

    fn contains(self, other: &VectorPoint) -> bool {
        self.intersects(other)
    }

    fn within(self, other: &VectorPoint) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        let tolerance = 3e-11;
        (&other.xyz - &self.xyz).abs().sum() < tolerance
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &VectorPoint) -> Option<VectorPoint> {
        if self.intersects(other) {
            Some(VectorPoint::try_from(self.xyz.to_owned()).unwrap())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiVectorPoint> for &VectorPoint {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn within(self, other: &MultiVectorPoint) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiVectorPoint) -> Option<VectorPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&ArcString> for &VectorPoint {
    fn distance(self, other: &ArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &ArcString) -> bool {
        false
    }

    fn within(self, other: &ArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<VectorPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiArcString> for &VectorPoint {
    fn distance(self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiArcString) -> bool {
        false
    }

    fn within(self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiArcString) -> Option<VectorPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&AngularBounds> for &VectorPoint {
    fn distance(self, other: &AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularBounds) -> Option<VectorPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&SphericalPolygon> for &VectorPoint {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> Option<VectorPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &VectorPoint {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.within(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<VectorPoint> {
        if self.intersects(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }
}

/// xyz vectors representing points on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiVectorPoint {
    pub xyz: Array2<f64>,
    // this kdtree is built off of normalized versions of the xyz vectors
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl From<&Vec<VectorPoint>> for MultiVectorPoint {
    fn from(points: &Vec<VectorPoint>) -> Self {
        let mut xyz = Array2::uninit((points.len(), 3));
        for (index, point) in points.iter().enumerate() {
            point.xyz.assign_to(xyz.index_axis_mut(Axis(0), index));
        }
        Self::try_from(unsafe { xyz.assume_init() }).unwrap()
    }
}

impl From<&Vec<MultiVectorPoint>> for MultiVectorPoint {
    fn from(multipoints: &Vec<MultiVectorPoint>) -> Self {
        let mut xyzs = vec![];
        for multipoint in multipoints {
            xyzs.push(multipoint.xyz.view());
        }
        // TODO: remove duplicates
        Self::try_from(concatenate(Axis(0), xyzs.as_slice()).unwrap()).unwrap()
    }
}

impl Iterator for MultiGeometryIntoIterator<MultiVectorPoint> {
    type Item = VectorPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.multi.len() {
            Some(
                VectorPoint::try_from(self.multi.xyz.slice(s![self.index, ..]).to_owned()).unwrap(),
            )
        } else {
            None
        }
    }
}

impl IntoIterator for MultiVectorPoint {
    type Item = VectorPoint;

    type IntoIter = MultiGeometryIntoIterator<MultiVectorPoint>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            multi: self,
            index: 0,
        }
    }
}

impl Into<Array1<VectorPoint>> for &MultiVectorPoint {
    fn into(self) -> Array1<VectorPoint> {
        Zip::from(self.xyz.rows())
            .par_map_collect(|row| VectorPoint::try_from(row.to_owned()).unwrap())
    }
}

impl Into<Vec<VectorPoint>> for &MultiVectorPoint {
    fn into(self) -> Vec<VectorPoint> {
        let points: Array1<VectorPoint> = self.into();
        points.to_vec()
    }
}

impl TryFrom<Array2<f64>> for MultiVectorPoint {
    type Error = String;

    fn try_from(xyz: Array2<f64>) -> Result<Self, Self::Error> {
        if xyz.shape()[1] != 3 {
            Err(format!("array should be Nx3, not Nx{:?}", xyz.shape()[1]))
        } else {
            let kdtree = vector_kdtree(&normalize_vectors(&xyz.view()).view());
            Ok(Self { xyz, kdtree })
        }
    }
}

impl Into<Array2<f64>> for MultiVectorPoint {
    fn into(self) -> Array2<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView2<'p, f64>> for &'p MultiVectorPoint {
    fn into(self) -> ArrayView2<'p, f64> {
        self.xyz.view()
    }
}

impl From<Vec<[f64; 3]>> for MultiVectorPoint {
    fn from(xyz: Vec<[f64; 3]>) -> Self {
        let kdtree = ImmutableKdTree::<f64, 3>::from(xyz.as_slice());
        Self {
            xyz: xyz.into(),
            kdtree,
        }
    }
}

impl Into<Vec<[f64; 3]>> for &MultiVectorPoint {
    fn into(self) -> Vec<[f64; 3]> {
        self.xyz
            .rows()
            .into_iter()
            .map(|row| unsafe { row.to_vec().try_into().unwrap_unchecked() })
            .collect()
    }
}

impl From<&Vec<(f64, f64, f64)>> for MultiVectorPoint {
    fn from(points: &Vec<(f64, f64, f64)>) -> Self {
        let mut xyz = Array2::uninit((points.len(), 2));
        for (index, tuple) in points.iter().enumerate() {
            array![tuple.0, tuple.1, tuple.2].assign_to(xyz.index_axis_mut(Axis(0), index));
        }
        Self::try_from(unsafe { xyz.assume_init() }).unwrap()
    }
}

impl TryFrom<&Vec<Vec<f64>>> for MultiVectorPoint {
    type Error = String;

    fn try_from(list: &Vec<Vec<f64>>) -> Result<Self, Self::Error> {
        let mut xyz = Array2::<f64>::default((list.len(), 3));
        for (i, mut point) in xyz.axis_iter_mut(Axis(0)).enumerate() {
            for (j, value) in point.iter_mut().enumerate() {
                *value = list[i][j];
            }
        }

        Self::try_from(xyz)
    }
}

impl TryFrom<Vec<Array1<f64>>> for MultiVectorPoint {
    type Error = String;

    fn try_from(xyzs: Vec<Array1<f64>>) -> Result<Self, Self::Error> {
        let mut points = Array2::uninit((xyzs.len(), 3));
        for (index, xyz) in xyzs.iter().enumerate() {
            if xyz.len() == 3 {
                xyz.assign_to(points.index_axis_mut(Axis(0), index));
            } else {
                return Err(format!("invalid shape {:?}", xyz.shape()));
            }
        }

        Self::try_from(unsafe { points.assume_init() })
    }
}

impl TryFrom<Vec<f64>> for MultiVectorPoint {
    type Error = String;

    fn try_from(list: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(
            Array2::from_shape_vec((list.len(), 3), list).map_err(|err| format!("{:?}", err))?,
        )
    }
}

impl TryFrom<Array1<f64>> for MultiVectorPoint {
    type Error = String;

    fn try_from(xyz: Array1<f64>) -> Result<Self, Self::Error> {
        if xyz.len() % 3 == 0 {
            Ok(Self::try_from(
                xyz.to_shape((xyz.len() / 3, 3))
                    .map_err(|err| format!("{:?}", err))?
                    .to_owned(),
            )?)
        } else {
            Err(format!("invalid shape {:?}", xyz.shape()))
        }
    }
}

impl MultiVectorPoint {
    /// normalize the given xyz vectors
    pub fn normalize(xyz: &ArrayView2<f64>) -> Result<Self, String> {
        if xyz.len() != 3 {
            Err(format!("array should have length 3, not {:?}", xyz.len()))
        } else {
            let normalized = normalize_vectors(xyz);
            let kdtree = vector_kdtree(&normalized.view());
            Ok(Self {
                xyz: normalized,
                kdtree,
            })
        }
    }

    /// from the given coordinates, build xyz vectors representing points on the sphere
    ///
    /// With radius *r*, longitude *l*, and latitude *b*:
    ///
    ///     x = r * cos(l) * cos(b)
    ///     y = r * sin(l) * cos(b)
    ///     z = r * sin(b)
    ///
    /// References
    /// ----------
    /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
    pub fn try_from_lonlats(coordinates: &ArrayView2<f64>, degrees: bool) -> Result<Self, String> {
        if coordinates.shape()[1] == 2 {
            let coordinates = if degrees {
                coordinates.to_radians()
            } else {
                coordinates.to_owned()
            };

            let lon = coordinates.slice(s![.., 0]);
            let lat = coordinates.slice(s![.., 1]);
            let lon_sin = &lon.sin();
            let lat_sin = &lat.sin();
            let lon_cos = &lon.cos();
            let lat_cos = &lat.cos();

            Ok(Self::try_from(
                stack(
                    Axis(1),
                    &[
                        (lon_cos * lat_cos).view(),
                        (lon_sin * lat_cos).view(),
                        lat_sin.view(),
                    ],
                )
                .unwrap(),
            )
            .unwrap())
        } else {
            Err(String::from("invalid shape"))
        }
    }

    pub fn nearest(&self, point: &VectorPoint) -> VectorPoint {
        unsafe {
            let normalized = point.normalized().xyz;
            let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&[
                normalized[0],
                normalized[1],
                normalized[2],
            ]);

            VectorPoint::try_from(self.xyz.slice(s![nearest.item as usize, ..]).to_owned())
                .unwrap_unchecked()
        }
    }

    fn recreate_kdtree(&mut self) {
        self.kdtree = vector_kdtree(&normalize_vectors(&self.xyz.view()).view())
    }

    /// normalize the underlying vectors to length 1 (the unit sphere) while preserving direction
    pub fn normalized(&self) -> Self {
        let normalized = normalize_vectors(&self.xyz.view());
        Self {
            xyz: normalized,
            kdtree: self.kdtree.to_owned(),
        }
    }

    /// convert to angle coordinates along the sphere
    ///
    /// With radius *r*, longitude *l*, and latitude *b*:
    ///
    ///     r = sqrt(x^2 + y^2 + z^2)
    ///     l = arctan(y / x)
    ///     b = arcsin(z / r)
    ///
    /// References
    /// ----------
    /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
    pub fn to_lonlats(&self, degrees: bool) -> Array2<f64> {
        vectors_to_lonlats(&self.xyz.view(), degrees)
    }

    /// lengths of the underlying xyz vectors
    ///
    ///     r = sqrt(x^2 + y^2 + z^2)
    pub fn vector_lengths(&self) -> Array1<f64> {
        vector_lengths(&self.xyz.view())
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vectors(&self.into(), &other.into());
        crossed.try_into().unwrap()
    }

    /// rotate the underlying vector by theta angle around other vectors
    pub fn vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
        let theta = if degrees { theta.to_radians() } else { theta };

        let a = &self.normalized().xyz;
        let ax = a.slice(s![.., 0]);
        let ay = a.slice(s![.., 1]);
        let az = a.slice(s![.., 2]);

        let b = &other.normalized().xyz;
        let bx = b.slice(s![.., 0]);
        let by = b.slice(s![.., 1]);
        let bz = b.slice(s![.., 2]);

        Self::try_from(
            -b * -a * b * (1.0 - theta.cos())
                + a * theta.cos()
                + stack(
                    Axis(0),
                    &[
                        (-&bz * &ay + &by * &az).view(),
                        (&bz * &ax - &bx * &az).view(),
                        (-&by * &ax - &bx * &ay).view(),
                    ],
                )
                .unwrap()
                    * theta.sin(),
        )
        .unwrap()
    }

    pub fn angles(&self, a: &MultiVectorPoint, b: &MultiVectorPoint, degrees: bool) -> Array1<f64> {
        Zip::from(self.xyz.rows())
            .and(a.xyz.rows())
            .and(b.xyz.rows())
            .par_map_collect(|point, a, b| vector_arc_angle(&point, &a, &b, degrees))
    }

    pub fn collinear(&self, a: &VectorPoint, b: &VectorPoint) -> Array1<bool> {
        let points: Vec<VectorPoint> = self.into();
        Array1::from_vec(
            points
                .par_iter()
                .map(|point| vectors_collinear(&a.xyz.view(), &point.xyz.view(), &b.xyz.view()))
                .collect(),
        )
    }
}

impl Sum for MultiVectorPoint {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let multipoints: Vec<MultiVectorPoint> = iter.collect();
        Self::from(&multipoints)
    }
}

impl ToString for MultiVectorPoint {
    fn to_string(&self) -> String {
        format!("MultiVectorPoint({})", self.xyz)
    }
}

impl PartialEq for MultiVectorPoint {
    fn eq(&self, other: &MultiVectorPoint) -> bool {
        let tolerance = 1e-11;
        if self.len() == other.len() {
            if self.xyz.sum() == other.xyz.sum() {
                let mut rows: Vec<ArrayView1<f64>> = self.xyz.rows().into_iter().collect();
                let mut other_rows: Vec<ArrayView1<f64>> = self.xyz.rows().into_iter().collect();

                rows.sort_by(|a, b| a.sum().partial_cmp(&b.sum()).unwrap());
                other_rows.sort_by(|a, b| a.sum().partial_cmp(&b.sum()).unwrap());

                return Zip::from(stack(Axis(0), rows.as_slice()).unwrap().rows())
                    .and(stack(Axis(0), other_rows.as_slice()).unwrap().rows())
                    .all(|a, b| (&a - &b).abs().sum() < tolerance);
            }
        }

        return true;
    }
}

impl PartialEq<Vec<VectorPoint>> for MultiVectorPoint {
    fn eq(&self, other: &Vec<VectorPoint>) -> bool {
        self.kdtree == MultiVectorPoint::from(other).kdtree
    }
}

impl Add<&MultiVectorPoint> for &MultiVectorPoint {
    type Output = MultiVectorPoint;

    fn add(self, rhs: &MultiVectorPoint) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&MultiVectorPoint> for MultiVectorPoint {
    fn add_assign(&mut self, other: &MultiVectorPoint) {
        self.extend(other.to_owned());
    }
}

impl Add<&VectorPoint> for &MultiVectorPoint {
    type Output = MultiVectorPoint;

    fn add(self, rhs: &VectorPoint) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&VectorPoint> for MultiVectorPoint {
    fn add_assign(&mut self, other: &VectorPoint) {
        self.push(other.to_owned());
    }
}

impl Geometry for &MultiVectorPoint {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        0.
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        let coordinates = self.to_lonlats(degrees);
        let x = coordinates.slice(s![.., 0]);
        let y = coordinates.slice(s![.., 1]);

        [
            min_1darray(&x).unwrap(),
            min_1darray(&y).unwrap(),
            max_1darray(&x).unwrap(),
            max_1darray(&y).unwrap(),
        ]
        .into()
    }

    /// This code implements Andrew's monotone chain algorithm, which is a simple
    /// variant of the Graham scan.  Rather than sorting by x-coordinate, instead
    /// we sort the points in CCW order around an origin O such that all points
    /// are guaranteed to be on one side of some geodesic through O.  This
    /// ensures that as we scan through the points, each new point can only
    /// belong at the end of the chain (i.e., the chain is monotone in terms of
    /// the angle around O from the starting point).
    /// from https://github.com/google/s2geometry/blob/master/src/s2/s2convex_hull_query.cc#L123
    /// https://www.researchgate.net/profile/Jayaram-Ma-2/publication/303522254/figure/fig1/AS:365886075621376@1464245446409/Monotone-Chain-Algorithm-and-graphic-illustration.png
    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        if self.len() < 3 {
            return None;
        }

        let radians = self.to_lonlats(false);

        let tolerance: f64 = 3e-11;
        let mut west_points: Vec<ArrayView1<f64>> = vec![];
        for point in radians.rows() {
            if west_points.is_empty() || (point[0] - west_points[0][0]).abs() < tolerance {
                west_points.push(point);
            } else if point[0] < west_points[0][0] {
                west_points = vec![point];
            }
        }

        // sort points by distance to the starting point, using squared euclidean distance along the 3D cloud to find and test the points
        let points = self.kdtree.nearest_n::<SquaredEuclidean>(
            west_points[0].as_slice().unwrap().try_into().unwrap(),
            NonZero::<usize>::try_from(self.len() - 1).unwrap(),
        );

        for point in points {}

        todo!();
    }

    fn points(&self) -> MultiVectorPoint {
        (*self).to_owned()
    }
}

impl Geometry for MultiVectorPoint {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> MultiVectorPoint {
        (&self).points()
    }
}

impl MultiGeometry for &MultiVectorPoint {
    fn len(&self) -> usize {
        self.xyz.nrows()
    }
}

impl MultiGeometry for MultiVectorPoint {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl ExtendMultiGeometry<VectorPoint> for MultiVectorPoint {
    fn extend(&mut self, other: MultiVectorPoint) {
        // self.xyz = concatenate(Axis(0), &[self.xyz.view(), other.xyz.view()]).unwrap();
        other
            .xyz
            .rows()
            .into_iter()
            .for_each(|row| unsafe { self.xyz.push_row(row.view()).unwrap_unchecked() });
        self.recreate_kdtree();
    }

    fn push(&mut self, other: VectorPoint) {
        unsafe { self.xyz.push_row(other.xyz.view()).unwrap_unchecked() };
        self.recreate_kdtree();
    }
}

impl GeometricOperations<&VectorPoint> for &MultiVectorPoint {
    fn distance(self, other: &VectorPoint) -> f64 {
        // find the nearest point in 3D space (kdtree is over normalized vectors)
        let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&[
            other.xyz[0],
            other.xyz[1],
            other.xyz[2],
        ]);

        vector_arc_length(
            &other.xyz.view(),
            &self.xyz.slice(s![nearest.item as usize, ..]),
        )
    }

    fn contains(self, other: &VectorPoint) -> bool {
        // points on the kdtree are normalized to the unit sphere,
        // so we must also normalize this point to compare them
        let other = normalize_vector(&other.xyz.view());

        // take advantage of the kdtree's distance function in 3D space
        let tolerance = 1e-10;
        let nearest = self
            .kdtree
            .nearest_one::<SquaredEuclidean>(&[other[0], other[1], other[2]]);

        nearest.distance < tolerance
    }

    fn within(self, other: &VectorPoint) -> bool {
        self.len() == 1 && other.within(self)
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &VectorPoint) -> Option<VectorPoint> {
        let tolerance = 1e-10;
        let other_point = other.xyz.view();

        for point in self.xyz.rows() {
            if (&point - &other_point).abs().sum() < tolerance {
                return Some(VectorPoint::try_from(point.to_owned()).unwrap());
            }
        }

        None
    }
}

impl GeometricOperations<&MultiVectorPoint> for &MultiVectorPoint {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        // TODO: write a more efficient algorithm than brute-force
        min_1darray(
            &Zip::from(self.xyz.rows())
                .par_map_collect(|point| {
                    min_1darray(
                        &Zip::from(other.xyz.rows())
                            .par_map_collect(|other_point| vector_arc_length(&point, &other_point))
                            .view(),
                    )
                    .unwrap_or(0.)
                })
                .view(),
        )
        .unwrap_or(0.)
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiVectorPoint) -> bool {
        if self.len() <= other.len() {
            match self.intersection(other) {
                Some(intersection) => intersection.len() == self.len(),
                None => false,
            }
        } else {
            false
        }
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        self.intersection(other).is_some()
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiVectorPoint) -> Option<MultiVectorPoint> {
        let tolerance: f64 = 3e-11;
        let other_point = other.xyz.view();

        let mut points: Vec<ArrayView1<f64>> = vec![];
        for point in self.xyz.rows() {
            if (&point - &other_point).abs().sum() < tolerance {
                points.push(point)
            }
        }

        if points.len() > 0 {
            Some(MultiVectorPoint::try_from(stack(Axis(0), points.as_slice()).unwrap()).unwrap())
        } else {
            None
        }
    }
}

impl GeometricOperations<&ArcString> for &MultiVectorPoint {
    fn distance(self, other: &ArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &ArcString) -> bool {
        false
    }

    fn within(self, other: &ArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &ArcString) -> bool {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        points.par_iter().any(|point| point.within(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<MultiVectorPoint> {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        Some(MultiVectorPoint::from(
            &points
                .par_iter()
                .filter_map(|point| {
                    if point.within(other) {
                        Some(point.to_owned())
                    } else {
                        None
                    }
                })
                .collect::<Vec<VectorPoint>>(),
        ))
    }
}

impl GeometricOperations<&MultiArcString> for &MultiVectorPoint {
    fn distance(self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiArcString) -> bool {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        points.par_iter().all(|point| point.within(other))
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        points.par_iter().any(|point| point.within(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiArcString) -> Option<MultiVectorPoint> {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        Some(MultiVectorPoint::from(
            &points
                .par_iter()
                .filter_map(|point| {
                    if point.within(other) {
                        Some(point.to_owned())
                    } else {
                        None
                    }
                })
                .collect::<Vec<VectorPoint>>(),
        ))
    }
}

impl GeometricOperations<&AngularBounds> for &MultiVectorPoint {
    fn distance(self, other: &AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularBounds) -> Option<MultiVectorPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiVectorPoint {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> Option<MultiVectorPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiVectorPoint {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        points.par_iter().all(|point| point.within(other))
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        // TODO: vectorize this across xyz array
        let points: Vec<VectorPoint> = self.into();
        points.par_iter().any(|point| point.within(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiVectorPoint> {
        other.intersection(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorpoint::{MultiVectorPoint, VectorPoint};

    #[test]
    fn test_normalize() {
        let xyz = Array1::<f64>::linspace(-100.0, 100.0, 18);
        let points = MultiVectorPoint::try_from(
            stack(Axis(1), &[xyz.view(), xyz.view(), xyz.view()]).unwrap(),
        )
        .unwrap();

        assert_ne!(
            points.vector_lengths(),
            array![1.0].broadcast(points.xyz.nrows()).unwrap()
        );

        let normalized = points.normalized();

        assert!(Zip::from(&normalized.vector_lengths()).all(|length| length == &1.0));

        assert!(Zip::from(&normalized.xyz.powi(2).sum_axis(Axis(1)).sqrt())
            .all(|length| length == &1.0),);
    }

    #[test]
    fn test_already_normalized() {
        for i in 0..3 {
            let mut xyz = array![0.0, 0.0, 0.0];
            xyz[i] = 1.0;
            let normalized = VectorPoint { xyz }.normalized().xyz;
            let length = normalized.powi(2).sum().sqrt();
            assert_eq!(length, 1.0);
        }
    }

    #[test]
    fn test_from_lonlat() {
        let tolerance = 3e-8;

        let a_lonlat = array![60.0, 0.0];
        let b_lonlat = array![60.0, 30.0];

        let a = VectorPoint::try_from_lonlat(&a_lonlat.view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&b_lonlat.view(), true).unwrap();

        assert!(Zip::from(&(a.to_lonlat(true) - a_lonlat).abs()).all(|point| point < &tolerance));
        assert!(Zip::from(&(b.to_lonlat(true) - b_lonlat).abs()).all(|point| point < &tolerance));

        let lons = Array1::<f64>::linspace(-360.0, 360.0, 360);

        let equator_lat = array![0.0];
        let equator_lats = equator_lat.broadcast(lons.len()).unwrap();
        let equators = Zip::from(&lons)
            .and(equator_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lon.to_owned(), lat.to_owned()].view(), true)
                    .unwrap()
            });
        let multi_equator = MultiVectorPoint::try_from_lonlats(
            &stack(Axis(1), &[lons.view(), equator_lats]).unwrap().view(),
            true,
        )
        .unwrap();

        assert!(Zip::from(multi_equator.xyz.rows())
            .and(&equators)
            .all(|multi, single| multi == single.xyz));

        assert_eq!(
            multi_equator.xyz.slice(s![.., 2]),
            Array1::<f64>::zeros(multi_equator.xyz.nrows())
        );

        let north_pole_lat = array![90.0];
        let north_pole_lats = north_pole_lat.broadcast(lons.len()).unwrap();
        let north_poles = Zip::from(&lons)
            .and(north_pole_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lon.to_owned(), lat.to_owned()].view(), true)
                    .unwrap()
            });
        let multi_north_pole = MultiVectorPoint::try_from_lonlats(
            &stack(Axis(1), &[lons.view(), north_pole_lats])
                .unwrap()
                .view(),
            true,
        )
        .unwrap();

        assert!(Zip::from(multi_north_pole.xyz.rows())
            .and(&north_poles)
            .all(|multi, single| multi == single.xyz));

        assert!(Zip::from(multi_north_pole.xyz.view())
            .and(
                stack(
                    Axis(1),
                    &[
                        Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                        Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                        Array1::<f64>::ones(multi_north_pole.xyz.nrows()).view()
                    ]
                )
                .unwrap()
                .view()
            )
            .all(|test, reference| (test - reference).abs() < tolerance));

        let south_pole_lat = array![-90.0];
        let south_pole_lats = south_pole_lat.broadcast(lons.len()).unwrap();
        let south_poles = Zip::from(&lons)
            .and(south_pole_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lon.to_owned(), lat.to_owned()].view(), true)
                    .unwrap()
            });
        let multi_south_pole = MultiVectorPoint::try_from_lonlats(
            &stack(Axis(1), &[lons.view(), south_pole_lats])
                .unwrap()
                .view(),
            true,
        )
        .unwrap();

        assert!(Zip::from(multi_south_pole.xyz.rows())
            .and(&south_poles)
            .all(|multi, single| multi == single.xyz));

        assert!(Zip::from(multi_south_pole.xyz.view())
            .and(
                stack(
                    Axis(1),
                    &[
                        Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                        Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                        (-1.0 * Array1::<f64>::ones(multi_north_pole.xyz.nrows())).view()
                    ]
                )
                .unwrap()
                .view()
            )
            .all(|test, reference| (test - reference).abs() < tolerance));
    }

    #[test]
    fn test_to_lonlat() {
        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let lonlats = array![[0., 90.], [0., -90.], [45., 0.], [315., 0.]];

        let a = VectorPoint::try_from(xyz.slice(s![0, ..]).to_owned()).unwrap();
        let ac = a.to_lonlat(true);
        assert_eq!(ac, lonlats.slice(s![0, ..]).to_owned());

        let b = VectorPoint::try_from(xyz.slice(s![1, ..]).to_owned()).unwrap();
        let bc = b.to_lonlat(true);
        assert_eq!(bc, lonlats.slice(s![1, ..]).to_owned());

        let c = VectorPoint::try_from(xyz.slice(s![2, ..]).to_owned()).unwrap();
        let cc = c.to_lonlat(true);
        assert_eq!(cc, lonlats.slice(s![2, ..]).to_owned());

        let d = VectorPoint::try_from(xyz.slice(s![3, ..]).to_owned()).unwrap();
        let dc = d.to_lonlat(true);
        assert_eq!(dc, lonlats.slice(s![3, ..]).to_owned());

        let abcd = MultiVectorPoint::try_from(xyz).unwrap();
        let abcdc = abcd.to_lonlats(true);
        assert_eq!(abcdc, lonlats);
        assert_eq!(
            abcdc,
            stack(Axis(0), &[ac.view(), bc.view(), cc.view(), dc.view()]).unwrap()
        )
    }

    #[test]
    fn test_distance() {
        let tolerance = 3e-8;

        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let a = VectorPoint::try_from(xyz.slice(s![0, ..]).to_owned()).unwrap();
        let b = VectorPoint::try_from(xyz.slice(s![1, ..]).to_owned()).unwrap();
        let c = VectorPoint::try_from(xyz.slice(s![2, ..]).to_owned()).unwrap();
        let d = VectorPoint::try_from(xyz.slice(s![3, ..]).to_owned()).unwrap();

        let ab = MultiVectorPoint::try_from(xyz.slice(s![..2, ..]).to_owned()).unwrap();
        let bc = MultiVectorPoint::try_from(xyz.slice(s![1..3, ..]).to_owned()).unwrap();
        let cd = MultiVectorPoint::try_from(xyz.slice(s![2.., ..]).to_owned()).unwrap();

        assert_eq!((&a).distance(&b), std::f64::consts::PI);
        assert_eq!((&b).distance(&c), std::f64::consts::PI / 2.);
        assert_eq!((&c).distance(&d), std::f64::consts::PI / 2.);

        assert!((&a).distance(&a) < tolerance);

        assert!((&ab).distance(&bc) < tolerance);
        assert!((&bc).distance(&cd) < tolerance);
        assert_eq!((&ab).distance(&cd), std::f64::consts::PI / 2.);
    }

    #[test]
    fn test_str() {
        assert_eq!(
            VectorPoint::try_from(array![0.0, 1.0, 2.0])
                .unwrap()
                .to_string(),
            "VectorPoint([0, 1, 2])"
        );
        assert_eq!(
            MultiVectorPoint::try_from(array![[0.0, 1.0, 2.0]])
                .unwrap()
                .to_string(),
            "MultiVectorPoint([[0, 1, 2]])"
        );
    }

    #[test]
    fn test_add() {
        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let a = VectorPoint::try_from(xyz.slice(s![0, ..]).to_owned()).unwrap();
        let b = VectorPoint::try_from(xyz.slice(s![1, ..]).to_owned()).unwrap();
        let c = VectorPoint::try_from(xyz.slice(s![2, ..]).to_owned()).unwrap();
        let d = VectorPoint::try_from(xyz.slice(s![3, ..]).to_owned()).unwrap();

        let reference_ab = MultiVectorPoint::try_from(xyz.slice(s![0..2, ..]).to_owned()).unwrap();
        let reference_bc = MultiVectorPoint::try_from(xyz.slice(s![1..3, ..]).to_owned()).unwrap();
        let reference_cd = MultiVectorPoint::try_from(xyz.slice(s![2..4, ..]).to_owned()).unwrap();
        let reference_da = MultiVectorPoint::try_from(
            stack(Axis(0), &[xyz.slice(s![3, ..]), xyz.slice(s![0, ..])]).unwrap(),
        )
        .unwrap();

        assert_eq!(&a + &b, reference_ab);

        assert_eq!(&b + &c, reference_bc);

        assert_eq!(&c + &d, reference_cd);

        assert_eq!(&d + &a, reference_da);

        let reference_abc = MultiVectorPoint::try_from(xyz.slice(s![..3, ..]).to_owned()).unwrap();
        let reference_abcd = MultiVectorPoint::try_from(xyz).unwrap();

        assert_eq!(&reference_ab + &c, &(&a + &b) + &c);
        assert_eq!(&(&a + &b) + &c, reference_abc);
        assert_eq!(&reference_ab + &c, reference_abc);

        let mut abc = reference_ab.to_owned();
        abc.push(c);
        assert_eq!(abc, reference_abc);

        assert_eq!(&reference_ab + &reference_bc, reference_abcd);

        let mut abcd = reference_ab.to_owned();
        abcd.extend(reference_bc);
        assert_eq!(abcd, reference_abcd);
    }
}
