use crate::geometry::{ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use rayon::prelude::*;
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
    xyz / vector_length(xyz)
}

/// normalize the given vectors to length 1 (the unit sphere) while preserving direction
pub fn normalize_vectors(xyz: &ArrayView2<f64>) -> Array2<f64> {
    xyz / &vector_lengths(xyz).broadcast((1, xyz.nrows())).unwrap().t()
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

/// radians subtended by this arc on the sphere
///    Notes
///    -----
///    The length is computed using the following:
///
///       l = arccos(A ⋅ B) / r^2
pub fn vector_arc_length(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // avoid domain issues of a.dot(b).acos()
    cross_vector(a, b)
        .view()
        .pow2()
        .sum()
        .sqrt()
        .atan2(a.dot(b))
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
pub fn vector_to_lonlat(xyz: &ArrayView1<f64>, degrees: bool) -> Array1<f64> {
    if xyz.abs().sum() == 0.0 {
        // directionless vector
        return array![std::f64::NAN, 0.0];
    }

    let mut lon = xyz[1].atan2(xyz[0]);
    let full_rotation = 2.0 * std::f64::consts::PI;
    if lon < 0.0 {
        lon += full_rotation;
    } else if lon > full_rotation {
        lon -= full_rotation;
    }

    let lat = xyz[2].atan2((xyz[0].powi(2) + xyz[1].powi(2)).sqrt());

    let radians = array![lon, lat,];
    return if degrees {
        radians.to_degrees()
    } else {
        radians
    };
}

pub fn vectors_to_lonlats(xyzs: &ArrayView2<f64>, degrees: bool) -> Array2<f64> {
    let mut lons = Zip::from(xyzs.rows()).par_map_collect(|xyz| {
        if xyz.abs().sum() == 0.0 {
            // directionless vector
            std::f64::NAN
        } else {
            xyz[1].atan2(xyz[0])
        }
    });

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

    // let lats = Zip::from(xyzs.slice(s![.., 2]))
    //     .and(&vector_lengths(xyzs))
    //     .par_map_collect(|z, r| (z / r).asin());
    let lats = Zip::from(xyzs.rows())
        .par_map_collect(|xyz| xyz[2].atan2((xyz[0].powi(2) + xyz[1].powi(2)).sqrt()));

    let radians = stack(Axis(1), &[lons.view(), lats.view()]).unwrap();
    if degrees {
        radians.to_degrees()
    } else {
        radians
    }
}

pub fn point_within_kdtree(xyz: &ArrayView1<f64>, kdtree: &ImmutableKdTree<f64, 3>) -> bool {
    // so we must also normalize this point to compare them
    let xyz = normalize_vector(&xyz);

    // take advantage of the kdtree's distance function in 3D space
    let tolerance = 1e-10;
    let nearest = kdtree.nearest_one::<SquaredEuclidean>(&[xyz[0], xyz[1], xyz[2]]);

    nearest.distance < tolerance
}

/// xyz vector representing a point on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct SphericalPoint {
    pub xyz: Array1<f64>,
}

impl TryFrom<Array1<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(xyz: Array1<f64>) -> Result<Self, Self::Error> {
        if xyz.len() != 3 {
            Err(format!("array should have length 3, not {:?}", xyz.len()))
        } else {
            Ok(Self { xyz })
        }
    }
}

impl Into<Array1<f64>> for SphericalPoint {
    fn into(self) -> Array1<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView1<'p, f64>> for &'p SphericalPoint {
    fn into(self) -> ArrayView1<'p, f64> {
        self.xyz.view()
    }
}

impl TryFrom<Vec<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(xyz: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(Array1::<f64>::from_vec(xyz))
    }
}

impl Into<Vec<f64>> for SphericalPoint {
    fn into(self) -> Vec<f64> {
        self.xyz.to_vec()
    }
}

impl From<[f64; 3]> for SphericalPoint {
    fn from(xyz: [f64; 3]) -> Self {
        Self {
            xyz: Array1::<f64>::from_vec(xyz.to_vec()),
        }
    }
}

impl From<(f64, f64, f64)> for SphericalPoint {
    fn from(xyz: (f64, f64, f64)) -> Self {
        Self {
            xyz: array![xyz.0, xyz.1, xyz.2],
        }
    }
}

impl Into<[f64; 3]> for SphericalPoint {
    fn into(self) -> [f64; 3] {
        self.xyz.to_vec().try_into().unwrap()
    }
}

impl Into<MultiSphericalPoint> for &SphericalPoint {
    fn into(self) -> MultiSphericalPoint {
        MultiSphericalPoint::try_from(self.xyz.to_shape((1, 3)).unwrap().to_owned()).unwrap()
    }
}

impl SphericalPoint {
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

    pub fn combine(&self, other: &Self) -> MultiSphericalPoint {
        MultiSphericalPoint::try_from(stack(Axis(0), &[self.xyz.view(), other.xyz.view()]).unwrap())
            .unwrap()
    }

    /// normalize this vector to length 1 (the unit sphere) while preserving direction
    pub fn normalized(&self) -> Self {
        Self::try_from(normalize_vector(&self.xyz.view())).unwrap()
    }

    /// create n number of points equally spaced on an arc between this point and another point
    pub fn interpolate_between(
        &self,
        other: &Self,
        n: usize,
    ) -> Result<MultiSphericalPoint, String> {
        MultiSphericalPoint::try_from(crate::arcstring::interpolate_points_along_vector_arc(
            &self.xyz.view(),
            &other.xyz.view(),
            n,
        )?)
    }

    /// angle on the sphere between this point and two other points
    pub fn angle_between(&self, a: &SphericalPoint, b: &SphericalPoint, degrees: bool) -> f64 {
        crate::arcstring::vector_arcs_angle_between(&a.into(), &self.into(), &b.into(), degrees)
    }

    /// whether this point lies exactly between the given points
    pub fn collinear(&self, a: &SphericalPoint, b: &SphericalPoint) -> bool {
        crate::arcstring::vectors_collinear(&a.xyz.view(), &self.xyz.view(), &b.xyz.view())
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
    pub fn vector_rotate_around(&self, other: &Self, theta: &f64, degrees: bool) -> Self {
        let theta = if degrees { &theta.to_radians() } else { theta };

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

impl ToString for SphericalPoint {
    fn to_string(&self) -> String {
        format!("SphericalPoint({})", self.xyz)
    }
}

impl PartialEq for SphericalPoint {
    fn eq(&self, other: &SphericalPoint) -> bool {
        let tolerance = 3e-11;
        (&self.xyz - &other.xyz).abs().sum() < tolerance
    }
}

impl Add<&SphericalPoint> for &SphericalPoint {
    type Output = MultiSphericalPoint;

    fn add(self, rhs: &SphericalPoint) -> Self::Output {
        self.combine(rhs)
    }
}

impl Add<&MultiSphericalPoint> for &SphericalPoint {
    type Output = MultiSphericalPoint;

    fn add(self, rhs: &MultiSphericalPoint) -> MultiSphericalPoint {
        let mut owned = rhs.to_owned();
        owned.push(self.to_owned());
        owned
    }
}

impl Geometry for &SphericalPoint {
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

    fn coords(&self) -> MultiSphericalPoint {
        self.to_owned().into()
    }

    fn boundary(&self) -> Option<SphericalPoint> {
        None
    }

    fn representative_point(&self) -> SphericalPoint {
        (*self).to_owned()
    }
}

impl Geometry for SphericalPoint {
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

    fn coords(&self) -> MultiSphericalPoint {
        self.into()
    }

    fn boundary(&self) -> Option<SphericalPoint> {
        (&self).boundary()
    }

    fn representative_point(&self) -> SphericalPoint {
        (&self).representative_point()
    }
}

impl GeometricOperations<&SphericalPoint> for &SphericalPoint {
    fn distance(self, other: &SphericalPoint) -> f64 {
        if self.xyz == other.xyz {
            0.0
        } else {
            vector_arc_length(&self.xyz.view(), &other.xyz.view())
        }
    }

    fn contains(self, _: &SphericalPoint) -> bool {
        false
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self == other
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self == other {
            Some(SphericalPoint::try_from(self.xyz.to_owned()).unwrap())
        } else {
            None
        }
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self == other
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &SphericalPoint {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPoint) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }
}

impl GeometricOperations<&crate::arcstring::ArcString> for &SphericalPoint {
    fn distance(self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn within(self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn intersects(self, other: &crate::arcstring::ArcString) -> bool {
        self.within(other)
    }

    fn intersection(self, other: &crate::arcstring::ArcString) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &crate::arcstring::ArcString) -> bool {
        self.within(other)
    }
}

impl GeometricOperations<&crate::arcstring::MultiArcString> for &SphericalPoint {
    fn distance(self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn within(self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn intersects(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.within(other)
    }

    fn intersection(self, other: &crate::arcstring::MultiArcString) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.within(other)
    }
}

impl GeometricOperations<&crate::angularbounds::AngularBounds> for &SphericalPoint {
    fn distance(self, other: &crate::angularbounds::AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::angularbounds::AngularBounds) -> bool {
        false
    }

    fn within(self, other: &crate::angularbounds::AngularBounds) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::angularbounds::AngularBounds) -> bool {
        false
    }

    fn intersects(self, other: &crate::angularbounds::AngularBounds) -> bool {
        self.within(other)
    }

    fn intersection(self, other: &crate::angularbounds::AngularBounds) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &crate::angularbounds::AngularBounds) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &SphericalPoint {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.within(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &SphericalPoint {
    fn distance(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.within(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<SphericalPoint> {
        if self.intersects(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }
}

/// xyz vectors representing points on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiSphericalPoint {
    pub xyz: Array2<f64>,
    // this kdtree is built off of normalized versions of the xyz vectors
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl From<&Vec<SphericalPoint>> for MultiSphericalPoint {
    fn from(points: &Vec<SphericalPoint>) -> Self {
        let mut xyz = Array2::uninit((points.len(), 3));
        for (index, row) in xyz.axis_iter_mut(Axis(0)).enumerate() {
            points[index].xyz.assign_to(row);
        }
        unsafe { Self::try_from(xyz.assume_init()).unwrap_unchecked() }
    }
}

impl From<&Vec<MultiSphericalPoint>> for MultiSphericalPoint {
    fn from(multipoints: &Vec<MultiSphericalPoint>) -> Self {
        let mut xyzs = vec![];
        for multipoint in multipoints {
            xyzs.push(multipoint.xyz.view());
        }
        // TODO: remove duplicates
        Self::try_from(concatenate(Axis(0), xyzs.as_slice()).unwrap()).unwrap()
    }
}

impl Into<Array1<SphericalPoint>> for &MultiSphericalPoint {
    fn into(self) -> Array1<SphericalPoint> {
        Zip::from(self.xyz.rows())
            .par_map_collect(|row| SphericalPoint::try_from(row.to_owned()).unwrap())
    }
}

impl Into<Vec<SphericalPoint>> for &MultiSphericalPoint {
    fn into(self) -> Vec<SphericalPoint> {
        let points: Array1<SphericalPoint> = self.into();
        points.to_vec()
    }
}

impl TryFrom<Array2<f64>> for MultiSphericalPoint {
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

impl Into<Array2<f64>> for MultiSphericalPoint {
    fn into(self) -> Array2<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView2<'p, f64>> for &'p MultiSphericalPoint {
    fn into(self) -> ArrayView2<'p, f64> {
        self.xyz.view()
    }
}

impl From<Vec<[f64; 3]>> for MultiSphericalPoint {
    fn from(xyzs: Vec<[f64; 3]>) -> Self {
        let kdtree = ImmutableKdTree::<f64, 3>::from(xyzs.as_slice());
        Self {
            xyz: xyzs.into(),
            kdtree,
        }
    }
}

impl Into<Vec<[f64; 3]>> for &MultiSphericalPoint {
    fn into(self) -> Vec<[f64; 3]> {
        self.xyz
            .rows()
            .into_iter()
            .map(|row| unsafe { row.to_vec().try_into().unwrap_unchecked() })
            .collect()
    }
}

impl From<&Vec<(f64, f64, f64)>> for MultiSphericalPoint {
    fn from(xyzs: &Vec<(f64, f64, f64)>) -> Self {
        let mut xyz = Array2::uninit((xyzs.len(), 3));
        for (index, tuple) in xyzs.iter().enumerate() {
            array![tuple.0, tuple.1, tuple.2].assign_to(xyz.index_axis_mut(Axis(0), index));
        }
        Self::try_from(unsafe { xyz.assume_init() }).unwrap()
    }
}

impl TryFrom<&Vec<Vec<f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: &Vec<Vec<f64>>) -> Result<Self, Self::Error> {
        let mut xyz = Array2::<f64>::uninit((list.len(), 3));
        for (index, row) in xyz.axis_iter_mut(Axis(0)).enumerate() {
            let point = &list[index];
            if point.len() == 3 {
                Array1::<f64>::from_vec(point.to_owned()).assign_to(row)
            } else {
                return Err(format!("invalid shape {}", point.len()));
            }
        }
        Self::try_from(unsafe { xyz.assume_init() })
    }
}

impl<'a> TryFrom<&Vec<ArrayView1<'a, f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: &Vec<ArrayView1<'a, f64>>) -> Result<Self, Self::Error> {
        let mut xyz = Array2::<f64>::uninit((list.len(), 3));
        for (index, row) in xyz.axis_iter_mut(Axis(0)).enumerate() {
            let point = &list[index];
            if point.len() == 3 {
                point.assign_to(row);
            } else {
                return Err(format!("invalid shape {:?}", point.shape()));
            }
        }
        Self::try_from(unsafe { xyz.assume_init() })
    }
}

impl TryFrom<&Vec<Array1<f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: &Vec<Array1<f64>>) -> Result<Self, Self::Error> {
        let list: Vec<ArrayView1<f64>> = list.par_iter().map(|point| point.view()).collect();
        Self::try_from(&list)
    }
}

impl TryFrom<Vec<f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(
            Array2::from_shape_vec((list.len() / 3, 3), list)
                .map_err(|err| format!("{:?}", err))?,
        )
    }
}

impl<'a> TryFrom<&ArrayView1<'a, f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(xyz: &ArrayView1<'a, f64>) -> Result<Self, Self::Error> {
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

impl MultiSphericalPoint {
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

    pub fn nearest(&self, point: &SphericalPoint) -> SphericalPoint {
        // since the kdtree is over normalized vectors, the nearest vector in 3D space is also the nearest in angular distance
        let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&[
            point.xyz[0],
            point.xyz[1],
            point.xyz[2],
        ]);

        SphericalPoint {
            xyz: self.xyz.slice(s![nearest.item as usize, ..]).to_owned(),
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

    pub fn angles_between(
        &self,
        a: &MultiSphericalPoint,
        b: &MultiSphericalPoint,
        degrees: bool,
    ) -> Array1<f64> {
        // vector_arc_angles(&a.xyz.view(), &self.xyz.view(), &b.xyz.view(), degrees)
        Zip::from(self.xyz.rows())
            .and(a.xyz.rows())
            .and(b.xyz.rows())
            .par_map_collect(|point, a, b| {
                crate::arcstring::vector_arcs_angle_between(&point, &a, &b, degrees)
            })
    }

    pub fn collinear(&self, a: &SphericalPoint, b: &SphericalPoint) -> Array1<bool> {
        let points: Vec<SphericalPoint> = self.into();
        Array1::from_vec(
            points
                .par_iter()
                .map(|point| {
                    crate::arcstring::vectors_collinear(
                        &a.xyz.view(),
                        &point.xyz.view(),
                        &b.xyz.view(),
                    )
                })
                .collect(),
        )
    }

    fn push_xyz(&mut self, xyz: &ArrayView1<f64>, recreate: bool) {
        if !point_within_kdtree(&xyz, &self.kdtree) {
            unsafe { self.xyz.push_row(*xyz).unwrap_unchecked() }
            if recreate {
                self.recreate_kdtree();
            }
        }
    }
}

impl Sum for MultiSphericalPoint {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let multipoints: Vec<MultiSphericalPoint> = iter.collect();
        Self::from(&multipoints)
    }
}

impl ToString for MultiSphericalPoint {
    fn to_string(&self) -> String {
        format!("MultiSphericalPoint({})", self.xyz)
    }
}

impl PartialEq for MultiSphericalPoint {
    fn eq(&self, other: &MultiSphericalPoint) -> bool {
        let tolerance = 1e-11;
        if self.len() == other.len() {
            if self.xyz.sum() == other.xyz.sum() {
                let mut rows: Vec<ArrayView1<f64>> = Zip::from(self.xyz.rows())
                    .par_map_collect(|xyz| xyz)
                    .to_vec();
                let mut other_rows: Vec<ArrayView1<f64>> = Zip::from(self.xyz.rows())
                    .par_map_collect(|xyz| xyz)
                    .to_vec();

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

impl PartialEq<Vec<SphericalPoint>> for MultiSphericalPoint {
    fn eq(&self, other: &Vec<SphericalPoint>) -> bool {
        self.kdtree == MultiSphericalPoint::from(other).kdtree
    }
}

impl Add<&MultiSphericalPoint> for &MultiSphericalPoint {
    type Output = MultiSphericalPoint;

    fn add(self, rhs: &MultiSphericalPoint) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&MultiSphericalPoint> for MultiSphericalPoint {
    fn add_assign(&mut self, other: &MultiSphericalPoint) {
        self.extend(other.to_owned());
    }
}

impl Add<&SphericalPoint> for &MultiSphericalPoint {
    type Output = MultiSphericalPoint;

    fn add(self, rhs: &SphericalPoint) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&SphericalPoint> for MultiSphericalPoint {
    fn add_assign(&mut self, other: &SphericalPoint) {
        self.push(other.to_owned());
    }
}

impl Geometry for &MultiSphericalPoint {
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

    fn coords(&self) -> MultiSphericalPoint {
        (*self).to_owned()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        None
    }

    fn representative_point(&self) -> SphericalPoint {
        SphericalPoint {
            xyz: self.xyz.slice(s![0, ..]).to_owned(),
        }
    }
}

impl Geometry for MultiSphericalPoint {
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

    fn coords(&self) -> MultiSphericalPoint {
        self.to_owned()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        (&self).boundary()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
    }
}

impl MultiGeometry for &MultiSphericalPoint {
    fn len(&self) -> usize {
        self.xyz.nrows()
    }
}

impl MultiGeometry for MultiSphericalPoint {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl ExtendMultiGeometry<SphericalPoint> for MultiSphericalPoint {
    fn extend(&mut self, other: MultiSphericalPoint) {
        other.xyz.rows().into_iter().for_each(|row| {
            if !point_within_kdtree(&row, &self.kdtree) {
                unsafe { self.xyz.push_row(row.view()).unwrap_unchecked() }
            }
        });
        self.recreate_kdtree();
    }

    fn push(&mut self, other: SphericalPoint) {
        self.push_xyz(&other.xyz.view(), true);
    }
}

impl GeometricOperations<&SphericalPoint> for &MultiSphericalPoint {
    fn distance(self, other: &SphericalPoint) -> f64 {
        self.nearest(other).distance(other)
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        point_within_kdtree(&other.xyz.view(), &self.kdtree)
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self.contains(other) {
            Some(other.to_owned())
        } else {
            None
        }
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &MultiSphericalPoint {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        min_1darray(
            &Zip::from(other.xyz.rows())
                .par_map_collect(|other_xyz| {
                    // since the kdtree is over normalized vectors, the nearest vector in 3D space is also the nearest in angular distance
                    let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&[
                        other_xyz[0],
                        other_xyz[1],
                        other_xyz[2],
                    ]);
                    vector_arc_length(&self.xyz.slice(s![nearest.item as usize, ..]), &other_xyz)
                })
                .view(),
        )
        .unwrap_or(std::f64::NAN)
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiSphericalPoint) -> bool {
        if self.len() < other.len() {
            self.xyz
                .rows()
                .into_iter()
                .all(|xyz| point_within_kdtree(&xyz, &other.kdtree))
        } else {
            false
        }
    }

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let (less, more) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        let points: Vec<ArrayView1<f64>> = less
            .xyz
            .rows()
            .into_iter()
            .filter(|xyz| point_within_kdtree(&xyz, &more.kdtree))
            .collect();

        if points.len() > 0 {
            Some(MultiSphericalPoint::try_from(&points).unwrap())
        } else {
            None
        }
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        let (less, more) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        less.xyz
            .rows()
            .into_iter()
            .any(|xyz| point_within_kdtree(&xyz, &more.kdtree))
    }
}

impl GeometricOperations<&crate::arcstring::ArcString> for &MultiSphericalPoint {
    fn distance(self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn within(self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn intersects(self, other: &crate::arcstring::ArcString) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(self, other: &crate::arcstring::ArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<ArrayView1<f64>> = self
            .xyz
            .rows()
            .into_iter()
            .filter(|xyz| crate::arcstring::arcstring_contains_point(other, &xyz))
            .collect();

        if intersections.len() > 0 {
            MultiSphericalPoint::try_from(&intersections).ok()
        } else {
            None
        }
    }

    fn touches(self, other: &crate::arcstring::ArcString) -> bool {
        self.xyz
            .rows()
            .into_iter()
            .any(|xyz| crate::arcstring::arcstring_contains_point(other, &xyz))
    }
}

impl GeometricOperations<&crate::arcstring::MultiArcString> for &MultiSphericalPoint {
    fn distance(self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, other: &crate::arcstring::MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn intersects(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(self, other: &crate::arcstring::MultiArcString) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::arcstring::MultiArcString) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::angularbounds::AngularBounds> for &MultiSphericalPoint {
    fn distance(self, other: &crate::angularbounds::AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::angularbounds::AngularBounds) -> bool {
        false
    }

    fn within(self, other: &crate::angularbounds::AngularBounds) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::angularbounds::AngularBounds) -> bool {
        false
    }

    fn intersects(self, other: &crate::angularbounds::AngularBounds) -> bool {
        other.intersects(self)
    }

    fn intersection(
        self,
        other: &crate::angularbounds::AngularBounds,
    ) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::angularbounds::AngularBounds) -> bool {
        self.intersects(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &MultiSphericalPoint {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.intersects(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &MultiSphericalPoint {
    fn distance(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        // TODO: find a better algorithm than brute-force
        self.xyz.rows().into_iter().all(|xyz| {
            other.polygons.par_iter().any(|polygon| {
                crate::sphericalpolygon::point_in_polygon_exterior(
                    &xyz,
                    &polygon.interior_point.xyz.view(),
                    &polygon.exterior.points.xyz.view(),
                )
            })
        })
    }

    fn crosses(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }
}
