use crate::geometry::{GeometricOperations, GeometricPredicates, Geometry, MultiGeometry};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::{array, Array1, Array2, ArrayView1, Axis};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// length of the underlying xyz vector
///
///     r = sqrt(x^2 + y^2 + z^2)
fn xyz_length(xyz: &[f64; 3]) -> f64 {
    (xyz[0].powi(2) + xyz[1].powi(2) + xyz[2].powi(2)).sqrt()
}

pub fn xyz_dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    xyz_sum(&xyz_mul_xyz(a, b))
}

pub fn xyz_cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn xyz_add_xyz(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn xyz_sub_xyz(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn xyz_mul_xyz(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

pub fn xyz_div_xyz(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] / b[0], a[1] / b[1], a[2] / b[2]]
}

pub fn xyz_add_f64(a: &[f64; 3], b: &f64) -> [f64; 3] {
    [a[0] + b, a[1] + b, a[2] + b]
}

pub fn xyz_sub_f64(a: &[f64; 3], b: &f64) -> [f64; 3] {
    [a[0] - b, a[1] - b, a[2] - b]
}

pub fn xyz_mul_f64(a: &[f64; 3], b: &f64) -> [f64; 3] {
    [a[0] * b, a[1] * b, a[2] * b]
}

pub fn xyz_div_f64(a: &[f64; 3], b: &f64) -> [f64; 3] {
    [a[0] / b, a[1] / b, a[2] / b]
}

pub fn xyz_neg(xyz: &[f64; 3]) -> [f64; 3] {
    [-xyz[0], -xyz[1], -xyz[2]]
}

pub fn xyz_sum(xyz: &[f64; 3]) -> f64 {
    xyz[0] + xyz[1] + xyz[2]
}

pub fn xyz_abs(xyz: &[f64; 3]) -> [f64; 3] {
    [xyz[0].abs(), xyz[1].abs(), xyz[2].abs()]
}

pub fn xyz_eq(a: &[f64; 3], b: &[f64; 3]) -> bool {
    xyz_sum(&xyz_abs(&xyz_sub_xyz(a, b))) < 3e-11
}

pub fn xyzs_sum(xyzs: &Vec<[f64; 3]>) -> [f64; 3] {
    let mut sum = [0.0, 0.0, 0.0];
    for xyz in xyzs {
        sum = xyz_add_xyz(&sum, xyz);
    }
    sum
}

pub fn xyzs_mean(xyzs: &Vec<[f64; 3]>) -> [f64; 3] {
    xyz_div_f64(&xyzs_sum(xyzs), &(xyzs.len() as f64))
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
fn lonlat_to_xyz(lonlat: &[f64; 2]) -> [f64; 3] {
    let lon = lonlat[0].to_radians();
    let lat = lonlat[1].to_radians();
    let (lon_sin, lon_cos) = lon.sin_cos();
    let (lat_sin, lat_cos) = lat.sin_cos();

    [lon_cos * lat_cos, lon_sin * lat_cos, lat_sin]
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
fn xyz_to_lonlat(xyz: &[f64; 3]) -> [f64; 2] {
    if xyz_eq(xyz, &[0.0, 0.0, 0.0]) {
        // directionless vector
        return [f64::NAN, 0.0];
    }

    let mut lon = xyz[1].atan2(xyz[0]);
    let full_rotation = 2.0 * std::f64::consts::PI;
    if lon < 0.0 {
        lon += full_rotation;
    } else if lon > full_rotation {
        lon -= full_rotation;
    }

    let lat = xyz[2].atan2((xyz[0].powi(2) + xyz[1].powi(2)).sqrt());

    [lon.to_degrees(), lat.to_degrees()]
}

/// rotate xyz vector by theta angle around another xyz vector
fn xyz_rotate_around(a: &[f64; 3], b: &[f64; 3], theta: &f64) -> [f64; 3] {
    let theta = theta.to_radians();
    let theta_sin = theta.sin();
    let theta_cos = theta.cos();

    let a = array![a[0], a[1], a[2]];
    let b = array![b[0], b[1], b[2]];

    let rotated = -&b * -&a * &b * (1.0 - theta_cos)
        + (&a * theta_cos)
        + array![
            -b[2] * a[1] + b[1] * a[2],
            b[2] * a[0] - b[0] * a[2],
            -b[1] * a[0] - b[0] * a[1],
        ] * theta_sin;

    [rotated[0], rotated[1], rotated[2]]
}

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
pub fn xyzs_distance_over_sphere_radians(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    if xyz_eq(a, b) {
        0.0
    } else {
        let distance = xyz_dot(a, b).acos();
        if !distance.is_nan() {
            distance
        } else {
            let crossed = xyz_cross(a, b);

            // avoid domain issues of a.dot(b).acos()
            (crossed[0].powi(2) + crossed[1].powi(2) + crossed[2].powi(2))
                .sqrt()
                .atan2(xyz_dot(a, b))
        }
    }
}

/// given three XYZ vector points on the sphere (`a`, `b`, and `c`), retrieve the angle at `b` formed by arcs `ab` and `bc`
///
///     cos(ca) = cos(bc) * cos(ab) + sin(bc) * sin(ab) * cos(b)
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. p132. 1994. Academic Press. doi:10.5555/180895.180907
///   `pdf <https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover>`_
pub fn xyz_two_arc_angle_radians(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let tolerance = 3e-11;

    // let abx = cross_vector(&a, &b);
    // let bcx = cross_vector(&b, &c);

    // let angle = if vector_arc_length(a, c) < tolerance
    //     // || vector_length(&abx.view()) < tolerance
    //     // || vector_length(&bcx.view()) < tolerance
    // {
    //     0.0
    // } else {
    //     let x = normalize_vector(&cross_vector(&abx.view(), &bcx.view()).view());

    //     let diff = (b * x).sum();
    //     let inner = (abx * bcx).sum();
    //     let mut angle = inner.acos();

    //     if angle.is_nan() {
    //         std::f64::consts::PI
    //     } else {
    //         if diff < 0.0 {
    //             angle = (2.0 * std::f64::consts::PI) - angle;
    //         }
    //         angle
    //     }
    // };
    //
    // angle

    let ab = xyzs_distance_over_sphere_radians(a, b);
    let bc = xyzs_distance_over_sphere_radians(b, c);
    let ca = xyzs_distance_over_sphere_radians(c, a);

    let radians = if ab < tolerance || bc < tolerance || ca < tolerance {
        // if any side of the triangle is negligibly small
        0.0
    } else {
        ((ca.cos() - (bc.cos() * ab.cos())) / (bc.sin() * ab.sin())).acos()
    };

    // check if B is directly between A and B
    if radians.is_nan() {
        if (ab + bc - ca) < tolerance {
            std::f64::consts::PI
        } else {
            0.0
        }
    } else {
        radians
    }
}

/// whether the angle formed between A->B->C is a clockwise turn
fn xyz_two_arc_is_clockwise(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    xyz_sum(&xyz_mul_xyz(
        &xyz_cross(&xyz_sub_xyz(a, b), &xyz_sub_xyz(c, b)),
        b,
    )) > 0.0
}

/// whether the three xyz points exist on the same great-circle arc
pub fn xyzs_collinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    if xyz_eq(a, b) || xyz_eq(b, c) {
        true
    } else {
        // let area = spherical_triangle_area(a, b, c);
        // area.is_nan() || area < tolerance

        let abc = xyz_two_arc_angle_radians(a, b, c);
        let cab = xyz_two_arc_angle_radians(c, a, b);
        let bca = xyz_two_arc_angle_radians(b, c, a);

        let tolerance = 3e-11;
        abc < tolerance
            || cab < tolerance
            || bca < tolerance
            || (abc - std::f64::consts::PI).abs() < tolerance
            || (cab - std::f64::consts::PI).abs() < tolerance
            || (bca - std::f64::consts::PI).abs() < tolerance

        // let left = arc_length(&a, &p);
        // let right = arc_length(&p, &b);
        // let total = arc_length(&a, &b);

        // let tolerance = 3e-11;
        // if left + right - total < tolerance {
        //     // ensure angle is flat
        //     if angle(&a, &point.xyz.view(), &b, false) - std::f64::consts::PI < tolerance {
        //         return true;
        //     }
        // }
    }
}

pub fn point_within_kdtree(xyz: &[f64; 3], kdtree: &ImmutableKdTree<f64, 3>) -> bool {
    // take advantage of the kdtree's distance function in 3D space
    kdtree.nearest_one::<SquaredEuclidean>(xyz).distance < 3e-11
}

/// 3D Cartesian vector representing a point on the unit sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct SphericalPoint {
    pub xyz: [f64; 3],
}

impl From<[f64; 3]> for SphericalPoint {
    fn from(xyz: [f64; 3]) -> Self {
        let length = xyz_length(&xyz);
        let xyz = if length < 2e-11 {
            xyz
        } else {
            [xyz[0] / length, xyz[1] / length, xyz[2] / length]
        };
        Self { xyz }
    }
}

impl From<&(f64, f64, f64)> for SphericalPoint {
    fn from(xyz: &(f64, f64, f64)) -> Self {
        Self::from([xyz.0, xyz.1, xyz.2])
    }
}

impl TryFrom<&Vec<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(xyz: &Vec<f64>) -> Result<Self, Self::Error> {
        let length = xyz.len();
        if length != 3 {
            Err(format!("3D vector should have length 3, not {length}"))
        } else {
            Ok(Self::from([xyz[0], xyz[1], xyz[2]]))
        }
    }
}

impl TryFrom<&Array1<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(xyz: &Array1<f64>) -> Result<Self, Self::Error> {
        let length = xyz.len();
        if length != 3 {
            Err(format!("3D vector should have length 3, not {length}"))
        } else {
            Ok(Self::from([xyz[0], xyz[1], xyz[2]]))
        }
    }
}

impl<'a> TryFrom<&ArrayView1<'a, f64>> for SphericalPoint {
    type Error = String;

    fn try_from(xyz: &ArrayView1<'a, f64>) -> Result<Self, Self::Error> {
        let length = xyz.len();
        if length != 3 {
            Err(format!("3D vector should have length 3, not {length}"))
        } else {
            Ok(Self::from([xyz[0], xyz[1], xyz[2]]))
        }
    }
}

impl From<SphericalPoint> for [f64; 3] {
    fn from(point: SphericalPoint) -> Self {
        point.xyz
    }
}

impl<'a> From<&'a SphericalPoint> for &'a [f64; 3] {
    fn from(point: &'a SphericalPoint) -> Self {
        &point.xyz
    }
}

impl From<SphericalPoint> for Array1<f64> {
    fn from(point: SphericalPoint) -> Self {
        array![point.xyz[0], point.xyz[1], point.xyz[2]]
    }
}

impl From<SphericalPoint> for Vec<f64> {
    fn from(point: SphericalPoint) -> Self {
        point.xyz.to_vec()
    }
}

impl Add<Self> for &SphericalPoint {
    type Output = SphericalPoint;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::from(xyz_add_xyz(&self.xyz, &rhs.xyz))
    }
}

impl Sub<Self> for &SphericalPoint {
    type Output = SphericalPoint;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::from(xyz_sub_xyz(&self.xyz, &rhs.xyz))
    }
}

impl Mul<Self> for &SphericalPoint {
    type Output = SphericalPoint;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output::from(xyz_mul_xyz(&self.xyz, &rhs.xyz))
    }
}

impl Div<Self> for &SphericalPoint {
    type Output = SphericalPoint;

    fn div(self, rhs: Self) -> Self::Output {
        Self::Output::from(xyz_div_xyz(&self.xyz, &rhs.xyz))
    }
}

impl AddAssign<&Self> for SphericalPoint {
    fn add_assign(&mut self, rhs: &Self) {
        self.xyz[0] += rhs.xyz[0];
        self.xyz[1] += rhs.xyz[1];
        self.xyz[2] += rhs.xyz[2];
    }
}

impl SubAssign<&Self> for SphericalPoint {
    fn sub_assign(&mut self, rhs: &Self) {
        self.xyz[0] -= rhs.xyz[0];
        self.xyz[1] -= rhs.xyz[1];
        self.xyz[2] -= rhs.xyz[2];
    }
}

impl MulAssign<&Self> for SphericalPoint {
    fn mul_assign(&mut self, rhs: &Self) {
        self.xyz[0] *= rhs.xyz[0];
        self.xyz[1] *= rhs.xyz[1];
        self.xyz[2] *= rhs.xyz[2];
    }
}

impl DivAssign<&Self> for SphericalPoint {
    fn div_assign(&mut self, rhs: &Self) {
        self.xyz[0] /= rhs.xyz[0];
        self.xyz[1] /= rhs.xyz[1];
        self.xyz[2] /= rhs.xyz[2];
    }
}

impl Add<&f64> for &SphericalPoint {
    type Output = SphericalPoint;

    fn add(self, rhs: &f64) -> Self::Output {
        Self::Output::from(xyz_add_f64(&self.xyz, rhs))
    }
}

impl Sub<&f64> for &SphericalPoint {
    type Output = SphericalPoint;

    fn sub(self, rhs: &f64) -> Self::Output {
        Self::Output::from(xyz_sub_f64(&self.xyz, rhs))
    }
}

impl Mul<&f64> for &SphericalPoint {
    type Output = SphericalPoint;

    fn mul(self, rhs: &f64) -> Self::Output {
        Self::Output::from(xyz_mul_f64(&self.xyz, rhs))
    }
}

impl Div<&f64> for &SphericalPoint {
    type Output = SphericalPoint;

    fn div(self, rhs: &f64) -> Self::Output {
        Self::Output::from(xyz_div_f64(&self.xyz, rhs))
    }
}

impl AddAssign<&f64> for SphericalPoint {
    fn add_assign(&mut self, rhs: &f64) {
        self.xyz[0] += rhs;
        self.xyz[1] += rhs;
        self.xyz[2] += rhs;
    }
}

impl SubAssign<&f64> for SphericalPoint {
    fn sub_assign(&mut self, rhs: &f64) {
        self.xyz[0] -= rhs;
        self.xyz[1] -= rhs;
        self.xyz[2] -= rhs;
    }
}

impl MulAssign<&f64> for SphericalPoint {
    fn mul_assign(&mut self, rhs: &f64) {
        self.xyz[0] *= rhs;
        self.xyz[1] *= rhs;
        self.xyz[2] *= rhs;
    }
}

impl DivAssign<&f64> for SphericalPoint {
    fn div_assign(&mut self, rhs: &f64) {
        self.xyz[0] /= rhs;
        self.xyz[1] /= rhs;
        self.xyz[2] /= rhs;
    }
}

impl Neg for &SphericalPoint {
    type Output = SphericalPoint;

    fn neg(self) -> Self::Output {
        Self::Output::from([-self.xyz[0], -self.xyz[1], -self.xyz[2]])
    }
}

impl SphericalPoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self::from([x, y, z])
    }

    pub fn from_lonlat(lonlat: &[f64; 2]) -> Self {
        Self::from(lonlat_to_xyz(lonlat))
    }

    pub fn to_lonlat(&self) -> [f64; 2] {
        xyz_to_lonlat(&self.xyz)
    }

    /// create n number of points equally spaced on an arc between this point and another point
    pub fn interpolate_between(
        &self,
        other: &Self,
        n: usize,
    ) -> Result<MultiSphericalPoint, String> {
        MultiSphericalPoint::try_from(crate::arcstring::interpolate_points_along_arc(
            (&self.xyz, &other.xyz),
            n,
        )?)
    }

    pub fn two_arc_angle(&self, a: &SphericalPoint, b: &SphericalPoint) -> f64 {
        xyz_two_arc_angle_radians(&a.xyz, &self.xyz, &b.xyz).to_degrees()
    }

    /// whether this point shares a line with two other points
    pub fn collinear(&self, a: &SphericalPoint, b: &SphericalPoint) -> bool {
        xyzs_collinear(&a.xyz, &self.xyz, &b.xyz)
    }

    pub fn clockwise_turn(&self, a: &Self, b: &Self) -> bool {
        xyz_two_arc_is_clockwise(&a.xyz, &self.xyz, &b.xyz)
    }

    pub fn vector_length(&self) -> f64 {
        xyz_length(&self.xyz)
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        Self::from(xyz_cross(&self.xyz, &other.xyz))
    }

    pub fn vector_dot(&self, other: &Self) -> f64 {
        xyz_dot(&self.xyz, &other.xyz)
    }

    /// rotate this xyz vector by theta angle around another xyz vector
    pub fn vector_rotate_around(&self, other: &Self, theta: &f64) -> Self {
        Self::from(xyz_rotate_around(&self.xyz, &other.xyz, theta))
    }
}

impl Display for SphericalPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalPoint({:?})", self.xyz)
    }
}

impl PartialEq for SphericalPoint {
    fn eq(&self, other: &SphericalPoint) -> bool {
        xyz_eq(&self.xyz, &other.xyz)
    }
}

impl Geometry for SphericalPoint {
    fn vertices(&self) -> MultiSphericalPoint {
        self.to_owned().into()
    }

    fn boundary(&self) -> Option<SphericalPoint> {
        None
    }

    fn representative(&self) -> SphericalPoint {
        self.to_owned()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.to_owned()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        None
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        0.
    }
}

impl GeometricPredicates<Self> for SphericalPoint {
    fn intersects(&self, other: &Self) -> bool {
        self == other
    }

    fn touches(&self, other: &Self) -> bool {
        self == other
    }

    fn crosses(&self, _: &Self) -> bool {
        false
    }

    fn within(&self, _: &Self) -> bool {
        false
    }

    fn contains(&self, _: &Self) -> bool {
        false
    }

    fn overlaps(&self, other: &Self) -> bool {
        self == other
    }

    fn covers(&self, other: &Self) -> bool {
        self == other
    }
}

impl GeometricOperations<Self> for SphericalPoint {
    fn union(&self, other: &Self) -> Option<MultiSphericalPoint> {
        MultiSphericalPoint::try_from(vec![self.xyz, other.xyz]).ok()
    }

    fn distance(&self, other: &Self) -> f64 {
        xyzs_distance_over_sphere_radians(&self.xyz, &other.xyz).to_degrees()
    }

    fn intersection(&self, other: &Self) -> Option<SphericalPoint> {
        if self == other {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, _: &Self) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

impl GeometricPredicates<MultiSphericalPoint> for SphericalPoint {
    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn within(&self, other: &MultiSphericalPoint) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn overlaps(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        if other.len() == 1 {
            xyz_eq(&self.xyz, &other.xyzs[0])
        } else {
            false
        }
    }
}

impl GeometricOperations<MultiSphericalPoint> for SphericalPoint {
    fn union(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        Some(other + self)
    }

    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, _: &MultiSphericalPoint) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

impl GeometricPredicates<crate::arcstring::ArcString> for SphericalPoint {
    fn intersects(&self, other: &crate::arcstring::ArcString) -> bool {
        self.within(other)
    }

    fn touches(&self, other: &crate::arcstring::ArcString) -> bool {
        if let Some(boundary) = other.boundary() {
            self.touches(&boundary)
        } else {
            false
        }
    }

    fn crosses(&self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn within(&self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn covers(&self, _: &crate::arcstring::ArcString) -> bool {
        false
    }
}

impl GeometricOperations<crate::arcstring::ArcString> for SphericalPoint {
    fn union(&self, _: &crate::arcstring::ArcString) -> Option<MultiSphericalPoint> {
        None
    }

    fn distance(&self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::arcstring::ArcString) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, _: &crate::arcstring::ArcString) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

impl GeometricPredicates<crate::arcstring::MultiArcString> for SphericalPoint {
    fn intersects(&self, other: &crate::arcstring::MultiArcString) -> bool {
        self.within(other)
    }

    fn touches(&self, other: &crate::arcstring::MultiArcString) -> bool {
        self.within(other)
    }

    fn crosses(&self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn within(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn covers(&self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString> for SphericalPoint {
    fn union(&self, _: &crate::arcstring::MultiArcString) -> Option<MultiSphericalPoint> {
        None
    }

    fn distance(&self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, _: &crate::arcstring::MultiArcString) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

impl GeometricPredicates<crate::sphericalpolygon::SphericalPolygon> for SphericalPoint {
    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.within(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(&other.boundary)
    }

    fn crosses(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon> for SphericalPoint {
    fn union(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> Option<MultiSphericalPoint> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<SphericalPoint> {
        if self.within(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(
        &self,
        _: &crate::sphericalpolygon::SphericalPolygon,
    ) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

impl GeometricPredicates<crate::sphericalpolygon::MultiSphericalPolygon> for SphericalPoint {
    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.within(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(&other.boundary().unwrap())
    }

    fn crosses(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon> for SphericalPoint {
    fn union(
        &self,
        _: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<SphericalPoint> {
        if self.intersects(other) {
            Some(self.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(
        &self,
        _: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> MultiSphericalPoint {
        MultiSphericalPoint::from(self.to_owned())
    }
}

/// xyz vectors representing points on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiSphericalPoint {
    pub xyzs: Vec<[f64; 3]>,
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl TryFrom<Vec<[f64; 3]>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(xyzs: Vec<[f64; 3]>) -> Result<Self, Self::Error> {
        if xyzs.is_empty() {
            Err(String::from("no points provided"))
        } else {
            let xyzs: Vec<[f64; 3]> = xyzs
                .into_iter()
                .map(|xyz| {
                    let length = xyz_length(&xyz);
                    if length < 3e-11 {
                        xyz
                    } else {
                        [xyz[0] / length, xyz[1] / length, xyz[2] / length]
                    }
                })
                .collect();
            let kdtree = ImmutableKdTree::<f64, 3>::from(xyzs.as_slice());
            Ok(Self { xyzs, kdtree })
        }
    }
}

impl From<&Vec<MultiSphericalPoint>> for MultiSphericalPoint {
    fn from(multipoints: &Vec<MultiSphericalPoint>) -> Self {
        let mut points = multipoints[0].xyzs.to_owned();
        for multipoint in multipoints.iter().skip(1) {
            for point in &multipoint.xyzs {
                if !points.contains(point) {
                    points.push(point.to_owned());
                }
            }
        }

        // we can assume that existing multipoints are at least length 1
        Self::try_from(points).unwrap()
    }
}

impl From<SphericalPoint> for MultiSphericalPoint {
    fn from(point: SphericalPoint) -> Self {
        Self::try_from(vec![point.xyz]).unwrap()
    }
}

impl TryFrom<Vec<SphericalPoint>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(points: Vec<SphericalPoint>) -> Result<Self, String> {
        Self::try_from(
            points
                .iter()
                .map(|point| point.xyz)
                .collect::<Vec<[f64; 3]>>(),
        )
    }
}

impl TryFrom<Array2<f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(xyzs: Array2<f64>) -> Result<Self, Self::Error> {
        let columns = xyzs.shape()[1];
        if columns != 3 {
            Err(format!(
                "array of 3D vectors should have shape Nx3, not Nx{columns}",
            ))
        } else {
            Self::try_from(
                xyzs.rows()
                    .into_iter()
                    .map(|xyz| [xyz[0], xyz[1], xyz[2]])
                    .collect::<Vec<[f64; 3]>>(),
            )
        }
    }
}

impl TryFrom<&Vec<(f64, f64, f64)>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(xyzs: &Vec<(f64, f64, f64)>) -> Result<Self, String> {
        Self::try_from(
            xyzs.iter()
                .map(|xyz| [xyz.0, xyz.1, xyz.2])
                .collect::<Vec<[f64; 3]>>(),
        )
    }
}

impl TryFrom<&Vec<Vec<f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: &Vec<Vec<f64>>) -> Result<Self, Self::Error> {
        let mut xyzs = vec![];
        for point in list {
            let length = point.len();
            if length != 3 {
                return Err(format!("3D vector should have length 3, not {length}",));
            } else {
                xyzs.push([point[0], point[1], point[2]]);
            }
        }
        Self::try_from(xyzs)
    }
}

impl<'a> TryFrom<&Vec<ArrayView1<'a, f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: &Vec<ArrayView1<'a, f64>>) -> Result<Self, Self::Error> {
        let mut xyzs = vec![];
        for point in list {
            let length = point.len();
            if length != 3 {
                return Err(format!("3D vector should have length 3, not {length}",));
            } else {
                xyzs.push([point[0], point[1], point[2]]);
            }
        }
        Self::try_from(xyzs)
    }
}

impl TryFrom<Vec<f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(list: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(
            Array2::from_shape_vec((list.len() / 3, 3), list).map_err(|err| format!("{err:?}"))?,
        )
    }
}

impl<'a> TryFrom<&ArrayView1<'a, f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(xyz: &ArrayView1<'a, f64>) -> Result<Self, Self::Error> {
        Self::try_from(
            xyz.to_shape((xyz.len() / 3, 3))
                .map_err(|err| format!("{err:?}"))?
                .to_owned(),
        )
    }
}

impl From<MultiSphericalPoint> for Vec<SphericalPoint> {
    fn from(points: MultiSphericalPoint) -> Self {
        points
            .xyzs
            .into_iter()
            .map(|xyz| SphericalPoint { xyz })
            .collect()
    }
}

impl From<&MultiSphericalPoint> for Array2<f64> {
    fn from(points: &MultiSphericalPoint) -> Self {
        let mut xyzs = Array2::uninit((points.len(), 3));
        for (index, row) in xyzs.axis_iter_mut(Axis(0)).enumerate() {
            let xyz = points.xyzs[index];
            array![xyz[0], xyz[1], xyz[2]].assign_to(row);
        }
        unsafe { xyzs.assume_init() }
    }
}

impl From<MultiSphericalPoint> for Vec<[f64; 3]> {
    fn from(points: MultiSphericalPoint) -> Self {
        points.xyzs
    }
}

impl MultiSphericalPoint {
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
    pub fn try_from_lonlats(lonlats: &[[f64; 2]]) -> Result<Self, String> {
        Self::try_from(lonlats.iter().map(lonlat_to_xyz).collect::<Vec<[f64; 3]>>())
    }

    /// retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point
    pub fn nearest(&self, other: &SphericalPoint) -> (SphericalPoint, f64) {
        // since the kdtree is over normalized vectors, the nearest vector in 3D space is also the nearest in angular distance
        let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&[
            other.xyz[0],
            other.xyz[1],
            other.xyz[2],
        ]);

        (
            SphericalPoint::from(self.xyzs[nearest.item as usize]),
            nearest.distance,
        )
    }

    fn recreate_kdtree(&mut self) {
        self.kdtree = ImmutableKdTree::<f64, 3>::from(self.xyzs.as_slice());
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
    pub fn to_lonlats(&self) -> Vec<[f64; 2]> {
        self.xyzs.iter().map(xyz_to_lonlat).collect()
    }

    /// lengths of the underlying xyz vectors
    ///
    ///     r = sqrt(x^2 + y^2 + z^2)
    pub fn vectors_lengths(&self) -> Vec<f64> {
        self.xyzs.iter().map(xyz_length).collect()
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        Self::try_from(
            self.xyzs
                .iter()
                .zip(other.xyzs.iter())
                .map(|(a, b)| xyz_cross(a, b))
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
    }

    /// rotate the underlying vectors by theta angle around other vectors
    pub fn vectors_rotate_around(&self, other: &Self, theta: f64) -> Self {
        Self::try_from(
            self.xyzs
                .iter()
                .zip(other.xyzs.iter())
                .map(|(a, b)| xyz_rotate_around(a, b, &theta))
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
    }
}

impl Sum for MultiSphericalPoint {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let multipoints: Vec<MultiSphericalPoint> = iter.collect();
        Self::from(&multipoints)
    }
}

impl Display for MultiSphericalPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiSphericalPoint({:?})", self.xyzs)
    }
}

impl PartialEq for MultiSphericalPoint {
    fn eq(&self, other: &MultiSphericalPoint) -> bool {
        let (less, more) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        for more_xyz in &less.xyzs {
            let mut found = false;
            for less_xyz in &more.xyzs {
                if xyz_eq(more_xyz, less_xyz) {
                    found = true;
                    break;
                }
            }

            if !found {
                return false;
            }
        }

        true
    }
}

impl Add<Self> for &MultiSphericalPoint {
    type Output = MultiSphericalPoint;

    fn add(self, rhs: Self) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&Self> for MultiSphericalPoint {
    fn add_assign(&mut self, other: &Self) {
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

impl Geometry for MultiSphericalPoint {
    fn vertices(&self) -> MultiSphericalPoint {
        self.to_owned()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        None
    }

    fn representative(&self) -> SphericalPoint {
        SphericalPoint::from(self.xyzs[0])
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        let mean = Array2::<f64>::from(self).mean_axis(Axis(0)).unwrap();
        SphericalPoint::from([mean[0], mean[1], mean[2]])
    }

    /// This code implements Andrew's monotone chain algorithm, which is a simple
    /// variant of the Graham scan.  Rather than sorting by x-coordinate, instead
    /// we sort the points in CCW order around an origin O such that all points
    /// are guaranteed to be on one side of some geodesic through O.  This
    /// ensures that as we scan through the points, each new point can only
    /// belong at the end of the chain (i.e., the chain is monotone in terms of
    /// the angle around O from the starting point).
    ///
    /// References
    /// ----------
    /// - https://github.com/google/s2geometry/blob/master/src/s2/s2convex_hull_query.cc#L123
    /// - https://www.researchgate.net/profile/Jayaram-Ma-2/publication/303522254/figure/fig1/AS:365886075621376@1464245446409/Monotone-Chain-Algorithm-and-graphic-illustration.png
    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        if self.len() < 3 {
            return None;
        }

        // list of vertices on the convex hull
        let mut convex_hull_point_indices = vec![];

        // mean center of all points
        let centroid = self.centroid();

        // the farthest point from the mean center must be on the convex hull
        let num_candidates = std::num::NonZero::try_from(self.len() - 1).unwrap();
        let farthest_neighbor_index = self
            .kdtree
            .nearest_n::<SquaredEuclidean>(&centroid.xyz, num_candidates)
            .last()
            .unwrap()
            .item;
        convex_hull_point_indices.push(farthest_neighbor_index);

        // iterate enough times to test all points
        for _ in 0..self.len() {
            let working_end =
                self.xyzs[convex_hull_point_indices[convex_hull_point_indices.len() - 1] as usize];

            // query the kdtree for all points, sorting them by distance from the current working end of the convex hull
            let candidates = self
                .kdtree
                .nearest_n::<SquaredEuclidean>(&working_end, num_candidates);

            for candidate in &candidates {
                // skip candidates already on the convex hull...
                if !convex_hull_point_indices.contains(&candidate.item) {
                    let point = self.xyzs[candidate.item as usize];

                    // test another point to see if the candidate has a clockwise turn toward it
                    let mut no_clockwise: bool = true;
                    for test_point in &candidates {
                        if test_point.item != candidate.item {
                            // if the candidate point is on the edge, it shouldn't have a clockwise turn to any other point
                            if xyz_two_arc_is_clockwise(
                                &working_end,
                                &point,
                                &self.xyzs[test_point.item as usize],
                            ) {
                                no_clockwise = false;
                                break;
                            }
                        }
                    }

                    // if the candidate point has no clockwise turns to any other point, it must be on the convex hull
                    if no_clockwise {
                        convex_hull_point_indices.push(candidate.item);
                        break;
                    }
                }
            }

            // if the last point in the chain equals the first, the arcstring is closed
            if convex_hull_point_indices.len() > 2
                && convex_hull_point_indices[0]
                    == convex_hull_point_indices[convex_hull_point_indices.len() - 1]
            {
                break;
            }
        }

        crate::sphericalpolygon::SphericalPolygon::try_new(
            crate::arcstring::ArcString::try_from(
                MultiSphericalPoint::try_from(
                    convex_hull_point_indices
                        .iter()
                        .map(|index| self.xyzs[*index as usize])
                        .collect::<Vec<[f64; 3]>>(),
                )
                .unwrap(),
            )
            .unwrap(),
            Some(centroid),
        )
        .ok()
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        0.
    }
}

impl MultiGeometry<SphericalPoint> for MultiSphericalPoint {
    fn len(&self) -> usize {
        self.xyzs.len()
    }

    fn extend(&mut self, other: MultiSphericalPoint) {
        self.xyzs.extend(other.xyzs);
        self.recreate_kdtree();
    }

    fn push(&mut self, point: SphericalPoint) {
        self.xyzs.push(point.xyz);
    }
}

impl GeometricPredicates<SphericalPoint> for MultiSphericalPoint {
    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        point_within_kdtree(&other.xyz, &self.kdtree)
    }

    fn overlaps(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }
}

impl GeometricOperations<SphericalPoint, SphericalPoint> for MultiSphericalPoint {
    fn union(&self, other: &SphericalPoint) -> Option<Self> {
        Some(self + other)
    }

    fn distance(&self, other: &SphericalPoint) -> f64 {
        self.nearest(other).0.distance(other)
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self.contains(other) {
            Some(other.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, _: &SphericalPoint) -> Self {
        self.to_owned()
    }
}

impl GeometricPredicates<Self> for MultiSphericalPoint {
    fn intersects(&self, other: &Self) -> bool {
        self.touches(other)
    }

    fn touches(&self, other: &Self) -> bool {
        let (less, more) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        less.xyzs
            .iter()
            .any(|xyz| point_within_kdtree(xyz, &more.kdtree))
    }

    fn crosses(&self, other: &Self) -> bool {
        self.touches(other) && !self.within(other) && !self.contains(other)
    }

    fn within(&self, other: &Self) -> bool {
        if self.len() < other.len() {
            self.xyzs
                .iter()
                .all(|xyz| point_within_kdtree(xyz, &other.kdtree))
        } else {
            false
        }
    }

    fn contains(&self, other: &Self) -> bool {
        other.within(self)
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.touches(other)
    }

    fn covers(&self, other: &Self) -> bool {
        self.contains(other)
    }
}

impl GeometricOperations<Self, SphericalPoint> for MultiSphericalPoint {
    fn union(&self, other: &Self) -> Option<Self> {
        Some(self + other)
    }

    fn distance(&self, other: &Self) -> f64 {
        // find the shortest distance between any two points between this and the other set,
        // using the normalized 3D Cartesian distance (much faster than calculating angular distance)
        let (self_index, other_index, cartesian_distance) = self
            .xyzs
            .iter()
            .enumerate()
            .map(|(self_index, self_xyz)| {
                let nearest = other.kdtree.nearest_one::<SquaredEuclidean>(self_xyz);
                (self_index, nearest.item as usize, nearest.distance)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        if cartesian_distance < 3e-11 {
            0.0
        } else {
            // calculate the angular distance
            SphericalPoint::from(self.xyzs[self_index])
                .distance(&SphericalPoint::from(other.xyzs[other_index]))
        }
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let (less, more) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        Self::try_from(
            less.xyzs
                .iter()
                .filter_map(|xyz| {
                    if point_within_kdtree(xyz, &more.kdtree) {
                        Some(*xyz)
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
    }

    fn symmetric_difference(&self, _: &Self) -> Self {
        self.to_owned()
    }
}

impl GeometricPredicates<crate::arcstring::ArcString> for MultiSphericalPoint {
    fn intersects(&self, other: &crate::arcstring::ArcString) -> bool {
        self.touches(other)
    }

    fn touches(&self, other: &crate::arcstring::ArcString) -> bool {
        self.xyzs
            .iter()
            .any(|xyz| crate::arcstring::arcstring_contains_point(other, xyz))
    }

    fn crosses(&self, other: &crate::arcstring::ArcString) -> bool {
        self.touches(other) && !self.within(other)
    }

    fn within(&self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn covers(&self, _: &crate::arcstring::ArcString) -> bool {
        false
    }
}

impl GeometricOperations<crate::arcstring::ArcString, SphericalPoint> for MultiSphericalPoint {
    fn union(&self, _: &crate::arcstring::ArcString) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::arcstring::ArcString) -> Option<Self> {
        Self::try_from(
            self.xyzs
                .iter()
                .filter_map(|xyz| {
                    if crate::arcstring::arcstring_contains_point(other, xyz) {
                        Some(*xyz)
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
    }

    fn symmetric_difference(&self, _: &crate::arcstring::ArcString) -> Self {
        self.to_owned()
    }
}

impl GeometricPredicates<crate::arcstring::MultiArcString> for MultiSphericalPoint {
    fn intersects(&self, other: &crate::arcstring::MultiArcString) -> bool {
        self.touches(other)
    }

    fn touches(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.touches(self)
    }

    fn crosses(&self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn within(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn contains(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.within(self)
    }

    fn covers(&self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString, SphericalPoint> for MultiSphericalPoint {
    fn union(&self, _: &crate::arcstring::MultiArcString) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> Option<Self> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, _: &crate::arcstring::MultiArcString) -> Self {
        self.to_owned()
    }
}

impl GeometricPredicates<crate::sphericalpolygon::SphericalPolygon> for MultiSphericalPoint {
    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }

    fn crosses(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn contains(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon, SphericalPoint>
    for MultiSphericalPoint
{
    fn union(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> Option<Self> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> Self {
        self.to_owned()
    }
}

impl GeometricPredicates<crate::sphericalpolygon::MultiSphericalPolygon> for MultiSphericalPoint {
    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }

    fn crosses(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        // TODO: find a better algorithm than brute-force; perhaps we can keep a kdtree of centroids for multigeometries?
        self.xyzs.iter().all(|xyz| {
            other.polygons.par_iter().any(|polygon| {
                crate::sphericalpolygon::point_in_polygon_boundary(
                    xyz,
                    &polygon.interior_point.xyz,
                    &polygon.boundary.points.xyzs,
                )
            })
        })
    }

    fn contains(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon, SphericalPoint>
    for MultiSphericalPoint
{
    fn union(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> Option<Self> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> Self {
        self.to_owned()
    }
}
