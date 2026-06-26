use crate::geometry::{
    GeometricOperations, GeometricRelationships, Geometry, GeometryCollection, MultiGeometry,
    MultiGeometryUnaryOperations,
};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use std::{
    collections::HashMap,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2, ArrayView1, Axis, array};

pub fn linspace(x0: f64, xend: f64, n: usize) -> Vec<f64> {
    let dx = (xend - x0) / ((n - 1) as f64);
    let mut x = vec![x0; n];
    for i in 1..n {
        x[i] = x[i - 1] + dx;
    }
    x
}

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

/// convert 3D Cartesian point on the sphere to angular coordinates
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

/// rotate xyz vector by theta radians around another xyz vector
fn xyz_rotate_around(a: &[f64; 3], b: &[f64; 3], theta: &f64) -> [f64; 3] {
    let theta_sin = theta.sin();
    let theta_cos = theta.cos();

    xyz_add_xyz(
        &xyz_add_xyz(
            &xyz_mul_f64(
                &xyz_mul_xyz(&xyz_mul_xyz(&xyz_neg(b), &xyz_neg(a)), b),
                &(1.0 - theta_cos),
            ),
            &xyz_mul_f64(a, &theta_cos),
        ),
        &xyz_mul_f64(
            &[
                -b[2] * a[1] + b[1] * a[2],
                b[2] * a[0] - b[0] * a[2],
                -b[1] * a[0] - b[0] * a[1],
            ],
            &theta_sin,
        ),
    )
}

fn haversine_distance_over_sphere_radians(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    2.0 * (((b[1] - a[1]) / 2.0).sin().powi(2)
        + a[1].cos() * b[1].cos() * ((b[0] - a[0]) / 2.0).sin().powi(2))
    .sqrt()
    .asin()
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
pub fn arc_distance_over_sphere(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    if xyz_eq(a, b) {
        0.0
    } else {
        let dotted = xyz_dot(a, b);
        let distance = dotted.acos();
        if distance.is_nan() {
            let crossed = xyz_cross(a, b);
            // avoid domain issues of a.dot(b).acos()
            xyz_sum(&xyz_mul_xyz(&crossed, &crossed))
                .sqrt()
                .atan2(dotted)
        } else {
            distance
        }
    }
}

/// given three (X, Y, Z) vector points on the sphere:
///   - `a`
///   - `b` (this point)
///   - `c`
///
/// retrieves the turning angle, in radians, at `b` formed by arcs `ab` and `bc`
///
///     cos(ca) = cos(bc) * cos(ab) + sin(bc) * sin(ab) * cos(b)
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. p132. 1994. Academic Press. doi:10.5555/180895.180907
///   `pdf <https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover>`_
pub fn xyz_two_arc_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let tolerance = 2e-8;

    // let abx = xyz_cross(&a, &b);
    // let bcx = xyz_cross(&b, &c);
    //
    // if arc_distance_over_sphere_radians(a, c) < tolerance
    //     || xyz_length(&abx) < tolerance
    //     || xyz_length(&bcx) < tolerance
    // {
    //     0.0
    // } else {
    //     let x = xyz_cross(&abx, &bcx);
    //
    //     let diff = xyz_sum(&xyz_mul_xyz(b, &x));
    //     let radians = xyz_sum(&xyz_mul_xyz(&abx, &bcx)).acos();
    //
    //     if radians.is_nan() {
    //         std::f64::consts::PI
    //     } else if diff < 0.0 {
    //         (2.0 * std::f64::consts::PI) - radians
    //     } else {
    //         radians
    //     }
    // }

    let ab = arc_distance_over_sphere(a, b);
    let bc = arc_distance_over_sphere(b, c);
    let ca = arc_distance_over_sphere(c, a);

    if ca < tolerance {
        // if the opposite side of the triangle is negligibly small
        0.0
    } else if ab < tolerance || bc < tolerance {
        std::f64::consts::PI / 2.0
    } else {
        let radians = ((ca.cos() - (bc.cos() * ab.cos())) / (bc.sin() * ab.sin())).acos();

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
}

/// whether the angle formed between A->B->C is a clockwise turn
fn xyz_two_arc_is_clockwise(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    xyz_sum(&xyz_mul_xyz(
        &xyz_cross(&xyz_sub_xyz(a, b), &xyz_sub_xyz(c, b)),
        b,
    )) > 0.0
}

/// whether the three xyz points exist on the same great-circle arc
pub fn xyzs_colinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    if xyz_eq(a, b) || xyz_eq(b, c) {
        true
    } else {
        let tolerance = 2e-8;

        let abc = xyz_two_arc_angle(a, b, c);
        let cab = xyz_two_arc_angle(c, a, b);
        let bca = xyz_two_arc_angle(b, c, a);

        abc < tolerance
            || cab < tolerance
            || bca < tolerance
            || (abc - std::f64::consts::PI).abs() < tolerance
            || (cab - std::f64::consts::PI).abs() < tolerance
            || (bca - std::f64::consts::PI).abs() < tolerance
    }
}

pub fn point_within_kdtree(xyz: &[f64; 3], kdtree: &ImmutableKdTree<f64, 3>) -> bool {
    // take advantage of the kdtree's distance function in 3D space
    kdtree.nearest_one::<SquaredEuclidean>(xyz).distance < 3e-11
}

pub fn arc_interpolate_points(
    a: &[f64; 3],
    b: &[f64; 3],
    n: usize,
) -> Result<Vec<[f64; 3]>, String> {
    let n = if n < 2 { 2 } else { n };
    let omega = arc_distance_over_sphere(a, b);

    let mut offsets = linspace(0.0, 1.0, n);
    offsets = if omega == 0.0 {
        offsets
    } else {
        offsets
            .iter()
            .map(|offset| (offset * omega).sin() / omega.sin())
            .collect()
    };

    Ok(offsets
        .iter()
        .zip(offsets.iter().rev())
        .map(|(offset, inverted_offset)| {
            xyz_add_xyz(&xyz_mul_f64(a, inverted_offset), &xyz_mul_f64(b, offset))
        })
        .collect())
}

/// single point on the unit sphere, represented internally as a 3-dimensional Cartesian point (X, Y, Z) with origin at the center of the unit sphere
#[cfg_attr(feature = "py", pyclass(from_py_object, str))]
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

impl From<&[f64; 2]> for SphericalPoint {
    fn from(lonlat: &[f64; 2]) -> Self {
        Self::from(lonlat_to_xyz(lonlat))
    }
}

impl From<&(f64, f64)> for SphericalPoint {
    fn from(xyz: &(f64, f64)) -> Self {
        Self::from(lonlat_to_xyz(&[xyz.0, xyz.1]))
    }
}

impl TryFrom<&Vec<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(point: &Vec<f64>) -> Result<Self, Self::Error> {
        let length = point.len();
        if length == 3 {
            Ok(Self::from([point[0], point[1], point[2]]))
        } else if length == 2 {
            Ok(Self::from(&[point[0], point[1]]))
        } else {
            Err(format!("3D vector should have length 3, not {length}"))
        }
    }
}

#[cfg(feature = "ndarray")]
impl TryFrom<&Array1<f64>> for SphericalPoint {
    type Error = String;

    fn try_from(point: &Array1<f64>) -> Result<Self, Self::Error> {
        let length = point.len();
        if length == 3 {
            Ok(Self::from([point[0], point[1], point[2]]))
        } else if length == 2 {
            Ok(Self::from(&[point[0], point[1]]))
        } else {
            Err(format!("3D vector should have length 3, not {length}"))
        }
    }
}

#[cfg(feature = "ndarray")]
impl<'a> TryFrom<&ArrayView1<'a, f64>> for SphericalPoint {
    type Error = String;

    fn try_from(point: &ArrayView1<'a, f64>) -> Result<Self, Self::Error> {
        let length = point.len();
        if length == 3 {
            Ok(Self::from([point[0], point[1], point[2]]))
        } else if length == 2 {
            Ok(Self::from(&[point[0], point[1]]))
        } else {
            Err(format!("3D vector should have length 3, not {length}"))
        }
    }
}

impl<'a> From<&'a SphericalPoint> for [f64; 2] {
    fn from(point: &'a SphericalPoint) -> Self {
        xyz_to_lonlat(&point.xyz)
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

    pub fn antipode(&self) -> SphericalPoint {
        SphericalPoint {
            xyz: xyz_sub_f64(&self.xyz, &1.0),
        }
    }

    /// given three (X, Y, Z) vector points on the sphere `a`, `b` (this point), and `c`,
    /// retrieve the angle in radians at `b` formed by arcs `ab` and `bc`
    /// (the smaller angle irrespective of turn orientation)
    pub fn two_arc_angle(&self, start: &SphericalPoint, end: &SphericalPoint) -> f64 {
        xyz_two_arc_angle(&start.xyz, &self.xyz, &end.xyz).to_degrees()
    }

    /// whether this point lies on an arc between two other points
    pub fn colinear(&self, a: &SphericalPoint, b: &SphericalPoint) -> bool {
        xyzs_colinear(&a.xyz, &self.xyz, &b.xyz)
    }

    /// whether the angle formed between this point and two other points is a clockwise turn
    pub fn is_clockwise_turn(&self, start: &Self, end: &Self) -> bool {
        xyz_two_arc_is_clockwise(&start.xyz, &self.xyz, &end.xyz)
    }

    /// create n number of equally-spaced points on the arc between this point and another point
    pub fn interpolate_points(&self, end: &Self, n: usize) -> Result<MultiSphericalPoint, String> {
        MultiSphericalPoint::try_from(arc_interpolate_points(&self.xyz, &end.xyz, n)?)
    }

    /// length of the underlying xyz vector
    pub fn vector_length(&self) -> f64 {
        xyz_length(&self.xyz)
    }

    /// cross product of this xyz vector with another xyz vector
    pub fn vector_cross(&self, other: &Self) -> Self {
        Self::from(xyz_cross(&self.xyz, &other.xyz))
    }

    /// dot product of this xyz vector with another xyz vector
    pub fn vector_dot(&self, other: &Self) -> f64 {
        xyz_dot(&self.xyz, &other.xyz)
    }

    /// rotate this xyz vector by theta radians around another xyz vector
    pub fn vector_rotate_around(&self, other: &Self, theta: &f64) -> Self {
        Self::from(xyz_rotate_around(
            &self.xyz,
            &other.xyz,
            &theta.to_radians(),
        ))
    }

    /// arc to another point
    pub fn to(&self, other: &Self) -> crate::arcstring::ArcString {
        crate::arcstring::ArcString {
            points: MultiSphericalPoint::try_from(vec![self.xyz, other.xyz]).unwrap(),
            closed: false,
        }
    }
}

impl Display for SphericalPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalPoint({:?})", self.xyz)
    }
}

impl PartialEq for SphericalPoint {
    fn eq(&self, other: &Self) -> bool {
        xyz_eq(&self.xyz, &other.xyz)
    }
}

impl Geometry for SphericalPoint {
    fn vertices(&self) -> MultiSphericalPoint {
        self.to_owned().into()
    }

    fn boundary(&self) -> Option<Self> {
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

    fn to_wkt(&self, angular: bool) -> String {
        format!(
            "POINT ({})",
            if angular {
                xyz_to_lonlat(&self.xyz).to_vec()
            } else {
                self.xyz.to_vec()
            }
            .into_iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<String>>()
            .join(" ")
        )
    }
}

impl GeometricRelationships<Self> for SphericalPoint {
    fn distance(&self, other: &Self) -> f64 {
        arc_distance_over_sphere(&self.xyz, &other.xyz).to_degrees()
    }

    fn equals(&self, other: &Self) -> bool {
        self == other
    }

    fn covers(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn within(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn touches(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn intersects(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl GeometricRelationships<MultiSphericalPoint> for SphericalPoint {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &MultiSphericalPoint) -> bool {
        other.equals(self)
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn within(&self, other: &MultiSphericalPoint) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn overlaps(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.within(other)
    }
}

impl GeometricRelationships<crate::arcstring::ArcString> for SphericalPoint {
    fn distance(&self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn within(&self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &crate::arcstring::ArcString) -> bool {
        other.touches(self)
    }

    fn intersects(&self, other: &crate::arcstring::ArcString) -> bool {
        self.within(other)
    }
}

impl GeometricRelationships<crate::arcstring::MultiArcString> for SphericalPoint {
    fn distance(&self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn within(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn touches(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.touches(self)
    }

    fn intersects(&self, other: &crate::arcstring::MultiArcString) -> bool {
        self.within(other)
    }
}

impl GeometricRelationships<crate::sphericalpolygon::SphericalPolygon> for SphericalPoint {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.within(other)
    }
}

impl GeometricRelationships<crate::sphericalpolygon::MultiSphericalPolygon> for SphericalPoint {
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.within(other)
    }
}

impl GeometricOperations<Self> for SphericalPoint {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        (
            if self.intersects(other) {
                Some(MultiSphericalPoint::from(self.to_owned()))
            } else {
                None
            },
            None,
            None,
        )
    }

    fn difference(&self, other: &Self) -> Option<MultiSphericalPoint> {
        if self.intersects(other) {
            None
        } else {
            Some(MultiSphericalPoint::from(self.to_owned()))
        }
    }

    fn union(&self, other: &Self) -> GeometryCollection {
        (
            if self.equals(other) {
                Some(MultiSphericalPoint::from(self.to_owned()))
            } else {
                MultiSphericalPoint::try_from(vec![self.to_owned(), other.to_owned()]).ok()
            },
            None,
            None,
        )
    }
}

impl GeometricOperations<MultiSphericalPoint> for SphericalPoint {
    fn intersection(&self, other: &MultiSphericalPoint) -> GeometryCollection {
        other.intersection(self)
    }

    fn difference(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        if self.within(other) {
            None
        } else {
            Some(MultiSphericalPoint::from(self.to_owned()))
        }
    }

    fn union(&self, other: &MultiSphericalPoint) -> GeometryCollection {
        other.union(self)
    }
}

impl GeometricOperations<crate::arcstring::ArcString> for SphericalPoint {
    fn intersection(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::ArcString) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString> for SphericalPoint {
    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::MultiArcString) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon> for SphericalPoint {
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon> for SphericalPoint {
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

/// collection of points on the sphere
#[cfg_attr(feature = "py", pyclass(from_py_object))]
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

impl TryFrom<&Vec<[f64; 2]>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(lonlats: &Vec<[f64; 2]>) -> Result<Self, Self::Error> {
        if lonlats.is_empty() {
            Err(String::from("no points provided"))
        } else {
            Self::try_from(lonlats.iter().map(lonlat_to_xyz).collect::<Vec<[f64; 3]>>())
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

#[cfg(feature = "ndarray")]
impl TryFrom<Array2<f64>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(points: Array2<f64>) -> Result<Self, Self::Error> {
        let columns = points.shape()[1];
        if columns == 3 {
            Self::try_from(
                points
                    .rows()
                    .into_iter()
                    .map(|xyz| [xyz[0], xyz[1], xyz[2]])
                    .collect::<Vec<[f64; 3]>>(),
            )
        } else if columns == 2 {
            Self::try_from(
                points
                    .rows()
                    .into_iter()
                    .map(|lonlat| lonlat_to_xyz(&[lonlat[0], lonlat[1]]))
                    .collect::<Vec<[f64; 3]>>(),
            )
        } else {
            Err(format!(
                "array of 3D vectors should have shape Nx3, not Nx{columns}",
            ))
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

impl TryFrom<&Vec<(f64, f64)>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(lonlats: &Vec<(f64, f64)>) -> Result<Self, String> {
        Self::try_from(
            lonlats
                .iter()
                .map(|lonlat| lonlat_to_xyz(&[lonlat.0, lonlat.1]))
                .collect::<Vec<[f64; 3]>>(),
        )
    }
}

impl TryFrom<&Vec<Vec<f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(points: &Vec<Vec<f64>>) -> Result<Self, Self::Error> {
        let mut xyzs = vec![];
        for point in points {
            let length = point.len();
            if length == 3 {
                xyzs.push([point[0], point[1], point[2]]);
            } else if length == 2 {
                xyzs.push(lonlat_to_xyz(&[point[0], point[1]]));
            } else {
                return Err(format!("3D vector should have length 3, not {length}"));
            }
        }
        Self::try_from(xyzs)
    }
}

#[cfg(feature = "ndarray")]
impl<'a> TryFrom<&Vec<ArrayView1<'a, f64>>> for MultiSphericalPoint {
    type Error = String;

    fn try_from(points: &Vec<ArrayView1<'a, f64>>) -> Result<Self, Self::Error> {
        let mut xyzs = vec![];
        for point in points {
            let length = point.len();
            if length == 3 {
                xyzs.push([point[0], point[1], point[2]]);
            } else if length == 2 {
                xyzs.push(lonlat_to_xyz(&[point[0], point[1]]));
            } else {
                return Err(format!("3D vector should have length 3, not {length}",));
            }
        }
        Self::try_from(xyzs)
    }
}

impl From<MultiSphericalPoint> for Vec<SphericalPoint> {
    fn from(points: MultiSphericalPoint) -> Self {
        points.xyzs.into_iter().map(SphericalPoint::from).collect()
    }
}

#[cfg(feature = "ndarray")]
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

impl From<&MultiSphericalPoint> for Vec<[f64; 2]> {
    fn from(points: &MultiSphericalPoint) -> Self {
        points.xyzs.iter().map(xyz_to_lonlat).collect()
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
        self.into()
    }

    /// retrieve the nearest of these points to the given point, along with the normalized 3D Cartesian distance to that point across the unit sphere
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

    /// lengths of the underlying xyz vectors
    pub fn vectors_lengths(&self) -> Vec<f64> {
        self.xyzs.iter().map(xyz_length).collect()
    }

    fn recreate_kdtree(&mut self) {
        self.kdtree = ImmutableKdTree::<f64, 3>::from(self.xyzs.as_slice());
    }

    fn unique(&self) -> HashMap<usize, Vec<usize>> {
        let mut unique = HashMap::<usize, Vec<usize>>::new();
        for (xyz_index, xyz) in self.xyzs.iter().enumerate() {
            if unique
                .values()
                .any(|duplicates| duplicates.contains(&xyz_index))
            {
                continue;
            }

            let close = self.kdtree.within_unsorted::<SquaredEuclidean>(xyz, 3e-11);

            if !unique.contains_key(&xyz_index) {
                unique.insert(xyz_index, vec![]);
            }
            if let Some(duplicates) = unique.get_mut(&xyz_index) {
                duplicates.extend(close.iter().filter_map(|dup| {
                    if dup.item as usize == xyz_index {
                        None
                    } else {
                        Some(dup.item as usize)
                    }
                }));
            }
        }

        unique
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
        write!(f, "{self}")
    }
}

impl PartialEq for MultiSphericalPoint {
    fn eq(&self, other: &Self) -> bool {
        let (shorter, longer) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        longer
            .xyzs
            .iter()
            .all(|xyz| point_within_kdtree(xyz, &shorter.kdtree))
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

    fn boundary(&self) -> Option<Self> {
        None
    }

    fn representative(&self) -> SphericalPoint {
        SphericalPoint::from(self.xyzs[0])
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        SphericalPoint::from(crate::sphericalpoint::xyzs_mean(&self.xyzs))
    }

    /// Smallest convex polygon containing these points.
    ///
    /// Implements Andrew's monotone chain algorithm.
    ///
    /// References
    /// ----------
    /// - M.A, Jayaram & Fleyeh, Hasan. (2016). Convex Hulls in Image Processing: A Scoping Review. American Journal of Intelligent Systems. 2016. 48-58. 10.5923/j.ajis.20160602.03. `pdf <https://www.researchgate.net/profile/Jayaram-Ma-2/publication/303522254>`_.
    /// - `s2convex_hull_query <https://github.com/google/s2geometry/blob/master/src/s2/s2convex_hull_query.cc#L123>`_
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

    fn to_wkt(&self, angular: bool) -> String {
        format!(
            "MULTIPOINT ({})",
            if angular {
                self.to_lonlats()
                    .into_iter()
                    .map(|lonlat| lonlat.to_vec())
                    .collect::<Vec<Vec<f64>>>()
            } else {
                self.xyzs
                    .iter()
                    .map(|xyz| xyz.to_vec())
                    .collect::<Vec<Vec<f64>>>()
            }
            .into_iter()
            .map(|point| point
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<String>>()
                .join(" "))
            .collect::<Vec<String>>()
            .join(", ")
        )
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
        self.recreate_kdtree();
    }
}

impl MultiGeometryUnaryOperations<SphericalPoint> for MultiSphericalPoint {
    fn unary_union(&self) -> Self {
        let unique = self.unique();
        Self::try_from(
            unique
                .keys()
                .map(|index| self.xyzs[*index])
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
    }

    fn unary_intersection(&self) -> Option<Self> {
        let unique = self.unique();
        Self::try_from(
            self.xyzs
                .iter()
                .enumerate()
                .filter_map(|(index, xyz)| {
                    if let Some(duplicates) = unique.get(&index)
                        && !duplicates.is_empty()
                    {
                        Some(xyz.to_owned())
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
    }

    fn unary_symmetric_difference(&self) -> Option<Self> {
        let unique = self.unique();
        Self::try_from(
            self.xyzs
                .iter()
                .enumerate()
                .filter_map(|(index, xyz)| {
                    if let Some(duplicates) = unique.get(&index)
                        && duplicates.is_empty()
                    {
                        Some(xyz.to_owned())
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
    }
}

impl GeometricRelationships<SphericalPoint> for MultiSphericalPoint {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        self.nearest(other).0.distance(other)
    }

    fn equals(&self, other: &SphericalPoint) -> bool {
        self.xyzs.iter().all(|xyz| xyz_eq(xyz, &other.xyz))
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        point_within_kdtree(&other.xyz, &self.kdtree)
    }

    fn within(&self, other: &SphericalPoint) -> bool {
        false
    }

    fn crosses(&self, other: &SphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn overlaps(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn disjoint(&self, other: &SphericalPoint) -> bool {
        !self.intersects(other)
    }
}

impl GeometricRelationships<Self> for MultiSphericalPoint {
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

    fn equals(&self, other: &Self) -> bool {
        self == other
    }

    fn covers(&self, other: &Self) -> bool {
        self.contains(other)
    }

    fn contains(&self, other: &Self) -> bool {
        other
            .xyzs
            .iter()
            .all(|xyz| point_within_kdtree(xyz, &self.kdtree))
    }

    fn within(&self, other: &Self) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &Self) -> bool {
        false
    }

    fn touches(&self, other: &Self) -> bool {
        let (shorter, longer) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        shorter
            .xyzs
            .iter()
            .any(|xyz| point_within_kdtree(xyz, &longer.kdtree))
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.touches(other) && self != other
    }

    fn intersects(&self, other: &Self) -> bool {
        self.touches(other)
    }

    fn disjoint(&self, other: &Self) -> bool {
        !self.intersects(other)
    }
}

impl GeometricRelationships<crate::arcstring::ArcString> for MultiSphericalPoint {
    fn distance(&self, other: &crate::arcstring::ArcString) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn covers(&self, other: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn contains(&self, other: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn within(&self, other: &crate::arcstring::ArcString) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &crate::arcstring::ArcString) -> bool {
        self.touches(other) && !self.within(other)
    }

    fn touches(&self, other: &crate::arcstring::ArcString) -> bool {
        other.touches(self)
    }

    fn overlaps(&self, other: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn intersects(&self, other: &crate::arcstring::ArcString) -> bool {
        other.intersects(self)
    }

    fn disjoint(&self, other: &crate::arcstring::ArcString) -> bool {
        !self.intersects(other)
    }
}

impl GeometricRelationships<crate::arcstring::MultiArcString> for MultiSphericalPoint {
    fn distance(&self, other: &crate::arcstring::MultiArcString) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn covers(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn contains(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn within(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn touches(&self, other: &crate::arcstring::MultiArcString) -> bool {
        other.touches(self)
    }

    fn overlaps(&self, other: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn intersects(&self, other: &crate::arcstring::MultiArcString) -> bool {
        self.touches(other)
    }

    fn disjoint(&self, other: &crate::arcstring::MultiArcString) -> bool {
        !self.intersects(other)
    }
}

impl GeometricRelationships<crate::sphericalpolygon::SphericalPolygon> for MultiSphericalPoint {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn covers(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn contains(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }

    fn overlaps(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other)
    }

    fn disjoint(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        !self.intersects(other)
    }
}

impl GeometricRelationships<crate::sphericalpolygon::MultiSphericalPolygon>
    for MultiSphericalPoint
{
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn covers(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn contains(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        // TODO: find a better algorithm than brute-force; perhaps we can keep a kdtree of centroids for multigeometries?
        self.xyzs.iter().all(|xyz| {
            other.polygons.iter().any(|polygon| {
                crate::arcstring::points_are_on_same_side(
                    xyz,
                    &polygon.interior_point.xyz,
                    &polygon.boundary,
                )
            })
        })
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }

    fn overlaps(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn disjoint(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<SphericalPoint, SphericalPoint> for MultiSphericalPoint {
    fn intersection(&self, other: &SphericalPoint) -> GeometryCollection {
        (
            if self.contains(other) {
                Some(MultiSphericalPoint::from(other.to_owned()))
            } else {
                None
            },
            None,
            None,
        )
    }

    fn difference(&self, other: &SphericalPoint) -> Option<Self> {
        let mut xyzs = self.xyzs.to_owned();
        let closest = self.kdtree.nearest_one::<SquaredEuclidean>(&other.xyz);
        if closest.distance < 3e-11 {
            xyzs.remove(closest.item as usize);
        }
        Self::try_from(xyzs).ok()
    }

    fn union(&self, other: &SphericalPoint) -> GeometryCollection {
        (
            Some(if self.contains(other) {
                self.unary_union()
            } else {
                (self + other).unary_union()
            }),
            None,
            None,
        )
    }
}

impl GeometricOperations<Self, SphericalPoint> for MultiSphericalPoint {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        let (shorter, longer) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        (
            Self::try_from(
                shorter
                    .xyzs
                    .iter()
                    .filter_map(|xyz| {
                        if point_within_kdtree(xyz, &longer.kdtree) {
                            Some(*xyz)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<[f64; 3]>>(),
            )
            .ok(),
            None,
            None,
        )
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        Self::try_from(
            self.xyzs
                .iter()
                .filter_map(|xyz| {
                    if point_within_kdtree(xyz, &other.kdtree) {
                        None
                    } else {
                        Some(*xyz)
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
    }

    fn union(&self, other: &Self) -> GeometryCollection {
        (
            MultiSphericalPoint::try_from({
                let mut xyzs = self.xyzs.to_owned();
                xyzs.extend(
                    other
                        .xyzs
                        .iter()
                        .filter_map(|xyz| {
                            if point_within_kdtree(xyz, &self.kdtree) {
                                None
                            } else {
                                Some(*xyz)
                            }
                        })
                        .collect::<Vec<[f64; 3]>>(),
                );
                xyzs
            })
            .ok(),
            None,
            None,
        )
    }
}

impl GeometricOperations<crate::arcstring::ArcString, SphericalPoint> for MultiSphericalPoint {
    fn intersection(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::ArcString) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString, SphericalPoint> for MultiSphericalPoint {
    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::MultiArcString) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon, SphericalPoint>
    for MultiSphericalPoint
{
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon, SphericalPoint>
    for MultiSphericalPoint
{
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiSphericalPoint> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}
