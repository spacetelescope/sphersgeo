use crate::{
    angularbounds::AngularBounds,
    geometry::{
        ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        MultiGeometryIntoIterator, MultiGeometryIterator,
    },
    sphericalpoint::{cross_vector, cross_vectors, MultiSphericalPoint, SphericalPoint},
    sphericalpolygon::{spherical_triangle_area, MultiSphericalPolygon, SphericalPolygon},
};
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{collections::VecDeque, iter::Sum};

pub fn interpolate_points_along_vector_arc(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    n: usize,
) -> Result<Array2<f64>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::linspace(0.0, 1.0, n);
    let t = t.to_shape((n, 1)).unwrap();
    let omega = vector_arc_length(a, b);

    if a.len() == b.len() {
        if a.len() == 3 && b.len() == 3 {
            let offsets = if omega == 0.0 {
                t.to_owned()
            } else {
                (t * omega).sin() / omega.sin()
            };
            let mut inverted_offsets = offsets.to_owned();
            inverted_offsets.invert_axis(Axis(0));

            Ok(concatenate(
                Axis(0),
                &[
                    (inverted_offsets * a + offsets * b).view(),
                    b.to_shape((1, 3)).unwrap().view(),
                ],
            )
            .unwrap())
        } else if a.len() == 2 && b.len() == 2 {
            Ok(concatenate(
                Axis(0),
                &[
                    (a * ((Zip::from(&t).par_map_collect(|t| 1.0 - t) * omega).sin()
                        / omega.sin())
                        + b * &((t * omega).sin() / omega.sin()).view())
                        .view(),
                    b.to_shape((1, 2)).unwrap().view(),
                ],
            )
            .unwrap())
        } else {
            Err(String::from(""))
        }
    } else {
        Err(String::from("shape must match"))
    }
}

/// given points A, B, and C on the unit sphere, retrieve the angle at B between arc AB and arc BC
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn vector_arc_angle(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
    degrees: bool,
) -> f64 {
    let tolerance = 3e-11;

    let ab = vector_arc_length(a, b);
    let bc = vector_arc_length(b, c);
    let ca = vector_arc_length(c, a);

    let angle = if ab > tolerance && bc > tolerance {
        (ca.cos() - bc.cos() * ab.cos()) / (bc.sin() * ab.sin()).acos()
    } else {
        (1.0 - ca.powi(2) / 2.0).acos()
    };

    if degrees {
        angle.to_degrees()
    } else {
        angle
    }
}

/// given points A, B, and C on the unit sphere, retrieve the angle at B between arc AB and arc BC
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn vector_arc_angles(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    c: &ArrayView2<f64>,
    degrees: bool,
) -> Array1<f64> {
    let abx = cross_vectors(a, b);
    let bcx = cross_vectors(b, c);
    let x = cross_vectors(&abx.view(), &bcx.view());

    let diff = (b * x).sum_axis(Axis(1));
    let mut inner = (abx * bcx).sum_axis(Axis(1));
    inner.par_mapv_inplace(|v| v.acos());

    let angles = stack(Axis(0), &[inner.view(), diff.view()])
        .unwrap()
        .map_axis(Axis(0), |v| {
            if v[1] < 0.0 {
                (2.0 * std::f64::consts::PI) - v[0]
            } else {
                v[0]
            }
        });

    if degrees {
        angles.to_degrees()
    } else {
        angles
    }
}

/// radians subtended by this arc on the sphere
///    Notes
///    -----
///    The length is computed using the following:
///
///       l = arccos(A ⋅ B) / r^2
pub fn vector_arc_length(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.dot(b).acos()
}

/// whether the three points exist on the same line
pub fn vectors_collinear(a: &ArrayView1<f64>, b: &ArrayView1<f64>, c: &ArrayView1<f64>) -> bool {
    let tolerance = 3e-11;
    spherical_triangle_area(a, b, c) < tolerance

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

/// Given the xyz vectors of the endpoints of two great circle arcs, find an intersecting point if it exists
///
/// References
/// ----------
/// - Method explained in an `e-mail
///     <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_
///     by Roger Stafford.
/// - https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
/// - Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
pub fn vector_arcs_intersection(
    a_0: &ArrayView1<f64>,
    a_1: &ArrayView1<f64>,
    b_0: &ArrayView1<f64>,
    b_1: &ArrayView1<f64>,
) -> Option<Array1<f64>> {
    let p = cross_vector(a_0, a_1);
    let q = cross_vector(b_0, b_1);

    let t = cross_vector(&p.view(), &q.view());

    let signs = array![
        -cross_vector(&a_0, &p.view()).dot(&t.view()),
        cross_vector(&a_1, &p.view()).dot(&t.view()),
        -cross_vector(&b_0, &q.view()).dot(&t.view()),
        cross_vector(&b_1, &q.view()).dot(&t.view()),
    ]
    .signum();

    let epsilon = 1e-6;
    if signs.iter().all(|sign| sign > &epsilon) {
        Some(t)
    } else if signs.iter().all(|sign| sign > &epsilon) {
        Some(-t)
    } else {
        None
    }
}

/// series of great circle arcs along the sphere
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct ArcString {
    pub points: MultiSphericalPoint,
}

impl From<MultiSphericalPoint> for ArcString {
    fn from(points: MultiSphericalPoint) -> Self {
        Self { points }
    }
}

impl Into<MultiSphericalPoint> for ArcString {
    fn into(self) -> MultiSphericalPoint {
        self.points
    }
}

impl Into<Vec<ArcString>> for &ArcString {
    fn into(self) -> Vec<ArcString> {
        let vectors = &self.points.xyz;
        let mut arcs = vec![];
        for index in 0..vectors.nrows() - 1 {
            arcs.push(ArcString {
                points: MultiSphericalPoint::try_from(
                    vectors.slice(s![index..index + 1, ..]).to_owned(),
                )
                .unwrap(),
            })
        }

        arcs
    }
}

impl ArcString {
    pub fn midpoints(&self) -> MultiSphericalPoint {
        MultiSphericalPoint::try_from(
            (&self.points.xyz.slice(s![..-1, ..]) + &self.points.xyz.slice(s![1.., ..]) / 2.0)
                .to_owned(),
        )
        .unwrap()
    }

    /// expanded list of 2x3 arrays representing the endpoints of each individual arc
    pub fn arcs(&self) -> Vec<ArcString> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).view().rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| {
                ArcString::from(unsafe {
                    MultiSphericalPoint::try_from(stack(Axis(0), &[a, b]).unwrap_unchecked())
                        .unwrap_unchecked()
                })
            })
            .to_vec()
    }

    /// whether this arcstring intersects itself
    pub fn intersects_self(&self) -> bool {
        self.points.len() > 3 && self.intersection_with_self().is_some()
    }

    /// points of intersection with itself
    pub fn intersection_with_self(&self) -> Option<MultiSphericalPoint> {
        if self.points.len() < 4 {
            return None;
        }

        let mut intersections = vec![];

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so instead we use brute-force and skip visited arcs
        for arc_index in 0..self.points.len() - 1 {
            let a_0 = self.points.xyz.slice(s![arc_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_index + 1, ..]);

            for other_arc_index in arc_index..self.points.len() - 1 {
                let b_0 = self.points.xyz.slice(s![other_arc_index, ..]);
                let b_1 = self.points.xyz.slice(s![other_arc_index + 1, ..]);

                if let Some(point) = vector_arcs_intersection(&a_0, &a_1, &b_0, &b_1) {
                    intersections.push(point);
                }
            }
        }

        if intersections.len() > 0 {
            Some(unsafe { MultiSphericalPoint::try_from(intersections).unwrap_unchecked() })
        } else {
            None
        }
    }

    pub fn lengths(&self) -> Array1<f64> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| vector_arc_length(&a, &b))
    }

    pub fn closed(&self) -> Self {
        let mut owned = self.to_owned();
        owned.close();
        owned
    }

    pub fn close(&mut self) {
        let tolerance = 1e-10;

        let first = self.points.xyz.slice(s![0, ..]);
        let last = self.points.xyz.slice(s![self.points.xyz.nrows() - 1, ..]);
        if (&first - &last).abs().sum() > tolerance {
            self.points
                .push(unsafe { SphericalPoint::try_from(first.to_owned()).unwrap_unchecked() });
        }
    }
}

impl ToString for ArcString {
    fn to_string(&self) -> String {
        format!("ArcString({:?})", self.points)
    }
}

impl Geometry for &ArcString {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.lengths().sum()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self.points).convex_hull()
    }

    fn points(&self) -> MultiSphericalPoint {
        self.points.to_owned()
    }
}

impl Geometry for ArcString {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> MultiSphericalPoint {
        (&self).points()
    }
}

impl GeometricOperations<&SphericalPoint> for &ArcString {
    fn distance(self, other: &SphericalPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        // check if point is one of the vertices of this linestring
        if (&self.points).contains(other) {
            return true;
        }

        // check if point is within the bounding box
        if self.bounds(false).contains(other) {
            // compare lengths to endpoints with the arc length
            for index in 0..self.points.xyz.nrows() - 1 {
                let a = self.points.xyz.slice(s![index, ..]);
                let b = self.points.xyz.slice(s![index + 1, ..]);
                let p = other.xyz.view();

                if vectors_collinear(&a.view(), &p.view(), &b.view()) {
                    return true;
                }
            }
        }

        return false;
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.intersection(other).is_some()
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self.contains(other) {
            Some(other.to_owned())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &ArcString {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        // check if points are vertices of this linestring
        if (&self.points).contains(other) {
            return true;
        }

        Zip::from(other.xyz.rows())
            .and(other.to_lonlats(false).rows())
            .any(|vector, lonlat| {
                // check if point is within the bounding box
                if self.bounds(false).contains_lonlat(lonlat[0], lonlat[1]) {
                    // compare lengths to endpoints with the arc length
                    for index in 0..self.points.xyz.nrows() - 1 {
                        let a = self.points.xyz.slice(s![index, ..]);
                        let b = self.points.xyz.slice(s![index + 1, ..]);
                        let p = vector.view();

                        if vectors_collinear(&a.view(), &p.view(), &b.view()) {
                            return true;
                        }
                    }
                }

                return false;
            })
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&ArcString> for &ArcString {
    fn distance(self, other: &ArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &ArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &ArcString) -> bool {
        self.points
            .to_owned()
            .into_iter()
            .all(|point| point.within(other))
    }

    fn intersects(self, other: &ArcString) -> bool {
        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so instead we use brute-force
        for arc_index in 0..self.points.len() - 1 {
            let a_0 = self.points.xyz.slice(s![arc_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_index + 1, ..]);

            for other_arc_index in 0..other.points.len() - 1 {
                let b_0 = other.points.xyz.slice(s![other_arc_index, ..]);
                let b_1 = other.points.xyz.slice(s![other_arc_index + 1, ..]);

                if vector_arcs_intersection(&a_0, &a_1, &b_0, &b_1).is_some() {
                    return true;
                }
            }
        }

        return false;
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so instead we use brute-force
        for arc_index in 0..self.points.len() - 1 {
            let a_0 = self.points.xyz.slice(s![arc_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_index + 1, ..]);

            for other_arc_index in 0..other.points.len() - 1 {
                let b_0 = other.points.xyz.slice(s![other_arc_index, ..]);
                let b_1 = other.points.xyz.slice(s![other_arc_index + 1, ..]);

                if let Some(point) = vector_arcs_intersection(&a_0, &a_1, &b_0, &b_1) {
                    intersections.push(point);
                }
            }
        }

        if intersections.len() > 0 {
            Some(MultiSphericalPoint::try_from(intersections).unwrap())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiArcString> for &ArcString {
    fn distance(self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&AngularBounds> for &ArcString {
    fn distance(self, other: &AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        self.points.within(other)
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        // TODO: this could probably just be a clip by bounds
        other
            .convex_hull()
            .map_or(false, |convex_hull| convex_hull.intersects(self))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularBounds) -> Option<MultiArcString> {
        // TODO: this could probably just be a clip by bounds
        if let Some(convex_hull) = other.convex_hull() {
            convex_hull.intersection(self)
        } else {
            None
        }
    }
}

impl GeometricOperations<&SphericalPolygon> for &ArcString {
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
    fn intersection(self, other: &SphericalPolygon) -> Option<MultiArcString> {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &ArcString {
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
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiArcString> {
        other.intersection(self)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct MultiArcString {
    pub arcstrings: VecDeque<ArcString>,
}

impl From<Vec<ArcString>> for MultiArcString {
    fn from(arcstrings: Vec<ArcString>) -> Self {
        Self {
            arcstrings: VecDeque::<ArcString>::from(arcstrings),
        }
    }
}

impl TryFrom<Vec<MultiSphericalPoint>> for MultiArcString {
    type Error = String;

    fn try_from(points: Vec<MultiSphericalPoint>) -> Result<Self, Self::Error> {
        let arcstrings: Vec<ArcString> = points
            .par_iter()
            .map(|points| ArcString::try_from(points.to_owned()).unwrap())
            .collect();
        Ok(Self::from(arcstrings))
    }
}

impl Into<Vec<MultiSphericalPoint>> for MultiArcString {
    fn into(self) -> Vec<MultiSphericalPoint> {
        self.arcstrings
            .into_par_iter()
            .map(|arcstring| arcstring.points)
            .collect()
    }
}

impl Into<Vec<ArcString>> for MultiArcString {
    fn into(self) -> Vec<ArcString> {
        self.arcstrings.into()
    }
}

impl MultiArcString {
    pub fn midpoints(&self) -> MultiSphericalPoint {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.midpoints())
            .sum()
    }

    pub fn lengths(&self) -> Array1<f64> {
        Array1::from_vec(
            self.arcstrings
                .par_iter()
                .map(|arcstring| arcstring.length())
                .collect(),
        )
    }
}

impl ToString for MultiArcString {
    fn to_string(&self) -> String {
        format!("MultiArcString({:?})", self.arcstrings)
    }
}

impl PartialEq<MultiArcString> for MultiArcString {
    fn eq(&self, other: &MultiArcString) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for arcstring in &self.arcstrings {
            if !other.arcstrings.contains(arcstring) {
                return false;
            }
        }

        return true;
    }
}

impl PartialEq<Vec<ArcString>> for MultiArcString {
    fn eq(&self, other: &Vec<ArcString>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for arcstring in &self.arcstrings {
            if !other.contains(arcstring) {
                return false;
            }
        }

        return true;
    }
}

impl Geometry for &MultiArcString {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.length())
            .sum()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.to_owned().points)
            .sum()
    }
}

impl Geometry for MultiArcString {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn points(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).points()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl MultiGeometry for &MultiArcString {
    fn len(&self) -> usize {
        self.arcstrings.len()
    }
}

impl MultiGeometry for MultiArcString {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl Sum for MultiArcString {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut arcstrings = vec![];
        for multiarcstring in iter {
            arcstrings.extend(multiarcstring.arcstrings);
        }
        MultiArcString::from(arcstrings)
    }
}

impl ExtendMultiGeometry<ArcString> for MultiArcString {
    fn extend(&mut self, other: Self) {
        self.arcstrings.extend(other.arcstrings);
    }

    fn push(&mut self, other: ArcString) {
        self.arcstrings.push_back(other);
    }
}

impl GeometricOperations<&SphericalPoint> for &MultiArcString {
    fn distance(self, other: &SphericalPoint) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &MultiArcString {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.contains(other))
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        self.intersection(other).is_some()
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .par_iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if intersections.len() > 0 {
            Some(MultiSphericalPoint::from(&intersections))
        } else {
            None
        }
    }
}

impl GeometricOperations<&ArcString> for &MultiArcString {
    fn distance(self, other: &ArcString) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if intersections.len() > 0 {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }
}

impl GeometricOperations<&MultiArcString> for &MultiArcString {
    fn distance(self, other: &MultiArcString) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if intersections.len() > 0 {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }
}

impl GeometricOperations<&AngularBounds> for &MultiArcString {
    fn distance(self, other: &AngularBounds) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularBounds) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiArcString {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiArcString {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }
}

impl<'a> Iterator for MultiGeometryIterator<'a, MultiArcString> {
    type Item = &'a ArcString;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.multi.len() {
            Some(&self.multi.arcstrings[self.index])
        } else {
            None
        }
    }
}

impl MultiArcString {
    #[allow(dead_code)]
    fn iter(&self) -> MultiGeometryIterator<MultiArcString> {
        MultiGeometryIterator::<MultiArcString> {
            multi: self,
            index: 0,
        }
    }
}

impl Iterator for MultiGeometryIntoIterator<MultiArcString> {
    type Item = ArcString;

    fn next(&mut self) -> Option<Self::Item> {
        self.multi.arcstrings.pop_front()
    }
}

impl IntoIterator for MultiArcString {
    type Item = ArcString;

    type IntoIter = MultiGeometryIntoIterator<MultiArcString>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            multi: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::MultiGeometry;
    use numpy::ndarray::{array, linspace, s};

    #[test]
    fn test_midpoint() {
        let tolerance = 1e-10;

        let mut avec = Array2::<f64>::zeros((0, 2));
        let mut bvec = Array2::<f64>::zeros((0, 2));
        for i in linspace(0., 11., 5) {
            for j in linspace(0., 11., 5) {
                let row = array![i, j];
                avec.push_row(row.view()).unwrap();
                bvec.push_row(row.view()).unwrap();
            }
        }
        avec += 7.0;
        bvec += 10.0;

        for a in avec.rows() {
            let a = SphericalPoint::try_from_lonlat(&a, true).unwrap();
            for b in bvec.rows() {
                let b = SphericalPoint::try_from_lonlat(&b, true).unwrap();
                let c = ArcString {
                    points: a.combine(&b),
                }
                .midpoints();
                let aclen = ArcString { points: &a + &c }.length();
                let bclen = ArcString { points: &b + &c }.length();
                assert!((aclen - bclen) < tolerance)
            }
        }
    }

    #[test]
    fn test_contains() {
        let arc = ArcString {
            points: MultiSphericalPoint::try_from_lonlats(
                &array![[-30.0, -30.0], [30.0, 30.0]].view(),
                true,
            )
            .unwrap(),
        };
        assert!((&arc).contains(
            &SphericalPoint::try_from_lonlat(&array![349.10660535, -12.30998866].view(), true)
                .unwrap()
        ));

        let vertical_arc = ArcString {
            points: MultiSphericalPoint::try_from_lonlats(
                &array![[60.0, 0.0], [60.0, 30.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!((&vertical_arc)
                .contains(&SphericalPoint::try_from_lonlat(&array![60.0, i].view(), true).unwrap()))
        }

        let horizontal_arc = ArcString {
            points: MultiSphericalPoint::try_from_lonlats(
                &array![[0.0, 60.0], [30.0, 60.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!((&horizontal_arc).contains(
                &SphericalPoint::try_from_lonlat(&array![i, 60.0].view(), true).unwrap()
            ));
        }
    }

    #[test]
    fn test_interpolate() {
        let tolerance = 1e-10;

        let a_lonlat = array![60.0, 0.0];
        let b_lonlat = array![60.0, 30.0];
        let lonlats =
            interpolate_points_along_vector_arc(&a_lonlat.view(), &b_lonlat.view(), 10).unwrap();

        let a = SphericalPoint::try_from_lonlat(&a_lonlat.view(), true).unwrap();
        let b = SphericalPoint::try_from_lonlat(&b_lonlat.view(), true).unwrap();

        assert!(Zip::from(&lonlats.slice(s![0, ..]))
            .and(&a_lonlat.view())
            .all(|test, reference| (test - reference).abs() < tolerance));
        assert!(Zip::from(&lonlats.slice(s![-1, ..]))
            .and(&b_lonlat.view())
            .all(|test, reference| (test - reference).abs() < tolerance));

        let xyzs = interpolate_points_along_vector_arc(&a.xyz.view(), &b.xyz.view(), 10).unwrap();

        assert!(Zip::from(&xyzs.slice(s![0, ..]))
            .and(&a.xyz.view())
            .all(|test, reference| (test - reference).abs() < tolerance));
        assert!(Zip::from(&xyzs.slice(s![-1, ..]))
            .and(&b.xyz.view())
            .all(|test, reference| (test - reference).abs() < tolerance));

        let arc_from_lonlats = ArcString {
            points: MultiSphericalPoint::try_from_lonlats(&lonlats.view(), true).unwrap(),
        };
        let arc_from_xyzs = ArcString {
            points: MultiSphericalPoint::try_from(xyzs.to_owned()).unwrap(),
        };

        for xyz in xyzs.rows() {
            let point = SphericalPoint::try_from(xyz.to_owned()).unwrap();
            assert!((&arc_from_lonlats).contains(&point));
            assert!((&arc_from_xyzs).contains(&point));
        }

        let distances_from_lonlats = arc_from_lonlats.lengths();
        let distances_from_xyz = arc_from_xyzs.lengths();

        assert!(Zip::from(&distances_from_lonlats)
            .and(&distances_from_xyz)
            .all(|from_lonlats, from_xyz| (from_lonlats - from_xyz).abs() < tolerance));
    }

    #[test]
    fn test_intersection() {
        let tolerance = 1e-10;

        let a = SphericalPoint::try_from_lonlat(&array![-10.0, -10.0].view(), true).unwrap();
        let b = SphericalPoint::try_from_lonlat(&array![10.0, 10.0].view(), true).unwrap();

        let c = SphericalPoint::try_from_lonlat(&array![-25.0, 10.0].view(), true).unwrap();
        let d = SphericalPoint::try_from_lonlat(&array![15.0, -10.0].view(), true).unwrap();

        // let e = VectorPoint::try_from_lonlat(&array![-20.0, 40.0].view(), true).unwrap();
        // let f = VectorPoint::try_from_lonlat(&array![20.0, 40.0].view(), true).unwrap();

        let reference_intersection = array![0.99912414, -0.02936109, -0.02981403];

        let ab = ArcString {
            points: a.combine(&b),
        };
        let cd = ArcString {
            points: c.combine(&d),
        };
        assert!((&ab).intersects(&cd));
        let r = (&ab).intersection(&cd);
        assert!(r.is_some());
        let r = r.unwrap();
        assert!(r.len() == 3);
        assert!(Zip::from(r.xyz.rows())
            .all(|point| (&point - &reference_intersection.view()).abs().sum() < tolerance));

        // assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
        // r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
        // assert r.shape == (2, 3)
        // assert_allclose(r[0], reference_intersection)
        // assert np.all(np.isnan(r[1]))

        // Test parallel arcs
        let r = (&ab).intersection(&ab);
        assert!(r.is_some());
        assert!(r.unwrap().xyz.is_all_nan());
    }

    #[test]
    fn test_length() {
        let tolerance = 1e-10;

        let a = SphericalPoint::try_from_lonlat(&array![90.0, 0.0].view(), true).unwrap();
        let b = SphericalPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        let ab = ArcString {
            points: a.combine(&b),
        };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - std::f64::consts::PI).abs() < tolerance);

        let a = SphericalPoint::try_from_lonlat(&array![135.0, 0.0].view(), true).unwrap();
        let b = SphericalPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        let ab = ArcString {
            points: a.combine(&b),
        };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - (3.0 / 4.0) * std::f64::consts::PI).abs() < tolerance);

        let a = SphericalPoint::try_from_lonlat(&array![0.0, 0.0].view(), true).unwrap();
        let b = SphericalPoint::try_from_lonlat(&array![0.0, 90.0].view(), true).unwrap();
        let ab = ArcString {
            points: a.combine(&b),
        };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - std::f64::consts::PI / 2.0).abs() < tolerance);
    }

    #[test]
    fn test_angle() {
        let a = SphericalPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        let b = SphericalPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        let c = SphericalPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);

        // TODO: More angle tests
    }

    #[test]
    fn test_angle_domain() {
        let a = SphericalPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        let b = SphericalPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        let c = SphericalPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);
        assert!(!(b.angle(&a, &c, false)).is_infinite());
    }

    #[test]
    fn test_length_domain() {
        let a = SphericalPoint::try_from(array![std::f64::NAN, 0.0, 0.0]).unwrap();
        let b = SphericalPoint::try_from(array![0.0, 0.0, std::f64::INFINITY]).unwrap();
        assert!((&a).distance(&b).is_nan());
    }

    #[test]
    fn test_angle_nearly_coplanar_vec() {
        // test from issue #222 + extra values
        let a = MultiSphericalPoint::try_from(
            array![1.0, 1.0, 1.0].broadcast((5, 3)).unwrap().to_owned(),
        )
        .unwrap();
        let b = MultiSphericalPoint::try_from(
            array![1.0, 0.9999999, 1.0]
                .broadcast((5, 3))
                .unwrap()
                .to_owned(),
        )
        .unwrap();
        let c = MultiSphericalPoint::try_from(array![
            [1.0, 0.5, 1.0],
            [1.0, 0.15, 1.0],
            [1.0, 0.001, 1.0],
            [1.0, 0.15, 1.0],
            [-1.0, 0.1, -1.0],
        ])
        .unwrap();
        // vectors = np.stack([A, B, C], axis=0)
        let angles = b.angles(&a, &c, false);

        assert!(
            Zip::from(&angles.slice(s![..-1]).abs_sub(std::f64::consts::PI))
                .all(|value| value < &1e-16)
        );
        assert!(Zip::from(&angles.slice(s![-1]).abs()).all(|value| value < &1e-32));
    }
}
