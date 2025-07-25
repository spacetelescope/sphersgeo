use crate::{
    geometry::{ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry},
    sphericalpoint::{
        cross_vector, min_1darray, normalize_vector, point_within_kdtree, vector_arc_length,
        MultiSphericalPoint, SphericalPoint,
    },
};
use numpy::ndarray::{array, concatenate, s, stack, Array1, Array2, ArrayView1, Axis, Zip};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{collections::VecDeque, fmt::Display, iter::Sum};

pub fn interpolate_points_along_vector_arc(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    n: usize,
) -> Result<Array2<f64>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::linspace(0.0, 1.0, n);
    let t = t.to_shape((n, 1)).unwrap();
    let omega = vector_arc_length(a, b, false);

    if a.len() == b.len() {
        if a.len() == 3 && b.len() == 3 {
            let offsets = if omega == 0.0 {
                t.to_owned()
            } else {
                (t * omega).sin() / omega.sin()
            };
            let mut inverted_offsets = offsets.to_owned();
            inverted_offsets.invert_axis(Axis(0));

            Ok(inverted_offsets * a + offsets * b)
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
            Err(String::from("invalid input"))
        }
    } else {
        Err(String::from("shapes must match"))
    }
}

/// Given xyz vectors of the endpoints of two great circle arcs, find the point at which the arcs cross
///
/// References
/// ----------
/// - Method explained in an `e-mail <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_ by Roger Stafford.
/// - https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
/// - Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
pub fn vector_arc_crossings(
    a_0: &ArrayView1<f64>,
    a_1: &ArrayView1<f64>,
    b_0: &ArrayView1<f64>,
    b_1: &ArrayView1<f64>,
) -> Option<Array1<f64>> {
    let p = cross_vector(a_0, a_1);
    let q = cross_vector(b_0, b_1);

    let t = cross_vector(&p.view(), &q.view());

    let signs = array![
        -cross_vector(a_0, &p.view()).dot(&t.view()),
        cross_vector(a_1, &p.view()).dot(&t.view()),
        -cross_vector(b_0, &q.view()).dot(&t.view()),
        cross_vector(b_1, &q.view()).dot(&t.view()),
    ]
    .signum();

    if signs.iter().all(|sign| sign.is_sign_positive()) {
        Some(t)
    } else if signs.iter().all(|sign| sign.is_sign_negative()) {
        Some(-t)
    } else {
        None
    }
}

pub fn arcstring_contains_point(arcstring: &ArcString, xyz: &ArrayView1<f64>) -> bool {
    let xyzs = &arcstring.points.xyz;

    // if the arcstring is not closed, make sure the point is not one of the terminal endpoints
    if !arcstring.closed {
        let tolerance = 1e-10;
        let normalized = normalize_vector(xyz);
        let start = xyzs.slice(s![0, ..]);
        let end = xyzs.slice(s![xyzs.nrows() - 1, ..]);
        if (&normalize_vector(&start) - &normalized).abs().sum() < tolerance
            || (&normalize_vector(&end) - &normalized).abs().sum() < tolerance
        {
            return false;
        }
    }

    // check if point is one of the vertices of this linestring
    if point_within_kdtree(xyz, &arcstring.points.kdtree) {
        return true;
    }

    // iterate over endpoints and check if collinear with the given point
    for index in 0..arcstring.points.xyz.nrows() - 1 {
        let a = xyzs.slice(s![index, ..]);
        let b = xyzs.slice(s![index + 1, ..]);

        if crate::sphericalpoint::vectors_collinear(&a.view(), xyz, &b.view()) {
            return true;
        }
    }

    false
}

/// for arc AB, the closest point T to given point C is
///
/// G = A x B
/// F = C x G
/// T = G x F
///
/// References
/// ----------
/// - https://stackoverflow.com/a/1302268
fn arc_distance_to_point(a: &ArrayView1<f64>, b: &ArrayView1<f64>, xyz: &ArrayView1<f64>) -> f64 {
    let g = crate::sphericalpoint::cross_vector(&a, &b);
    let f = crate::sphericalpoint::cross_vector(xyz, &g.view());
    let t = crate::sphericalpoint::cross_vector(&g.view(), &f.view());
    vector_arc_length(&t.view(), xyz, false)
}

/// series of great circle arcs along the sphere
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct ArcString {
    pub points: MultiSphericalPoint,
    pub closed: bool,
}

impl From<MultiSphericalPoint> for ArcString {
    fn from(points: MultiSphericalPoint) -> Self {
        let (points, closed) = if points.xyz.slice(s![-1, ..]) == points.xyz.slice(s![0, ..]) {
            (
                MultiSphericalPoint::try_from(points.xyz.slice(s![..-1, ..]).to_owned()).unwrap(),
                true,
            )
        } else {
            (points, false)
        };

        Self { points, closed }
    }
}

impl From<ArcString> for MultiSphericalPoint {
    fn from(arcstring: ArcString) -> Self {
        arcstring.points
    }
}

impl From<&ArcString> for Vec<ArcString> {
    fn from(arcstring: &ArcString) -> Self {
        let vectors = &arcstring.points.xyz;
        let mut arcs = vec![];
        for index in 0..vectors.nrows() - 1 {
            arcs.push(ArcString {
                points: MultiSphericalPoint::try_from(
                    vectors.slice(s![index..index + 1, ..]).to_owned(),
                )
                .unwrap(),
                closed: false,
            })
        }

        arcs
    }
}

impl ArcString {
    pub fn midpoints(&self) -> MultiSphericalPoint {
        MultiSphericalPoint::try_from(
            ((&self.points.xyz.slice(s![..-1, ..]) + &self.points.xyz.slice(s![1.., ..])) / 2.0)
                .to_owned(),
        )
        .unwrap()
    }

    /// expanded list of 2x3 arrays representing the endpoints of each individual arc
    pub fn arcs(&self) -> Vec<ArcString> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).view().rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| {
                ArcString::from(
                    MultiSphericalPoint::try_from(stack(Axis(0), &[a, b]).unwrap()).unwrap(),
                )
            })
            .to_vec()
    }

    /// whether this arcstring intersects itself
    pub fn crosses_self(&self) -> bool {
        if self.points.len() >= 4 {
            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so instead we use brute-force and skip visited arcs
            let start_index = if self.closed { -1 } else { 0 };
            for a_0_index in start_index..(self.points.len() - 1) as isize {
                let a_0 = self.points.xyz.slice(s![a_0_index, ..]);
                let a_1 = self.points.xyz.slice(s![a_0_index + 1, ..]);

                for b_0_index in a_0_index + 2..(self.points.len() - 1) as isize {
                    let b_0 = self.points.xyz.slice(s![b_0_index, ..]);
                    let b_1 = self.points.xyz.slice(s![b_0_index + 1, ..]);
                    if let Some(point) = vector_arc_crossings(&a_0, &a_1, &b_0, &b_1) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// points of intersection with itself
    pub fn crossings_with_self(&self) -> Option<MultiSphericalPoint> {
        if self.points.len() >= 4 {
            let mut crossings = vec![];

            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so instead we use brute-force and skip visited arcs
            let start_index = if self.closed { -1 } else { 0 };
            for a_0_index in start_index..(self.points.len() - 1) as isize {
                let a_0 = self.points.xyz.slice(s![a_0_index, ..]);
                let a_1 = self.points.xyz.slice(s![a_0_index + 1, ..]);

                for b_0_index in a_0_index + 2..(self.points.len() - 1) as isize {
                    let b_0 = self.points.xyz.slice(s![b_0_index, ..]);
                    let b_1 = self.points.xyz.slice(s![b_0_index + 1, ..]);

                    if let Some(point) = vector_arc_crossings(&a_0, &a_1, &b_0, &b_1) {
                        crossings.push(point);
                    }
                }
            }

            if !crossings.is_empty() {
                return Some(MultiSphericalPoint::try_from(&crossings).unwrap());
            }
        }

        None
    }

    pub fn lengths(&self) -> Array1<f64> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| vector_arc_length(&a, &b, false))
    }
}

impl Display for ArcString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArcString({:?})", self.points)
    }
}

impl Geometry for &ArcString {
    fn vertices(&self) -> MultiSphericalPoint {
        self.points.to_owned()
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.lengths().sum()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.representative_point()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        Some(self.points.to_owned())
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.points.convex_hull()
    }
}

impl Geometry for ArcString {
    fn vertices(&self) -> MultiSphericalPoint {
        (&self).vertices()
    }

    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        (&self).boundary()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<&SphericalPoint> for &ArcString {
    fn distance(self, other: &SphericalPoint, degrees: bool) -> f64 {
        let mut distances = Zip::from(self.points.xyz.rows())
            .and(crate::sphericalpoint::shift_rows(&self.points.xyz.view(), 1).rows())
            .par_map_collect(|a, b| arc_distance_to_point(&a, &b, &other.xyz.view()));

        if degrees {
            distances = distances.to_degrees();
        }

        crate::sphericalpoint::min_1darray(&distances.view()).unwrap_or(f64::NAN)
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        arcstring_contains_point(self, &other.xyz.view())
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
        self.intersects(other)
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &ArcString {
    fn distance(self, other: &MultiSphericalPoint, degrees: bool) -> f64 {
        let mut distances = Array1::<f64>::uninit(self.points.xyz.nrows() * other.xyz.nrows());
        for (index, point) in other.xyz.rows().into_iter().enumerate() {
            Zip::from(self.points.xyz.rows())
                .and(crate::sphericalpoint::shift_rows(&self.points.xyz.view(), 1).rows())
                .par_map_collect(|a, b| arc_distance_to_point(&a, &b, &point))
                .assign_to(distances.slice_mut(s![
                    index * other.xyz.nrows()..(index + 1) * other.xyz.nrows()
                ]));
        }
        let mut distances = unsafe { distances.assume_init() };

        if degrees {
            distances = distances.to_degrees();
        }

        min_1darray(&distances.view()).unwrap_or(f64::NAN)
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        // check if points are vertices of this linestring
        if (&self.points).contains(other) {
            return true;
        }

        Zip::from(other.xyz.rows()).any(|xyz| {
            // compare lengths to endpoints with the arc length
            for index in 0..self.points.xyz.nrows() - 1 {
                let a = self.points.xyz.slice(s![index, ..]);
                let b = self.points.xyz.slice(s![index + 1, ..]);

                if crate::sphericalpoint::vectors_collinear(&a.view(), &xyz, &b.view()) {
                    return true;
                }
            }

            false
        })
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.intersects(other)
    }
}

impl GeometricOperations<&ArcString> for &ArcString {
    fn distance(self, other: &ArcString, degrees: bool) -> f64 {
        todo!()
    }

    fn contains(self, other: &ArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &ArcString) -> bool {
        Zip::from(self.points.xyz.rows()).all(|xyz| arcstring_contains_point(other, &xyz))
    }

    fn crosses(self, other: &ArcString) -> bool {
        if self.within(other) || self.contains(other) {
            return false;
        }

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so instead we use brute-force
        let start_index = if self.closed { -1 } else { 0 };
        for arc_index in start_index..(self.points.xyz.nrows() - 1) as isize {
            let a_0 = self.points.xyz.slice(s![arc_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_index + 1, ..]);

            let other_start_index = if other.closed { -1 } else { 0 };
            for other_arc_index in other_start_index..(other.points.xyz.nrows() - 1) as isize {
                let b_0 = other.points.xyz.slice(s![other_arc_index, ..]);
                let b_1 = other.points.xyz.slice(s![other_arc_index + 1, ..]);

                if vector_arc_crossings(&a_0, &a_1, &b_0, &b_1).is_some() {
                    return true;
                }
            }
        }

        false
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.eq(other)
    }

    fn intersection(self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        // find crossings first
        let start_index = if self.closed { -1 } else { 0 };
        for arc_index in start_index..(self.points.xyz.nrows() - 1) as isize {
            let a_0 = self.points.xyz.slice(s![arc_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_index + 1, ..]);

            let other_start_index = if other.closed { -1 } else { 0 };
            for other_arc_index in other_start_index..(other.points.xyz.nrows() - 1) as isize {
                let b_0 = other.points.xyz.slice(s![other_arc_index, ..]);
                let b_1 = other.points.xyz.slice(s![other_arc_index + 1, ..]);

                if let Some(point) = vector_arc_crossings(&a_0, &a_1, &b_0, &b_1) {
                    intersections.push(point);
                }
            }
        }

        // then add intersections
        intersections.extend(other.points.xyz.rows().into_iter().filter_map(|point| {
            if arcstring_contains_point(self, &point.view()) {
                Some(point.to_owned())
            } else {
                None
            }
        }));

        if !intersections.is_empty() {
            Some(MultiSphericalPoint::try_from(&intersections).unwrap())
        } else {
            None
        }
    }

    fn touches(self, other: &ArcString) -> bool {
        (Zip::from(other.points.xyz.rows()).any(|xyz| arcstring_contains_point(self, &xyz))
            || Zip::from(self.points.xyz.rows()).any(|xyz| arcstring_contains_point(other, &xyz)))
            && !self.crosses(other)
    }
}

impl GeometricOperations<&MultiArcString> for &ArcString {
    fn distance(self, other: &MultiArcString, degrees: bool) -> f64 {
        other.distance(self, degrees)
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &MultiArcString) -> bool {
        other.crosses(self)
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &MultiArcString) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &ArcString {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon, degrees: bool) -> f64 {
        other.distance(self, degrees)
    }

    fn contains(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.crosses(&other.boundary)
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &ArcString {
    fn distance(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
        degrees: bool,
    ) -> f64 {
        other.distance(self, degrees)
    }

    fn contains(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.crosses(self)
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
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
            .map(|points| ArcString::from(points.to_owned()))
            .collect();
        Ok(Self::from(arcstrings))
    }
}

impl TryFrom<Vec<Array2<f64>>> for MultiArcString {
    type Error = String;

    fn try_from(xyzs: Vec<Array2<f64>>) -> Result<Self, Self::Error> {
        let mut arcstrings = vec![];
        for xyz in xyzs {
            arcstrings.push(ArcString::from(MultiSphericalPoint::try_from(xyz)?));
        }
        Ok(Self::from(arcstrings))
    }
}

impl From<MultiArcString> for Vec<MultiSphericalPoint> {
    fn from(arcstrings: MultiArcString) -> Self {
        arcstrings
            .arcstrings
            .into_par_iter()
            .map(|arcstring| arcstring.points)
            .collect()
    }
}

impl From<MultiArcString> for Vec<ArcString> {
    fn from(arcstrings: MultiArcString) -> Self {
        arcstrings.arcstrings.into()
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

impl Display for MultiArcString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiArcString({:?})", self.arcstrings)
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

        true
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

        true
    }
}

impl Geometry for &MultiArcString {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.to_owned().points)
            .sum()
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.length())
            .sum()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        self.arcstrings[0].representative_point()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.vertices().centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        Some(self.vertices())
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.vertices().convex_hull()
    }
}

impl Geometry for MultiArcString {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).vertices()
    }

    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        (&self).boundary()
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
    fn distance(self, other: &SphericalPoint, degrees: bool) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
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

    fn crosses(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &MultiArcString {
    fn distance(self, other: &MultiSphericalPoint, degrees: bool) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
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

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .par_iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(MultiSphericalPoint::from(&intersections))
        } else {
            None
        }
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }
}

impl GeometricOperations<&ArcString> for &MultiArcString {
    fn distance(self, other: &ArcString, degrees: bool) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
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

    fn crosses(self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .par_iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }

    fn touches(self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }
}

impl GeometricOperations<&MultiArcString> for &MultiArcString {
    fn distance(self, other: &MultiArcString, degrees: bool) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
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

    fn crosses(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .par_iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }

    fn touches(self, other: &MultiArcString) -> bool {
        self.intersects(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &MultiArcString {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon, degrees: bool) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn crosses(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn touches(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.intersects(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &MultiArcString {
    fn distance(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
        degrees: bool,
    ) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other, degrees))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn crosses(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn touches(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
    }
}
