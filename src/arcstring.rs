use crate::{
    edgegraph::EdgeGraph,
    geometry::{
        GeometricOperations, GeometricPredicates, Geometry, GeometryCollection, MultiGeometry,
    },
    sphericalpoint::{
        point_within_kdtree, xyz_add_xyz, xyz_cross, xyz_div_f64, xyz_dot, xyz_eq, xyz_mul_f64,
        xyz_neg, xyzs_collinear, xyzs_distance_over_sphere_radians, MultiSphericalPoint,
        SphericalPoint,
    },
};
use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign},
};

#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg(feature = "ndarray")]
use numpy::ndarray::{array, concatenate, s, Array1, Array2, ArrayView2, Axis};

pub fn linspace(x0: f64, xend: f64, n: usize) -> Vec<f64> {
    let dx = (xend - x0) / ((n - 1) as f64);
    let mut x = vec![x0; n];
    for i in 1..n {
        x[i] = x[i - 1] + dx;
    }
    x
}

pub fn interpolate_points_along_arc(
    arc: (&[f64; 3], &[f64; 3]),
    n: usize,
) -> Result<Vec<[f64; 3]>, String> {
    let n = if n < 2 { 2 } else { n };
    let omega = crate::sphericalpoint::xyzs_distance_over_sphere_radians(arc.0, arc.1);

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
            xyz_add_xyz(
                &xyz_mul_f64(arc.0, inverted_offset),
                &xyz_mul_f64(arc.1, offset),
            )
        })
        .collect())
}

/// Given xyz vectors of the endpoints of two great circle arcs, find the point at which the arcs cross
///
/// References
/// ----------
/// - Method explained in an `e-mail <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_ by Roger Stafford.
/// - https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
/// - Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
pub fn xyz_two_arc_crossing(
    a: (&[f64; 3], &[f64; 3]),
    b: (&[f64; 3], &[f64; 3]),
) -> Option<[f64; 3]> {
    let p = xyz_cross(a.0, a.1);
    let q = xyz_cross(b.0, b.1);

    let t = xyz_cross(&p, &q);

    let result = [
        xyz_dot(&xyz_neg(&xyz_cross(a.0, &p)), &t),
        xyz_dot(&xyz_cross(a.1, &p), &t),
        xyz_dot(&xyz_neg(&xyz_cross(b.0, &q)), &t),
        xyz_dot(&xyz_cross(b.1, &q), &t),
    ];

    if result.iter().all(|result| result.is_sign_positive()) {
        Some(t)
    } else if result.iter().all(|sign| sign.is_sign_negative()) {
        Some(xyz_neg(&t))
    } else {
        None
    }
}

pub fn arc_crosses_arcstring(arc: (&[f64; 3], &[f64; 3]), arcstring: &ArcString) -> bool {
    for other_arc_index in 0..arcstring.points.len() - if arcstring.closed { 0 } else { 1 } {
        let other_arc = (
            &arcstring.points.xyzs[other_arc_index],
            &arcstring.points.xyzs[if other_arc_index < arcstring.points.len() - 1 {
                other_arc_index + 1
            } else {
                0
            }],
        );
        if let Some(point) = xyz_two_arc_crossing(arc, other_arc) {
            if xyz_eq(&point, arc.0)
                || xyz_eq(&point, arc.1)
                || xyz_eq(&point, other_arc.0)
                || xyz_eq(&point, other_arc.1)
            {
                continue;
            } else {
                return true;
            }
        }
    }

    false
}

pub fn arcstring_contains_point(arcstring: &ArcString, xyz: &[f64; 3]) -> bool {
    let xyzs = &arcstring.points.xyzs;

    // check if point is one of the vertices of this linestring
    if point_within_kdtree(xyz, &arcstring.points.kdtree) {
        return true;
    }

    // iterate over individual arcs and check if the given point is collinear with their endpoints
    for arc_index in 0..xyzs.len() - if arcstring.closed { 0 } else { 1 } {
        let arc_0 = xyzs[arc_index];
        let arc_1 = xyzs[if arc_index < xyzs.len() - 1 {
            arc_index + 1
        } else {
            0
        }];

        if xyzs_collinear(&arc_0, xyz, &arc_1) {
            return true;
        }
    }

    false
}

pub fn split_arc_at_points<'a>(
    arc: Vec<&'a [f64; 3]>,
    points: Vec<&'a [f64; 3]>,
) -> Vec<Vec<&'a [f64; 3]>> {
    let mut arcs = vec![arc];
    for point in points {
        for arc_index in 0..arcs.len() {
            let arc_0 = arcs[arc_index][0];
            let arc_1 = arcs[arc_index][1];

            // skip if the point is equal to one of the endpoints
            if xyz_eq(arc_0, point) || xyz_eq(point, arc_1) {
                continue;
            }

            if xyzs_collinear(arc_0, point, arc_1) {
                // replace arc with the arc split in two at the collinear point
                arcs[arc_index] = vec![arcs[arc_index][0], point];
                arcs.insert(arc_index + 1, vec![point, arcs[arc_index][1]]);
            }
        }
    }

    arcs
}

pub fn split_arcstring_at_points(arcstring: &ArcString, points: Vec<&[f64; 3]>) -> MultiArcString {
    let mut arcstrings = vec![arcstring.to_owned()];

    for point in &points {
        for arcstring_index in 0..arcstrings.len() {
            let arcstring = arcstrings[arcstring_index].to_owned();
            for arc_index in 0..arcstring.points.len() - if arcstring.closed { 0 } else { 1 } {
                let arc_0 = points[arc_index];
                let arc_1 = points[if arc_index < arcstring.points.len() - 1 {
                    arc_index + 1
                } else {
                    0
                }];

                if xyzs_collinear(arc_0, point, arc_1) {
                    let mut a = vec![];
                    a.extend_from_slice(&arcstring.points.xyzs[..arc_index + 1]);
                    a.push(**point);
                    // replace arc with the arc split in two at the collinear point
                    arcstrings[arcstring_index] =
                        ArcString::try_from(MultiSphericalPoint::try_from(a).unwrap()).unwrap();

                    let mut b = vec![**point];
                    b.extend_from_slice(&arcstring.points.xyzs[arc_index + 1..]);
                    arcstrings.insert(
                        arcstring_index + 1,
                        ArcString::try_from(MultiSphericalPoint::try_from(b).unwrap()).unwrap(),
                    );
                }
            }
        }
    }

    MultiArcString::try_from(arcstrings).unwrap()
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
fn arc_radians_over_sphere_to_point(a: &[f64; 3], b: &[f64; 3], xyz: &[f64; 3]) -> f64 {
    let g = xyz_cross(a, b);
    let f = xyz_cross(xyz, &g);
    let t = xyz_cross(&g, &f);
    xyzs_distance_over_sphere_radians(&t, xyz)
}

/// series of great circle arcs along the sphere
#[cfg_attr(feature = "py", pyclass)]
#[derive(Clone, Debug)]
pub struct ArcString {
    pub points: MultiSphericalPoint,
    pub closed: bool,
}

impl TryFrom<MultiSphericalPoint> for ArcString {
    type Error = String;

    fn try_from(points: MultiSphericalPoint) -> Result<Self, Self::Error> {
        let num_points = points.len();
        if num_points < 2 {
            Err(format!(
                "cannot build an arcstring with less than 2 points (received {num_points})",
            ))
        } else {
            Ok(if xyz_eq(&points.xyzs[0], &points.xyzs[num_points - 1]) {
                Self {
                    points: MultiSphericalPoint::try_from(points.xyzs[..num_points - 1].to_vec())?,
                    closed: true,
                }
            } else {
                Self {
                    points,
                    closed: false,
                }
            })
        }
    }
}

#[cfg(feature = "ndarray")]
impl<'a> TryFrom<Vec<ArrayView2<'a, f64>>> for ArcString {
    type Error = String;

    fn try_from(mut edges: Vec<ArrayView2<'a, f64>>) -> Result<Self, Self::Error> {
        if edges.is_empty() {
            return Err(String::from(
                "cannot create arcstring from empty set of edges...",
            ));
        }
        let mut connected = edges.pop().unwrap().to_owned();
        for _ in 0..edges.len() {
            let end = connected.slice(s![connected.nrows() - 1, ..]).to_owned();
            for edge_index in 0..edges.len() {
                if (&edges[edge_index].slice(s![0, ..]) - &end).abs().sum() < 3e-11 {
                    connected = concatenate![
                        Axis(0),
                        connected.view(),
                        edges[edge_index].slice(s![1.., ..])
                    ];
                } else if (&edges[edge_index].slice(s![edges[edge_index].nrows() - 1, ..]) - &end)
                    .abs()
                    .sum()
                    < 3e-11
                {
                    let edge = edges.get_mut(edge_index).unwrap();
                    edge.invert_axis(Axis(0));
                    connected = concatenate![Axis(0), connected.view(), edge.slice(s![1.., ..])];
                }
            }
        }

        if edges.is_empty() {
            Self::try_from(MultiSphericalPoint::try_from(connected)?)
        } else {
            Err(format!("{} disjoint edges left over", edges.len()))
        }
    }
}

impl From<ArcString> for MultiSphericalPoint {
    fn from(arcstring: ArcString) -> Self {
        arcstring.points
    }
}

impl From<&ArcString> for Vec<ArcString> {
    fn from(arcstring: &ArcString) -> Self {
        let num_points = arcstring.points.len();
        if num_points <= 2 {
            vec![arcstring.to_owned()]
        } else {
            // iterate over vertex indices, stopping short of final index if not closed
            (0..arcstring.points.len() - if arcstring.closed { 0 } else { 1 })
                .map(|index| {
                    ArcString::try_from(
                        MultiSphericalPoint::try_from(if index < arcstring.points.len() {
                            arcstring.points.xyzs[index..index + 2].to_vec()
                        } else {
                            // add additional edge returning to initial point
                            vec![arcstring.points.xyzs[index], arcstring.points.xyzs[0]]
                        })
                        .unwrap(),
                    )
                    .unwrap()
                })
                .collect()
        }
    }
}

impl ArcString {
    pub fn try_new(points: MultiSphericalPoint, closed: Option<bool>) -> Result<Self, String> {
        let mut instance = Self::try_from(points)?;
        if let Some(closed) = closed {
            instance.closed = closed;
        }
        Ok(instance)
    }

    pub fn midpoints(&self) -> MultiSphericalPoint {
        let mut midpoints = (0..self.points.len() - 1)
            .map(|index| {
                xyz_div_f64(
                    &xyz_add_xyz(&self.points.xyzs[index], &self.points.xyzs[index + 1]),
                    &2.0,
                )
            })
            .collect::<Vec<[f64; 3]>>();

        if self.closed {
            // if the arcstring is closed, also add the midpoint of the final closing arc
            midpoints.push(xyz_div_f64(
                &xyz_add_xyz(
                    &self.points.xyzs[self.points.len() - 1],
                    &self.points.xyzs[0],
                ),
                &2.0,
            ));
        }

        MultiSphericalPoint::try_from(midpoints).unwrap()
    }

    /// each individual arc in this arcstring
    pub fn arcs(&self) -> Vec<ArcString> {
        let mut arcs = (0..self.points.len() - 1)
            .map(|index| {
                ArcString::try_from(
                    MultiSphericalPoint::try_from(vec![
                        self.points.xyzs[index],
                        self.points.xyzs[index + 1],
                    ])
                    .unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<ArcString>>();

        if self.closed {
            // if the arcstring is closed, also add the final closing arc
            arcs.push(
                ArcString::try_from(
                    MultiSphericalPoint::try_from(vec![
                        self.points.xyzs[self.points.len() - 1],
                        self.points.xyzs[0],
                    ])
                    .unwrap(),
                )
                .unwrap(),
            );
        }

        arcs
    }

    /// whether this arcstring intersects itself
    pub fn crosses_self(&self) -> bool {
        if self.points.len() >= 4 {
            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so I guess the best we can do instead is use brute-force and skip visited arcs
            for arc_index in 0..self.points.len() - 1 {
                let arc = (
                    &self.points.xyzs[arc_index],
                    &self.points.xyzs[arc_index + 1],
                );

                // due to the nature of the search we can assume that previous indices are already checked
                for other_arc_index in
                    arc_index + 2..self.points.len() - if self.closed { 0 } else { 1 }
                {
                    let other_arc = (
                        &self.points.xyzs[other_arc_index],
                        &self.points.xyzs[if other_arc_index < self.points.len() - 1 {
                            other_arc_index + 1
                        } else {
                            0
                        }],
                    );
                    if let Some(point) = xyz_two_arc_crossing(arc, other_arc) {
                        if xyz_eq(&point, arc.0)
                            || xyz_eq(&point, arc.1)
                            || xyz_eq(&point, other_arc.0)
                            || xyz_eq(&point, other_arc.1)
                        {
                            continue;
                        } else {
                            return true;
                        }
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
            // so I guess the best we can do instead is use brute-force and skip visited arcs
            for arc_index in 0..self.points.len() - 1 {
                let arc = (
                    &self.points.xyzs[arc_index],
                    &self.points.xyzs[arc_index + 1],
                );

                // due to the nature of the search we can assume that previous indices are already checked
                for other_arc_index in arc_index + 2..self.points.len() - 1 {
                    let other_arc = (
                        &self.points.xyzs[other_arc_index],
                        &self.points.xyzs[other_arc_index + 1],
                    );

                    if let Some(point) = xyz_two_arc_crossing(arc, other_arc) {
                        if xyz_eq(&point, arc.0)
                            || xyz_eq(&point, arc.1)
                            || xyz_eq(&point, other_arc.0)
                            || xyz_eq(&point, other_arc.1)
                        {
                            continue;
                        } else {
                            crossings.push(point);
                        }
                    }
                }
            }

            if !crossings.is_empty() {
                return Some(MultiSphericalPoint::try_from(crossings).unwrap());
            }
        }

        None
    }

    pub fn lengths(&self) -> Vec<f64> {
        let mut lengths = (0..self.points.len() - 1)
            .map(|index| {
                xyzs_distance_over_sphere_radians(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                )
            })
            .collect::<Vec<f64>>();

        if self.closed {
            // if the arcstring is closed, also add the length of the final closing arc
            lengths.push(xyzs_distance_over_sphere_radians(
                &self.points.xyzs[self.points.len() - 1],
                &self.points.xyzs[0],
            ));
        }

        lengths
    }

    /// whether this arcstring shares endpoints with another, ignoring closed arcstrings
    pub fn adjoins(&self, other: &ArcString) -> bool {
        if let Some(boundary) = self.boundary() {
            if let Some(other_boundary) = other.boundary() {
                return boundary.touches(&other_boundary);
            }
        }

        false
    }

    /// join this arcstring to another
    pub fn join(&self, other: &ArcString) -> Option<ArcString> {
        if self.closed || other.closed {
            None
        } else {
            let mut graph = EdgeGraph::<Self>::from(vec![self, other]);
            graph.split_edges();
            graph.remove_multisourced_edges();
            graph.remove_degenerate_edges();

            let arcstrings: Vec<ArcString> = Vec::<ArcString>::from(graph);
            if arcstrings.len() == 1 {
                Some(arcstrings[0].to_owned())
            } else {
                None
            }
        }
    }
}

impl PartialEq for ArcString {
    fn eq(&self, other: &Self) -> bool {
        // either all points are equal in order, or all points are equal in reverse order
        self.points.len() == other.points.len()
            && (self
                .points
                .xyzs
                .iter()
                .zip(other.points.xyzs.iter())
                .all(|(a, b)| xyz_eq(a, b))
                || self
                    .points
                    .xyzs
                    .iter()
                    .zip(other.points.xyzs.iter().rev())
                    .all(|(a, b)| xyz_eq(a, b)))
    }
}

impl Display for ArcString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArcString({:?})", self.points.xyzs)
    }
}

impl Add<Self> for &ArcString {
    type Output = MultiArcString;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::try_from(vec![self.to_owned(), rhs.to_owned()]).unwrap()
    }
}

impl Geometry for ArcString {
    fn vertices(&self) -> MultiSphericalPoint {
        self.points.to_owned()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        if !self.closed {
            Some(
                MultiSphericalPoint::try_from(vec![
                    self.points.xyzs[0],
                    self.points.xyzs[self.points.len() - 1],
                ])
                .unwrap(),
            )
        } else {
            None
        }
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.centroid()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.points.convex_hull()
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.lengths().iter().sum()
    }
}

impl GeometricPredicates<SphericalPoint> for ArcString {
    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        arcstring_contains_point(self, &other.xyz)
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        if self.touches(other) {
            // endpoints are not part of the interior of an arcstring
            if let Some(boundary) = self.boundary() {
                !boundary.contains(other)
            } else {
                true
            }
        } else {
            false
        }
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        self.touches(other)
    }
}

impl GeometricOperations<SphericalPoint> for ArcString {
    fn union(&self, _: &SphericalPoint) -> Option<MultiArcString> {
        None
    }

    fn distance(&self, other: &SphericalPoint) -> f64 {
        let mut distances = (0..self.points.len() - 1)
            .map(|index| {
                arc_radians_over_sphere_to_point(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                    &other.xyz,
                )
            })
            .collect::<Vec<f64>>();

        if self.closed {
            // if the arcstring is closed, also add the midpoint of the final closing arc
            distances.push(arc_radians_over_sphere_to_point(
                &self.points.xyzs[self.points.len() - 1],
                &self.points.xyzs[0],
                &other.xyz,
            ));
        }

        match distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(distance) => *distance,
            None => f64::NAN,
        }
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self.contains(other) {
            Some(other.to_owned())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, other: &SphericalPoint) -> MultiArcString {
        split_arcstring_at_points(self, vec![&other.xyz])
    }
}

impl GeometricPredicates<MultiSphericalPoint> for ArcString {
    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        other.intersects(self)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        if self.covers(other) {
            if let Some(boundary) = self.boundary() {
                for xyz in &other.xyzs {
                    for endpoint_xyz in &boundary.xyzs {
                        if xyz_eq(xyz, endpoint_xyz) {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        for xyz in &other.xyzs {
            if arcstring_contains_point(self, xyz) {
                return true;
            }
        }
        false
    }
}

impl GeometricOperations<MultiSphericalPoint> for ArcString {
    fn union(&self, _: &MultiSphericalPoint) -> Option<MultiArcString> {
        None
    }

    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        let mut distances = vec![];
        for xyz in &other.xyzs {
            distances.extend((0..self.points.len() - 1).map(|index| {
                arc_radians_over_sphere_to_point(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                    xyz,
                )
            }));

            if self.closed {
                // if the arcstring is closed, also add the midpoint of the final closing arc
                distances.push(arc_radians_over_sphere_to_point(
                    &self.points.xyzs[self.points.len() - 1],
                    &self.points.xyzs[0],
                    xyz,
                ));
            }
        }

        match distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(distance) => *distance,
            None => f64::NAN,
        }
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, other: &MultiSphericalPoint) -> MultiArcString {
        split_arcstring_at_points(self, other.xyzs.iter().collect())
    }
}

impl GeometricPredicates<Self> for ArcString {
    fn intersects(&self, other: &Self) -> bool {
        self.touches(other) || self.crosses(other) || self.eq(other)
    }

    fn touches(&self, other: &Self) -> bool {
        !self.crosses(other)
            && other
                .points
                .xyzs
                .iter()
                .any(|xyz| arcstring_contains_point(self, xyz))
            || self
                .points
                .xyzs
                .iter()
                .any(|xyz| arcstring_contains_point(other, xyz))
    }

    fn crosses(&self, other: &Self) -> bool {
        if self.within(other) || self.contains(other) {
            return false;
        }

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so I guess the best we can do instead is use brute-force
        for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
            if arc_crosses_arcstring(
                (
                    &self.points.xyzs[arc_index],
                    &self.points.xyzs[if arc_index < self.points.len() - 1 {
                        arc_index + 1
                    } else {
                        0
                    }],
                ),
                &other,
            ) {
                return true;
            }
        }

        false
    }

    fn within(&self, other: &Self) -> bool {
        self.points
            .xyzs
            .iter()
            .all(|xyz| arcstring_contains_point(other, xyz))
    }

    fn contains(&self, other: &Self) -> bool {
        other.within(self)
    }

    fn overlaps(&self, other: &Self) -> bool {
        if self != other && !self.within(other) && !self.contains(other) {
            for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
                let arc = (
                    self.points.xyzs[arc_index],
                    self.points.xyzs[if arc_index <= self.points.len() - 1 {
                        arc_index + 1
                    } else {
                        0
                    }],
                );

                // TODO: handle case where an arcstring has both endpoints on the other arcstring, but cuts a corner...
                if arcstring_contains_point(other, &arc.0)
                    && arcstring_contains_point(other, &arc.1)
                {
                    return true;
                }
            }
        }

        false
    }

    fn covers(&self, other: &Self) -> bool {
        self.contains(other) || self == other
    }
}

impl GeometricOperations<ArcString> for ArcString {
    fn union(&self, other: &ArcString) -> Option<MultiArcString> {
        Some(self + other)
    }

    fn distance(&self, other: &ArcString) -> f64 {
        todo!()
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so I guess the best we can do instead is use brute-force
        for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
            let arc = (
                &self.points.xyzs[arc_index],
                &self.points.xyzs[if arc_index < self.points.len() - 1 {
                    arc_index + 1
                } else {
                    0
                }],
            );

            for other_arc_index in 0..other.points.len() - if other.closed { 0 } else { 1 } {
                let other_arc = (
                    &other.points.xyzs[other_arc_index],
                    &other.points.xyzs[if other_arc_index < other.points.len() - 1 {
                        other_arc_index + 1
                    } else {
                        0
                    }],
                );

                if let Some(point) = xyz_two_arc_crossing(arc, other_arc) {
                    intersections.push(point);
                }
            }
        }

        MultiSphericalPoint::try_from(intersections).ok()
    }

    fn symmetric_difference(&self, other: &ArcString) -> MultiArcString {
        if let Some(points) = self.intersection(other) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricPredicates<MultiArcString> for ArcString {
    fn intersects(&self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        other.touches(self)
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        other.crosses(self)
    }

    fn within(&self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn overlaps(&self, other: &MultiArcString) -> bool {
        other.overlaps(self)
    }

    fn covers(&self, other: &MultiArcString) -> bool {
        self.contains(other) || other == &MultiArcString::try_from(vec![self.to_owned()]).unwrap()
    }
}

impl GeometricOperations<MultiArcString> for ArcString {
    fn union(&self, other: &MultiArcString) -> Option<MultiArcString> {
        Some(other + self)
    }

    fn distance(&self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn intersection(&self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, other: &MultiArcString) -> MultiArcString {
        if let Some(points) = self.intersection(other) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricPredicates<crate::sphericalpolygon::SphericalPolygon> for ArcString {
    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.crosses(&other.boundary)
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

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon> for ArcString {
    fn union(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> Option<MultiArcString> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn symmetric_difference(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> MultiArcString {
        if let Some(points) = self.intersection(&other.boundary) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricPredicates<crate::sphericalpolygon::MultiSphericalPolygon> for ArcString {
    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.crosses(self)
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

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon> for ArcString {
    fn union(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> Option<MultiArcString> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn symmetric_difference(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> MultiArcString {
        if let Some(other_boundary) = other.boundary() {
            if let Some(points) = self.intersection(&other_boundary) {
                split_arcstring_at_points(self, points.xyzs.iter().collect())
            } else {
                MultiArcString::try_from(vec![self.to_owned()]).unwrap()
            }
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

#[cfg_attr(feature = "py", pyclass)]
#[derive(Debug, Clone)]
pub struct MultiArcString {
    pub arcstrings: Vec<ArcString>,
}

impl TryFrom<Vec<ArcString>> for MultiArcString {
    type Error = String;

    fn try_from(arcstrings: Vec<ArcString>) -> Result<Self, Self::Error> {
        if !arcstrings.is_empty() {
            Ok(Self { arcstrings })
        } else {
            Err(String::from("no arcstrings provided"))
        }
    }
}

impl From<Vec<MultiSphericalPoint>> for MultiArcString {
    fn from(points: Vec<MultiSphericalPoint>) -> Self {
        let arcstrings: Vec<ArcString> = points
            .iter()
            .map(|points| ArcString::try_from(points.to_owned()).unwrap())
            .collect();
        Self::try_from(arcstrings).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl TryFrom<Vec<Array2<f64>>> for MultiArcString {
    type Error = String;

    fn try_from(xyzs: Vec<Array2<f64>>) -> Result<Self, Self::Error> {
        let mut arcstrings = vec![];
        for xyz in xyzs {
            arcstrings.push(ArcString::try_from(MultiSphericalPoint::try_from(xyz)?)?);
        }
        Self::try_from(arcstrings)
    }
}

impl From<MultiArcString> for Vec<MultiSphericalPoint> {
    fn from(arcstrings: MultiArcString) -> Self {
        arcstrings
            .arcstrings
            .into_iter()
            .map(|arcstring| arcstring.points)
            .collect()
    }
}

impl From<MultiArcString> for Vec<ArcString> {
    fn from(arcstrings: MultiArcString) -> Self {
        arcstrings.arcstrings
    }
}

impl MultiArcString {
    pub fn midpoints(&self) -> MultiSphericalPoint {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.midpoints())
            .sum()
    }

    pub fn lengths(&self) -> Vec<f64> {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.length())
            .collect()
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

impl Add<Self> for &MultiArcString {
    type Output = MultiArcString;

    fn add(self, rhs: Self) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&Self> for MultiArcString {
    fn add_assign(&mut self, other: &Self) {
        self.extend(other.to_owned());
    }
}

impl Add<&ArcString> for &MultiArcString {
    type Output = MultiArcString;

    fn add(self, rhs: &ArcString) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&ArcString> for MultiArcString {
    fn add_assign(&mut self, other: &ArcString) {
        self.push(other.to_owned());
    }
}

impl Geometry for MultiArcString {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.to_owned().points)
            .sum()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        Some(
            self.arcstrings
                .iter()
                .filter_map(|arcstring| arcstring.boundary())
                .sum(),
        )
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.arcstrings[0].representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.vertices().centroid()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.vertices().convex_hull()
    }

    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.length())
            .sum()
    }
}

impl Sum for MultiArcString {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut arcstrings = vec![];
        for multiarcstring in iter {
            arcstrings.extend(multiarcstring.arcstrings);
        }
        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl MultiGeometry<ArcString> for MultiArcString {
    fn len(&self) -> usize {
        self.arcstrings.len()
    }

    fn extend(&mut self, other: Self) {
        self.arcstrings.extend(other.arcstrings);
    }

    fn push(&mut self, other: ArcString) {
        self.arcstrings.push(other);
    }
}

impl GeometricPredicates<SphericalPoint> for MultiArcString {
    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }
}

impl GeometricOperations<SphericalPoint, ArcString> for MultiArcString {
    fn union(&self, _: &SphericalPoint) -> Option<Self> {
        None
    }

    fn distance(&self, other: &SphericalPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn symmetric_difference(&self, other: &SphericalPoint) -> Self {
        let mut arcstrings = vec![];
        for arcstring in &self.arcstrings {
            arcstrings.extend(split_arcstring_at_points(arcstring, vec![&other.xyz]).arcstrings);
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricPredicates<MultiSphericalPoint> for MultiArcString {
    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.contains(other))
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        self.contains(other)
    }
}

impl GeometricOperations<MultiSphericalPoint, ArcString> for MultiArcString {
    fn union(&self, _: &MultiSphericalPoint) -> Option<Self> {
        None
    }

    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(MultiSphericalPoint::from(&intersections))
        } else {
            None
        }
    }

    fn symmetric_difference(&self, other: &MultiSphericalPoint) -> Self {
        let mut arcstrings = vec![];
        for arcstring in &self.arcstrings {
            arcstrings.extend(
                split_arcstring_at_points(arcstring, other.xyzs.iter().collect()).arcstrings,
            );
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricPredicates<ArcString> for MultiArcString {
    fn intersects(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn within(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn contains(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn overlaps(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.overlaps(other))
    }

    fn covers(&self, other: &ArcString) -> bool {
        // TODO: handle case where adjoining arcstrings in this multiarcstring jointly cover the other arcstring
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.covers(other))
    }
}

impl GeometricOperations<ArcString, ArcString> for MultiArcString {
    fn union(&self, other: &ArcString) -> Option<Self> {
        Some(self + other)
    }

    fn distance(&self, other: &ArcString) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, other: &ArcString) -> Self {
        let mut arcstrings = vec![];
        if let Some(points) = self.intersection(other) {
            for arcstring in &self.arcstrings {
                arcstrings.extend(
                    split_arcstring_at_points(arcstring, points.xyzs.iter().collect()).arcstrings,
                );
            }
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricPredicates<Self> for MultiArcString {
    fn intersects(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &Self) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn within(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn contains(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.overlaps(other))
    }

    fn covers(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.covers(other))
    }
}

impl GeometricOperations<Self, ArcString> for MultiArcString {
    fn union(&self, other: &Self) -> Option<Self> {
        Some(self + other)
    }

    fn distance(&self, other: &Self) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &Self) -> Option<MultiSphericalPoint> {
        let intersections: Vec<MultiSphericalPoint> = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.intersection(other))
            .collect();

        if !intersections.is_empty() {
            Some(intersections.into_iter().sum())
        } else {
            None
        }
    }

    fn symmetric_difference(&self, other: &Self) -> Self {
        let mut arcstrings = vec![];
        if let Some(points) = self.intersection(other) {
            for arcstring in &self.arcstrings {
                arcstrings.extend(
                    split_arcstring_at_points(arcstring, points.xyzs.iter().collect()).arcstrings,
                );
            }
        }

        Self::try_from(arcstrings).unwrap()
    }
}

impl GeometricPredicates<crate::sphericalpolygon::SphericalPolygon> for MultiArcString {
    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn contains(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon, ArcString> for MultiArcString {
    fn union(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> Option<Self> {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn symmetric_difference(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> Self {
        let mut arcstrings = vec![];
        if let Some(points) = self.intersection(&other.boundary) {
            for arcstring in &self.arcstrings {
                arcstrings.extend(
                    split_arcstring_at_points(arcstring, points.xyzs.iter().collect()).arcstrings,
                );
            }
        }

        Self::try_from(arcstrings).unwrap()
    }
}

impl GeometricPredicates<crate::sphericalpolygon::MultiSphericalPolygon> for MultiArcString {
    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn contains(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn covers(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon, ArcString>
    for MultiArcString
{
    fn union(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> Option<Self> {
        None
    }

    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersection(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> Option<Self> {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn symmetric_difference(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> Self {
        let mut arcstrings = vec![];
        if let Some(other_boundary) = other.boundary() {
            if let Some(points) = self.intersection(&other_boundary) {
                for arcstring in &self.arcstrings {
                    arcstrings.extend(
                        split_arcstring_at_points(arcstring, points.xyzs.iter().collect())
                            .arcstrings,
                    );
                }
            }
        }

        Self::try_from(arcstrings).unwrap()
    }
}

impl GeometryCollection<ArcString> for MultiArcString {
    fn join_self(&self) -> Self {
        let mut graph = EdgeGraph::<ArcString>::from(self);
        graph.split_edges();
        graph.remove_multisourced_edges();
        graph.remove_degenerate_edges();

        Self::try_from(Vec::<ArcString>::from(graph)).unwrap()
    }

    fn overlap_self(&self) -> Option<Self> {
        let mut graph = EdgeGraph::<ArcString>::from(self);
        graph.split_edges();
        graph.remove_unisourced_edges();
        graph.remove_degenerate_edges();

        Self::try_from(Vec::<ArcString>::from(graph)).ok()
    }

    fn symmetric_difference_self(&self) -> Option<Self> {
        let mut split_graph = EdgeGraph::<ArcString>::from(self);
        split_graph.split_edges();

        let mut overlap_graph = split_graph.to_owned();
        overlap_graph.remove_unisourced_edges();
        overlap_graph.remove_degenerate_edges();

        let mut arcstrings = Vec::<ArcString>::from(split_graph);
        arcstrings.extend(Vec::<ArcString>::from(overlap_graph));

        Self::try_from(arcstrings).ok()
    }
}
