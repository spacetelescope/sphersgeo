use crate::{
    edgegraph::EdgeGraph,
    geometry::{
        GeometricOperations, GeometricRelationships, Geometry, GeometryCollection, MultiGeometry,
        MultiGeometryUnaryOperations,
    },
    sphericalpoint::{
        MultiSphericalPoint, SphericalPoint, arc_distance_over_sphere, point_within_kdtree,
        xyz_add_xyz, xyz_cross, xyz_div_f64, xyz_dot, xyz_eq, xyz_mul_xyz, xyz_neg, xyz_sub_xyz,
        xyz_sum, xyzs_coplanar,
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
use numpy::ndarray::{Array2, ArrayView2, Axis, concatenate, s};

/// Given xyz vectors of two great circle arcs, find the point at which the arcs cross
///
/// References
/// ----------
/// - Method explained in an `e-mail <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_ by Roger Stafford.
/// - https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
/// - Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
pub fn arcs_crossing(
    arc_a: (&[f64; 3], &[f64; 3]),
    arc_b: (&[f64; 3], &[f64; 3]),
) -> Option<[f64; 3]> {
    let p = xyz_cross(arc_a.0, arc_a.1);
    let q = xyz_cross(arc_b.0, arc_b.1);

    let t = xyz_cross(&p, &q);

    let result = [
        xyz_dot(&xyz_neg(&xyz_cross(arc_a.0, &p)), &t),
        xyz_dot(&xyz_cross(arc_a.1, &p), &t),
        xyz_dot(&xyz_neg(&xyz_cross(arc_b.0, &q)), &t),
        xyz_dot(&xyz_cross(arc_b.1, &q), &t),
    ];

    if result.iter().all(|result| result.is_sign_positive()) {
        Some(t)
    } else if result.iter().all(|sign| sign.is_sign_negative()) {
        Some(xyz_neg(&t))
    } else {
        None
    }
}

pub fn arc_crossings_with_arcstring(
    arc: (&[f64; 3], &[f64; 3]),
    arcstring: &ArcString,
) -> Option<Vec<[f64; 3]>> {
    let mut crossings = vec![];
    for other_arc_index in 0..arcstring.points.xyzs.len() - if arcstring.closed { 0 } else { 1 } {
        let other_arc = (
            &arcstring.points.xyzs[other_arc_index],
            &arcstring.points.xyzs[if other_arc_index < arcstring.points.xyzs.len() - 1 {
                other_arc_index + 1
            } else {
                0
            }],
        );
        if let Some(point) = arcs_crossing(arc, other_arc) {
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

    if !crossings.is_empty() {
        Some(crossings)
    } else {
        None
    }
}

/// whether two points are on the same side of an arcstring
///
/// uses the classical even-crossings algorithm;
/// if the number of crossings between the arc and the arcstring is even,
/// then the two points are on the same side of the arcstring
pub fn points_are_on_same_side(
    point_a: &[f64; 3],
    point_b: &[f64; 3],
    arcstring: &ArcString,
) -> bool {
    arc_crossings_with_arcstring((point_a, point_b), arcstring)
        .is_none_or(|crossings| crossings.len() % 2 == 0)
}

pub fn point_is_along_arcstring(xyz: &[f64; 3], arcstring: &ArcString) -> bool {
    let xyzs = &arcstring.points.xyzs;

    // check if point is one of the vertices of this linestring
    if point_within_kdtree(xyz, &arcstring.points.kdtree) {
        return true;
    }

    // iterate over individual arcs and check if the given point is colinear with their endpoints
    for arc_index in 0..xyzs.len() - if arcstring.closed { 0 } else { 1 } {
        let arc_0 = xyzs[arc_index];
        let arc_1 = xyzs[if arc_index < xyzs.len() - 1 {
            arc_index + 1
        } else {
            0
        }];

        if xyzs_coplanar(&arc_0, xyz, &arc_1) {
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

            if xyzs_coplanar(arc_0, point, arc_1) {
                // replace arc with the arc split in two at the colinear point
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
            for arc_a_index in 0..arcstring.points.len() - if arcstring.closed { 0 } else { 1 } {
                let arc_b_index = if arc_a_index < arcstring.points.len() - 1 {
                    arc_a_index + 1
                } else {
                    // if the index is greater than the length, the arcstring is closed and we should loop back to the start
                    0
                };

                let arc_0 = arcstring.points.xyzs[arc_a_index];
                let arc_1 = arcstring.points.xyzs[arc_b_index];

                if xyzs_coplanar(&arc_0, point, &arc_1) {
                    // replace arc with the arc split in two at the colinear point

                    // add the first segment up to the colinear point
                    let mut a = vec![];
                    a.extend_from_slice(&arcstring.points.xyzs[..arc_a_index + 1]);
                    a.push(**point);
                    arcstrings[arcstring_index] =
                        ArcString::try_from(MultiSphericalPoint::try_from(a).unwrap()).unwrap();

                    // add the second segment starting from the colinear point
                    let mut b = vec![**point];
                    if arc_b_index > 0 {
                        b.extend_from_slice(&arcstring.points.xyzs[arc_b_index..]);
                    } else {
                        // handle case where end point is the start point of the arcstring
                        b.push(arcstring.points.xyzs[arc_b_index]);
                    }
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
fn arc_closest_distance_to_point(a: &[f64; 3], b: &[f64; 3], xyz: &[f64; 3]) -> ([f64; 3], f64) {
    let g = xyz_cross(a, b);
    let f = xyz_cross(xyz, &g);
    let t = xyz_cross(&g, &f);
    (t, arc_distance_over_sphere(&t, xyz))
}

/// Inner products of the normal vectors at each vertex of the given arcstring.
///
/// Normal vectors point into the sphere if the angle is clockwise and outward if counter-clockwise.
/// Thus, a negative inner product is a right turn, and positive is left.
///
/// If the arcstring is NOT closed the first and last vertices do NOT have turn orientations.
pub fn xyzs_turn_orientations(xyzs: &[[f64; 3]], closed: bool) -> Vec<f64> {
    (0..xyzs.len())
        .map(|index| {
            // if the arcstring is not closed, the first and last points have no turning angle
            if !closed && (index == 0 || index == xyzs.len() - 1) {
                f64::NAN
            } else {
                let a = xyzs[if index > 0 { index - 1 } else { xyzs.len() - 1 }];
                let b = xyzs[index];
                let c = xyzs[if index < xyzs.len() - 1 { index + 1 } else { 0 }];

                xyz_sum(&xyz_mul_xyz(
                    &b,
                    &xyz_cross(&xyz_sub_xyz(&a, &b), &xyz_sub_xyz(&c, &b)),
                ))
            }
        })
        .collect()
}

/// Angle in radians at each vertex of the given arcstring (the smaller angle regardless of turn orientation).
///
/// If the arcstring is NOT closed the first and last vertices do NOT have angles.
pub fn xyzs_turn_angles(xyzs: &[[f64; 3]], closed: bool) -> Vec<f64> {
    (0..xyzs.len())
        .map(|index| {
            // if the arcstring is not closed, the first and last points have no turning angle
            if !closed && (index == 0 || index == xyzs.len() - 1) {
                f64::NAN
            } else {
                let a = xyzs[if index > 0 { index - 1 } else { xyzs.len() - 1 }];
                let b = xyzs[index];
                let c = xyzs[if index < xyzs.len() - 1 { index + 1 } else { 0 }];

                crate::sphericalpoint::xyz_two_arc_angle(&a, &b, &c)
            }
        })
        .collect()
}

/// series of great circle arcs along the sphere
#[cfg_attr(feature = "py", pyclass(from_py_object))]
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

    /// degrees subtended on the sphere by each arc
    pub fn lengths(&self) -> Vec<f64> {
        let mut lengths = (0..self.points.len() - 1)
            .map(|index| {
                arc_distance_over_sphere(&self.points.xyzs[index], &self.points.xyzs[index + 1])
                    .to_degrees()
            })
            .collect::<Vec<f64>>();

        if self.closed {
            // if the arcstring is closed, also add the length of the final closing arc
            lengths.push(arc_distance_over_sphere(
                &self.points.xyzs[self.points.len() - 1],
                &self.points.xyzs[0],
            ));
        }

        lengths
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

    /// whether this arcstring crosses itself
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
                    if let Some(point) = arcs_crossing(arc, other_arc) {
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

    /// points at which this arcstring crosses itself
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

                    if let Some(point) = arcs_crossing(arc, other_arc) {
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

    /// whether this arcstring shares endpoints with another, ignoring closed arcstrings
    pub fn adjoins(&self, other: &ArcString) -> bool {
        if let Some(boundary) = self.boundary() {
            if let Some(other_boundary) = other.boundary() {
                return boundary.touches(&other_boundary);
            }
        }

        false
    }

    /// remove redundant vertices that already lie along the arcstring
    pub fn simplify(&mut self) {
        loop {
            if self.points.xyzs.len() <= 2 {
                // can't simplify a line with only two points
                break;
            }

            let mut unecessary_indices = vec![];
            for index in 1..self.points.xyzs.len() + if self.closed { 1 } else { 0 } {
                let index = if index < self.points.xyzs.len() {
                    index
                } else {
                    // if the index is greater than the length, the arcstring is closed and we should loop back to the start
                    self.points.xyzs.len() - index + 1
                };

                let a = self.points.xyzs[index - 1];
                let b = self.points.xyzs[index];
                let c = self.points.xyzs[if index + 1 < self.points.xyzs.len() {
                    index + 1
                } else {
                    // if the index is greater than the length, the arcstring is closed and we should loop back to the start
                    self.points.xyzs.len() - index + 1
                }];

                if xyzs_coplanar(&a, &b, &c) {
                    unecessary_indices.push(index);
                }
            }

            if unecessary_indices.is_empty() {
                break;
            } else {
                for index in unecessary_indices.iter().rev() {
                    self.points.xyzs.remove(*index);
                }
            }
        }
    }

    pub fn insert_vertices(&mut self, points: &Vec<&SphericalPoint>) {
        let vertices = &mut self.points.xyzs;
        for point in points {
            let mut vertex_index = 0;
            while vertex_index < vertices.len() - if self.closed { 0 } else { 1 } {
                let next_vertex_index = if vertex_index < vertices.len() - 1 {
                    vertex_index + 1
                } else {
                    // if the index is greater than the length, the arcstring is closed and we should loop back to the start
                    0
                };

                let arc_0 = vertices[vertex_index];
                let arc_1 = vertices[next_vertex_index];

                if xyzs_coplanar(&arc_0, &point.xyz, &arc_1) {
                    // insert the new vertex in between the existing ones
                    vertices.insert(vertex_index, point.xyz);
                    vertex_index += 1;
                }

                vertex_index += 1;
            }
        }
    }
}

impl PartialEq for ArcString {
    fn eq(&self, other: &Self) -> bool {
        if self.boundary() != other.boundary() {
            return false;
        }

        let (shorter, longer) = if self.points.len() < other.points.len() {
            (self, other)
        } else {
            (other, self)
        };

        for xyz in &longer.points.xyzs {
            if !point_is_along_arcstring(xyz, shorter) {
                return false;
            }
        }

        for xyz in &shorter.points.xyzs {
            if !point_is_along_arcstring(xyz, longer) {
                return false;
            }
        }

        true
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
        Self::Output {
            arcstrings: vec![self.to_owned(), rhs.to_owned()],
        }
    }
}

impl Geometry for ArcString {
    fn vertices(&self) -> MultiSphericalPoint {
        self.points.to_owned()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        if self.closed {
            None
        } else {
            Some(
                MultiSphericalPoint::try_from(vec![
                    self.points.xyzs[0],
                    self.points.xyzs[self.points.len() - 1],
                ])
                .unwrap(),
            )
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

    fn to_wkt(&self, angular: bool) -> String {
        let mut points = self.points.to_owned();
        if self.closed {
            // wrap around original point if closed
            points.xyzs.push(points.xyzs[0]);
        }
        points.to_wkt(angular).replace("MULTIPOINT", "LINESTRING")
    }
}

impl GeometricRelationships<SphericalPoint> for ArcString {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        let mut distances = (0..self.points.len() - 1)
            .map(|index| {
                arc_closest_distance_to_point(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                    &other.xyz,
                )
                .1
            })
            .collect::<Vec<f64>>();

        if self.closed {
            // if the arcstring is closed, also add the midpoint of the final closing arc
            distances.push(
                arc_closest_distance_to_point(
                    &self.points.xyzs[self.points.len() - 1],
                    &self.points.xyzs[0],
                    &other.xyz,
                )
                .1,
            );
        }

        match distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(distance) => *distance,
            None => f64::NAN,
        }
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.points.touches(other) || self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        if self.covers(other) {
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
        point_is_along_arcstring(&other.xyz, self)
    }
}

impl GeometricRelationships<MultiSphericalPoint> for ArcString {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        let mut distances = vec![];
        for xyz in &other.xyzs {
            distances.extend((0..self.points.len() - 1).map(|index| {
                arc_closest_distance_to_point(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                    xyz,
                )
                .1
            }));

            if self.closed {
                // if the arcstring is closed, also add the midpoint of the final closing arc
                distances.push(
                    arc_closest_distance_to_point(
                        &self.points.xyzs[self.points.len() - 1],
                        &self.points.xyzs[0],
                        xyz,
                    )
                    .1,
                );
            }
        }

        match distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(distance) => *distance,
            None => f64::NAN,
        }
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.points.touches(other) || self.contains(other)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        if self.covers(other) {
            // endpoints are not part of the interior of an arcstring
            if let Some(boundary) = self.boundary() {
                if boundary.intersects(other) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        for xyz in &other.xyzs {
            if point_is_along_arcstring(xyz, self) {
                return true;
            }
        }
        false
    }
}

impl GeometricRelationships<Self> for ArcString {
    fn distance(&self, other: &Self) -> f64 {
        todo!()
    }

    fn equals(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }

        let (shorter, longer) = if self.points.len() < other.points.len() {
            (self, other)
        } else {
            (other, self)
        };

        let mut simple_shorter = shorter.to_owned();
        simple_shorter.simplify();

        if &simple_shorter == longer {
            return true;
        }

        let mut simple_longer = longer.to_owned();
        simple_longer.simplify();

        simple_shorter == simple_longer
    }

    fn intersects(&self, other: &Self) -> bool {
        self.overlaps(other) || self.crosses(other) || self.equals(other)
    }

    fn touches(&self, other: &Self) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
    }

    fn crosses(&self, other: &Self) -> bool {
        if self.within(other) || self.contains(other) {
            return false;
        }

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed connected space so there's no good way to sort by longitude
        // so I guess the best we can do instead is use brute-force
        for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
            if arc_crossings_with_arcstring(
                (
                    &self.points.xyzs[arc_index],
                    &self.points.xyzs[if arc_index < self.points.len() - 1 {
                        arc_index + 1
                    } else {
                        0
                    }],
                ),
                other,
            )
            .is_some()
            {
                return true;
            }
        }

        false
    }

    fn within(&self, other: &Self) -> bool {
        self.points
            .xyzs
            .iter()
            .all(|xyz| point_is_along_arcstring(xyz, other))
    }

    fn contains(&self, other: &Self) -> bool {
        other.within(self)
    }

    fn overlaps(&self, other: &Self) -> bool {
        let mut simple_self = self.to_owned();
        simple_self.simplify();

        let mut simple_other = other.to_owned();
        simple_other.simplify();

        if simple_self.equals(&simple_other) && !self.within(other) && !self.contains(other) {
            for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
                let arc = (
                    self.points.xyzs[arc_index],
                    self.points.xyzs[if arc_index < self.points.len() {
                        arc_index + 1
                    } else {
                        0
                    }],
                );

                // TODO: handle case where an arcstring has both endpoints on the other arcstring, but cuts a corner...
                if point_is_along_arcstring(&arc.0, other)
                    && point_is_along_arcstring(&arc.1, other)
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

impl GeometricRelationships<MultiArcString> for ArcString {
    fn distance(&self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn equals(&self, other: &MultiArcString) -> bool {
        other.equals(self)
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
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

impl GeometricRelationships<crate::sphericalpolygon::SphericalPolygon> for ArcString {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        if self.within(other) {
            0.0
        } else {
            self.distance(&other.boundary)
        }
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.crosses(&other.boundary)
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }
}

impl GeometricRelationships<crate::sphericalpolygon::MultiSphericalPolygon> for ArcString {
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        if let Some(boundary) = self.boundary() {
            boundary.touches(other)
        } else {
            false
        }
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.crosses(self)
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }
}

impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for ArcString {
    fn intersection(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::sphericalpoint::SphericalPoint) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpoint::MultiSphericalPoint> for ArcString {
    fn intersection(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::MultiSphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<Self> for ArcString {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        let mut crossings = vec![];

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

                if let Some(point) = arcs_crossing(arc, other_arc) {
                    crossings.push(point);
                }
            }
        }

        crate::geometry::GeometryCollection {
            points: MultiSphericalPoint::try_from(crossings).ok(),
            strings: todo!(),
            polygons: None,
        }
    }

    fn difference(&self, other: &Self) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &Self) -> GeometryCollection {
        GeometryCollection {
            points: None,
            strings: if self.closed || other.closed {
                Some(self + other)
            } else {
                let mut graph = EdgeGraph::<Self>::from(vec![self, other]);
                graph.split_edges();
                graph.remove_degenerate_edges();

                MultiArcString::try_from(Vec::<ArcString>::from(graph)).ok()
            },
            polygons: None,
        }
    }
}

impl GeometricOperations<MultiArcString> for ArcString {
    fn intersection(&self, other: &MultiArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &MultiArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon> for ArcString {
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon> for ArcString {
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

/// collection of arcstrings
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Debug, Clone)]
pub struct MultiArcString {
    pub arcstrings: Vec<ArcString>,
}

impl From<ArcString> for MultiArcString {
    fn from(arcstring: ArcString) -> Self {
        Self::try_from(vec![arcstring]).unwrap()
    }
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
            .into_iter()
            .map(|points| ArcString::try_from(points).unwrap())
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
    pub fn insert_vertices(&mut self, points: &Vec<&SphericalPoint>) {
        for arcstring in self.arcstrings.iter_mut() {
            arcstring.insert_vertices(points);
        }
    }
}

impl Display for MultiArcString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiArcString({:?})", self.arcstrings)
    }
}

impl PartialEq for MultiArcString {
    fn eq(&self, other: &Self) -> bool {
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
        let boundaries = self
            .arcstrings
            .iter()
            .filter_map(|arcstring| arcstring.boundary())
            .collect::<Vec<MultiSphericalPoint>>();
        if !boundaries.is_empty() {
            Some(boundaries.into_iter().sum())
        } else {
            None
        }
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

    fn to_wkt(&self, angular: bool) -> String {
        format!(
            "MULTILINESTRING ({})",
            self.arcstrings
                .iter()
                .map(|arcstring| arcstring.to_wkt(angular).replace("LINESTRING ", ""))
                .collect::<Vec<String>>()
                .join("), (")
        )
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

impl MultiGeometryUnaryOperations<ArcString> for MultiArcString {
    fn unary_union(&self) -> Self {
        // add all nodes to graph
        let mut graph =
            EdgeGraph::<ArcString>::from(self.arcstrings.iter().collect::<Vec<&ArcString>>());

        // split edges based on overlapping nodes / intersecting edges
        graph.split_edges();

        // prune 0-length edges and nodes without edges
        graph.remove_degenerate_edges();
        graph.remove_orphaned_nodes();

        // trace arcstrings from graph
        Self::try_from(Vec::<ArcString>::from(graph)).unwrap()
    }

    fn unary_intersection(&self) -> Option<Self> {
        // add all nodes to graph
        let mut graph =
            EdgeGraph::<ArcString>::from(self.arcstrings.iter().collect::<Vec<&ArcString>>());

        // split edges based on overlapping nodes / intersecting edges
        graph.split_edges();

        // remove all edges that only came from a single arcstring
        graph.remove_unisourced_edges();

        // prune 0-length edges and nodes without edges
        graph.remove_degenerate_edges();
        graph.remove_orphaned_nodes();

        Self::try_from(Vec::<ArcString>::from(graph)).ok()
    }

    fn unary_symmetric_difference(&self) -> Option<Self> {
        // add all nodes to graph
        let mut graph =
            EdgeGraph::<ArcString>::from(self.arcstrings.iter().collect::<Vec<&ArcString>>());

        // split edges based on overlapping nodes / intersecting edges
        graph.split_edges();

        // remove all edges shared by multiple arcstrings
        graph.remove_multisourced_edges();

        // prune 0-length edges and nodes without edges
        graph.remove_degenerate_edges();
        graph.remove_orphaned_nodes();

        Self::try_from(Vec::<ArcString>::from(graph)).ok()
    }
}

impl GeometricRelationships<SphericalPoint> for MultiArcString {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
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

impl GeometricRelationships<MultiSphericalPoint> for MultiArcString {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
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

impl GeometricRelationships<ArcString> for MultiArcString {
    fn distance(&self, other: &ArcString) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn equals(&self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.equals(other))
    }

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

impl GeometricRelationships<Self> for MultiArcString {
    fn distance(&self, other: &Self) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn equals(&self, other: &Self) -> bool {
        let (shorter, longer) = if self.arcstrings.len() < other.arcstrings.len() {
            (self, other)
        } else {
            (other, self)
        };

        shorter.arcstrings.iter().all(|shorter_arcstring| {
            longer
                .arcstrings
                .iter()
                .any(|longer_arcstring| shorter_arcstring.equals(longer_arcstring))
        })
    }

    fn intersects(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &Self) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
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

impl GeometricRelationships<crate::sphericalpolygon::SphericalPolygon> for MultiArcString {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
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
}

impl GeometricRelationships<crate::sphericalpolygon::MultiSphericalPolygon> for MultiArcString {
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.touches(other))
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
}

impl GeometricOperations<crate::sphericalpoint::SphericalPoint, ArcString> for MultiArcString {
    fn intersection(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::sphericalpoint::SphericalPoint) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpoint::MultiSphericalPoint, ArcString> for MultiArcString {
    fn intersection(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::MultiSphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<ArcString, ArcString> for MultiArcString {
    fn intersection(&self, other: &ArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &ArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<Self, ArcString> for MultiArcString {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &Self) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon, ArcString> for MultiArcString {
    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon, ArcString>
    for MultiArcString
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
    ) -> Option<MultiArcString> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}
