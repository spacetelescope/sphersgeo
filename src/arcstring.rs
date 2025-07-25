use crate::{
    edgegraph::{EdgeGraph, GeometryGraph, ToGraph},
    geometry::{GeometricOperations, Geometry, GeometryCollection, MultiGeometry},
    sphericalpoint::{
        point_within_kdtree, xyz_add_xyz, xyz_cross, xyz_div_f64, xyz_dot, xyz_eq, xyz_neg,
        xyz_radians_over_sphere_between, xyzs_collinear, MultiSphericalPoint, SphericalPoint,
    },
};
use numpy::ndarray::{array, concatenate, s, Array1, Array2, ArrayView2, Axis};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt::Display;
use std::iter::Sum;

pub fn xyz_interpolate_between(
    a: &[f64; 3],
    b: &[f64; 3],
    n: usize,
) -> Result<Vec<[f64; 3]>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::linspace(0.0, 1.0, n);
    let t = t.to_shape((n, 1)).unwrap();
    let omega = crate::sphericalpoint::xyz_radians_over_sphere_between(a, b);

    let offsets = if omega == 0.0 {
        t.to_owned()
    } else {
        (t * omega).sin() / omega.sin()
    };
    let mut inverted_offsets = offsets.to_owned();
    inverted_offsets.invert_axis(Axis(0));

    Ok(
        (inverted_offsets * array![a[0], a[1], a[2]] + offsets * array![b[0], b[1], b[2]])
            .rows()
            .into_iter()
            .map(|xyz| [xyz[0], xyz[1], xyz[2]])
            .collect(),
    )
}

/// Given xyz vectors of the endpoints of two great circle arcs, find the point at which the arcs cross
///
/// References
/// ----------
/// - Method explained in an `e-mail <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_ by Roger Stafford.
/// - https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
/// - Spinielli, Enrico. 2014. “Understanding Great Circle Arcs Intersection Algorithm.” October 19, 2014. https://enrico.spinielli.net/posts/2014-10-19-understanding-great-circle-arcs.
pub fn xyz_two_arc_crossing(
    a_0: &[f64; 3],
    a_1: &[f64; 3],
    b_0: &[f64; 3],
    b_1: &[f64; 3],
) -> Option<[f64; 3]> {
    let p = xyz_cross(a_0, a_1);
    let q = xyz_cross(b_0, b_1);

    let t = xyz_cross(&p, &q);

    let signs = array![
        xyz_dot(&xyz_neg(&xyz_cross(a_0, &p)), &t),
        xyz_dot(&xyz_cross(a_1, &p), &t),
        xyz_dot(&xyz_neg(&xyz_cross(b_0, &q)), &t),
        xyz_dot(&xyz_cross(b_1, &q), &t),
    ]
    .signum();

    if signs.iter().all(|sign| sign.is_sign_positive()) {
        Some(t)
    } else if signs.iter().all(|sign| sign.is_sign_negative()) {
        Some([-t[0], -t[1], -t[2]])
    } else {
        None
    }
}

pub fn arcstring_contains_point(arcstring: &ArcString, xyz: &[f64; 3]) -> bool {
    let xyzs = &arcstring.points.xyzs;

    // if the arcstring is not closed, make sure the point is not one of the terminal endpoints
    if !arcstring.closed {
        let start = xyzs[0];
        let end = xyzs[xyzs.len() - 1];
        if xyz_eq(&start, xyz) || xyz_eq(xyz, &end) {
            return false;
        }
    }

    // check if point is one of the vertices of this linestring
    if point_within_kdtree(xyz, &arcstring.points.kdtree) {
        return true;
    }

    // iterate over individual arcs and check if the given point is collinear with their endpoints
    for arc_index in 0..xyzs.len() - if arcstring.closed { 0 } else { 1 } {
        let arc_0 = xyzs[arc_index];
        let arc_1 = xyzs[if arc_index == xyzs.len() - 1 {
            0
        } else {
            arc_index + 1
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
            if xyz_eq(&arc_0, point) || xyz_eq(point, &arc_1) {
                continue;
            }

            if xyzs_collinear(&arc_0, point, &arc_1) {
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
                let arc_1 = points[if arc_index == arcstring.points.len() - 1 {
                    0
                } else {
                    arc_index + 1
                }];

                if xyzs_collinear(&arc_0, point, &arc_1) {
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
    xyz_radians_over_sphere_between(&t, xyz)
}

/// series of great circle arcs along the sphere
#[pyclass]
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
            let (points, closed) = if xyz_eq(&points.xyzs[0], &points.xyzs[num_points - 1]) {
                (
                    MultiSphericalPoint::try_from(points.xyzs[..num_points].to_vec())?,
                    true,
                )
            } else {
                (points, false)
            };

            Ok(Self { points, closed })
        }
    }
}

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
                if (&edges[edge_index].slice(s![0, ..]) - &end).abs().sum() < 2e-8 {
                    connected = concatenate![
                        Axis(0),
                        connected.view(),
                        edges[edge_index].slice(s![1.., ..])
                    ];
                } else if (&edges[edge_index].slice(s![edges[edge_index].nrows() - 1, ..]) - &end)
                    .abs()
                    .sum()
                    < 2e-8
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
                        MultiSphericalPoint::try_from(if index == arcstring.points.len() - 1 {
                            // add additional edge returning to initial point
                            vec![arcstring.points.xyzs[index], arcstring.points.xyzs[0]]
                        } else {
                            arcstring.points.xyzs[index..index + 2].to_vec()
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
    pub fn new(points: MultiSphericalPoint, closed: bool) -> Result<Self, String> {
        let mut instance = Self::try_from(points)?;
        instance.closed = closed;
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
            let tolerance = 1e-11;

            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so I guess the best we can do instead is use brute-force and skip visited arcs
            for arc_index in 0..self.points.len() - 1 {
                let arc_start = self.points.xyzs[arc_index];
                let arc_end = self.points.xyzs[arc_index + 1];

                // due to the nature of the search we can assume that previous indices are already checked
                for other_arc_index in
                    arc_index + 2..self.points.len() - if self.closed { 0 } else { 1 }
                {
                    let other_arc_start = self.points.xyzs[other_arc_index];
                    let other_arc_end = self.points.xyzs[if other_arc_index == self.points.len() - 1
                    {
                        0
                    } else {
                        other_arc_index + 1
                    }];
                    if let Some(point) =
                        xyz_two_arc_crossing(&arc_start, &arc_end, &other_arc_start, &other_arc_end)
                    {
                        if xyz_eq(&point, &arc_start)
                            || xyz_eq(&point, &arc_end)
                            || xyz_eq(&point, &other_arc_start)
                            || xyz_eq(&point, &other_arc_end)
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
                let arc_start = self.points.xyzs[arc_index];
                let arc_end = self.points.xyzs[arc_index + 1];

                // due to the nature of the search we can assume that previous indices are already checked
                for other_arc_index in arc_index + 2..self.points.len() - 1 {
                    let other_arc_start = self.points.xyzs[other_arc_index];
                    let other_arc_end = self.points.xyzs[other_arc_index + 1];

                    if let Some(point) =
                        xyz_two_arc_crossing(&arc_start, &arc_end, &other_arc_start, &other_arc_end)
                    {
                        if xyz_eq(&point, &arc_start)
                            || xyz_eq(&point, &arc_end)
                            || xyz_eq(&point, &other_arc_start)
                            || xyz_eq(&point, &other_arc_end)
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
                xyz_radians_over_sphere_between(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                )
            })
            .collect::<Vec<f64>>();

        if self.closed {
            // if the arcstring is closed, also add the length of the final closing arc
            lengths.push(xyz_radians_over_sphere_between(
                &self.points.xyzs[self.points.len() - 1],
                &self.points.xyzs[0],
            ));
        }

        lengths
    }

    /// whether this arcstring shares endpoints with another, ignoring closed arcstrings
    pub fn adjoins(&self, other: &ArcString) -> bool {
        if !self.closed && !other.closed {
            let start = self.points.xyzs[0];
            let end = self.points.xyzs[self.points.len() - 1];

            let other_start = other.points.xyzs[0];
            let other_end = other.points.xyzs[self.points.len() - 1];

            xyz_eq(&end, &other_start)
                || xyz_eq(&other_end, &start)
                || xyz_eq(&end, &other_end)
                || xyz_eq(&start, &other_start)
        } else {
            false
        }
    }

    /// join this arcstring to another
    pub fn join(&self, other: &ArcString) -> Option<ArcString> {
        if self.closed || other.closed {
            None
        } else {
            let mut graph = EdgeGraph::<Self>::from(vec![self, other]);
            graph.split_edges();
            graph.prune_overlapping_edges();
            graph.prune_degenerate_edges();

            let arcstrings = graph.find_disjoint_geometries();
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
        self.lengths().iter().sum()
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.centroid()
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

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative()
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

impl GeometricOperations<SphericalPoint> for ArcString {
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

    fn contains(&self, other: &SphericalPoint) -> bool {
        arcstring_contains_point(self, &other.xyz)
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        if self.contains(other) {
            Some(other.to_owned())
        } else {
            None
        }
    }

    fn split(&self, other: &SphericalPoint) -> MultiArcString {
        split_arcstring_at_points(self, vec![&other.xyz])
    }
}

impl GeometricOperations<MultiSphericalPoint> for ArcString {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        let mut distances = vec![];
        for xyz in &other.xyzs {
            distances.extend((0..self.points.len() - 1).map(|index| {
                arc_radians_over_sphere_to_point(
                    &self.points.xyzs[index],
                    &self.points.xyzs[index + 1],
                    &xyz,
                )
            }));

            if self.closed {
                // if the arcstring is closed, also add the midpoint of the final closing arc
                distances.push(arc_radians_over_sphere_to_point(
                    &self.points.xyzs[self.points.len() - 1],
                    &self.points.xyzs[0],
                    &xyz,
                ));
            }
        }

        match distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(distance) => *distance,
            None => f64::NAN,
        }
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        // check if points are vertices of this linestring
        if self.points.contains(other) {
            return true;
        }

        for xyz in &other.xyzs {
            for index in 0..self.points.len() - 1 {
                if !xyzs_collinear(&self.points.xyzs[index], &xyz, &self.points.xyzs[index + 1]) {
                    return false;
                }
            }

            if self.closed {
                // if the arcstring is closed, also check the final closing arc
                if !xyzs_collinear(
                    &self.points.xyzs[self.points.len() - 1],
                    &xyz,
                    &self.points.xyzs[0],
                ) {
                    return false;
                }
            }
        }

        true
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        other.intersects(self)
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn split(&self, other: &MultiSphericalPoint) -> MultiArcString {
        split_arcstring_at_points(self, other.xyzs.iter().collect())
    }
}

impl GeometricOperations<ArcString> for ArcString {
    fn distance(&self, other: &ArcString) -> f64 {
        todo!()
    }

    fn contains(&self, other: &ArcString) -> bool {
        other.within(self)
    }

    fn within(&self, other: &ArcString) -> bool {
        self.points
            .xyzs
            .iter()
            .all(|xyz| arcstring_contains_point(other, xyz))
    }

    fn touches(&self, other: &ArcString) -> bool {
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

    fn crosses(&self, other: &ArcString) -> bool {
        if self.within(other) || self.contains(other) {
            return false;
        }

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so I guess the best we can do instead is use brute-force
        for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
            let arc_start = self.points.xyzs[arc_index];
            let arc_end = self.points.xyzs[if arc_index == self.points.len() - 1 {
                0
            } else {
                arc_index + 1
            }];

            for other_arc_index in 0..other.points.len() - if other.closed { 0 } else { 1 } {
                let other_arc_start = other.points.xyzs[other_arc_index];
                let other_arc_end = other.points.xyzs[if other_arc_index == other.points.len() - 1 {
                    0
                } else {
                    other_arc_index + 1
                }];
                if let Some(point) =
                    xyz_two_arc_crossing(&arc_start, &arc_end, &other_arc_start, &other_arc_end)
                {
                    if xyz_eq(&point, &arc_start)
                        || xyz_eq(&point, &arc_end)
                        || xyz_eq(&point, &other_arc_start)
                        || xyz_eq(&point, &other_arc_end)
                    {
                        continue;
                    } else {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.eq(other)
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so I guess the best we can do instead is use brute-force
        for arc_index in 0..self.points.len() - if self.closed { 0 } else { 1 } {
            let arc_start = self.points.xyzs[arc_index];
            let arc_end = self.points.xyzs[if arc_index == self.points.len() - 1 {
                0
            } else {
                arc_index + 1
            }];

            for other_arc_index in 0..other.points.len() - if other.closed { 0 } else { 1 } {
                let other_arc_start = other.points.xyzs[other_arc_index];
                let other_arc_end = other.points.xyzs[if other_arc_index == other.points.len() - 1 {
                    0
                } else {
                    other_arc_index + 1
                }];

                if let Some(point) =
                    xyz_two_arc_crossing(&arc_start, &arc_end, &other_arc_start, &other_arc_end)
                {
                    intersections.push(point);
                }
            }
        }

        MultiSphericalPoint::try_from(intersections).ok()
    }

    fn split(&self, other: &ArcString) -> MultiArcString {
        if let Some(points) = self.intersection(other) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricOperations<MultiArcString> for ArcString {
    fn distance(&self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(&self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        other.touches(self)
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        other.crosses(self)
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn intersection(&self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
        other.intersection(self)
    }

    fn split(&self, other: &MultiArcString) -> MultiArcString {
        if let Some(points) = self.intersection(other) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon> for ArcString {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        other.touches(self)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.crosses(&other.boundary)
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn split(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> MultiArcString {
        if let Some(points) = self.intersection(&other.boundary) {
            split_arcstring_at_points(self, points.xyzs.iter().collect())
        } else {
            MultiArcString::try_from(vec![self.to_owned()]).unwrap()
        }
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon> for ArcString {
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.crosses(self)
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        other.intersection(self)
    }

    fn split(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> MultiArcString {
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

#[pyclass]
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
            .par_iter()
            .map(|points| ArcString::try_from(points.to_owned()).unwrap())
            .collect();
        Self::try_from(arcstrings).unwrap()
    }
}

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
            .into_par_iter()
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
            .par_iter()
            .map(|arcstring| arcstring.midpoints())
            .sum()
    }

    pub fn lengths(&self) -> Vec<f64> {
        self.arcstrings
            .par_iter()
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

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.arcstrings[0].representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.vertices().centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        Some(
            self.arcstrings
                .iter()
                .filter_map(|arcstring| arcstring.boundary())
                .sum(),
        )
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

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative()
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

impl Sum for MultiArcString {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut arcstrings = vec![];
        for multiarcstring in iter {
            arcstrings.extend(multiarcstring.arcstrings);
        }
        MultiArcString::try_from(arcstrings).unwrap()
    }
}

// impl MultiGeometry for &MultiArcString {
//     fn len(&self) -> usize {
//         self.arcstrings.len()
//     }
// }

impl MultiGeometry<ArcString> for MultiArcString {
    fn len(&self) -> usize {
        self.arcstrings.len()
    }
    // }

    // impl ExtendableMultiGeometry<ArcString> for MultiArcString {
    fn extend(&mut self, other: Self) {
        self.arcstrings.extend(other.arcstrings);
    }

    fn push(&mut self, other: ArcString) {
        self.arcstrings.push(other);
    }
}

impl GeometricOperations<SphericalPoint, ArcString> for MultiArcString {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn split(&self, other: &SphericalPoint) -> MultiArcString {
        let mut arcstrings = vec![];
        for arcstring in &self.arcstrings {
            arcstrings.extend(split_arcstring_at_points(arcstring, vec![&other.xyz]).arcstrings);
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricOperations<MultiSphericalPoint, ArcString> for MultiArcString {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.contains(other))
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
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

    fn split(&self, other: &MultiSphericalPoint) -> MultiArcString {
        let mut arcstrings = vec![];
        for arcstring in &self.arcstrings {
            arcstrings.extend(
                split_arcstring_at_points(arcstring, other.xyzs.iter().collect()).arcstrings,
            );
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricOperations<ArcString, ArcString> for MultiArcString {
    fn distance(&self, other: &ArcString) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(&self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.touches(other))
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiSphericalPoint> {
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

    fn split(&self, other: &ArcString) -> MultiArcString {
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

impl GeometricOperations<MultiArcString, ArcString> for MultiArcString {
    fn distance(&self, other: &MultiArcString) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(&self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(&self, other: &MultiArcString) -> Option<MultiSphericalPoint> {
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

    fn split(&self, other: &MultiArcString) -> MultiArcString {
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

impl GeometricOperations<crate::sphericalpolygon::SphericalPolygon, ArcString> for MultiArcString {
    fn distance(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, _: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn split(&self, other: &crate::sphericalpolygon::SphericalPolygon) -> MultiArcString {
        let mut arcstrings = vec![];
        if let Some(points) = self.intersection(&other.boundary) {
            for arcstring in &self.arcstrings {
                arcstrings.extend(
                    split_arcstring_at_points(arcstring, points.xyzs.iter().collect()).arcstrings,
                );
            }
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl GeometricOperations<crate::sphericalpolygon::MultiSphericalPolygon, ArcString>
    for MultiArcString
{
    fn distance(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(&self, _: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        false
    }

    fn within(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn touches(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.intersects(other)
    }

    fn crosses(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.crosses(other))
    }

    fn intersects(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.arcstrings
            .par_iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        &self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<MultiArcString> {
        self.arcstrings
            .par_iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }

    fn split(&self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> MultiArcString {
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

        MultiArcString::try_from(arcstrings).unwrap()
    }
}

impl ToGraph<ArcString> for MultiArcString {
    fn to_graph(&self) -> EdgeGraph<ArcString> {
        let mut graph = EdgeGraph::<ArcString>::default();
        for arcstring in self.arcstrings.iter() {
            graph.push(arcstring);
        }
        graph
    }
}

impl GeometryCollection<ArcString> for MultiArcString {
    fn join(&self) -> Self {
        let mut graph = self.to_graph();
        graph.split_edges();
        graph.prune_overlapping_edges();
        graph.prune_degenerate_edges();

        MultiArcString::try_from(graph.find_disjoint_geometries()).unwrap()
    }

    fn overlap(&self) -> Option<Self> {
        let mut graph = self.to_graph();
        graph.split_edges();
        graph.prune_nonoverlapping_edges();
        graph.prune_degenerate_edges();

        let arcstrings = graph.find_disjoint_geometries();
        if !arcstrings.is_empty() {
            Some(MultiArcString { arcstrings })
        } else {
            None
        }
    }

    fn symmetric_split(&self) -> Self {
        let mut split_graph = self.to_graph();
        split_graph.split_edges();

        let mut overlap_graph = self.to_graph();
        overlap_graph.split_edges();
        overlap_graph.prune_nonoverlapping_edges();
        overlap_graph.prune_degenerate_edges();

        let mut arcstrings = split_graph.find_disjoint_geometries();
        arcstrings.extend(overlap_graph.find_disjoint_geometries());

        MultiArcString { arcstrings }
    }
}
