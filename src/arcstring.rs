use crate::{
    geometry::{GeometricOperations, Geometry, MultiGeometry},
    sphericalpoint::{
        cross_vector, min_1darray, normalize_vector, point_within_kdtree, vector_arc_radians,
        MultiSphericalPoint, SphericalPoint,
    },
};
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt::Display;
use std::iter::Sum;

pub fn interpolate_points_along_vector_arc(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    n: usize,
) -> Result<Array2<f64>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::linspace(0.0, 1.0, n);
    let t = t.to_shape((n, 1)).unwrap();
    let omega = vector_arc_radians(a, b, false);

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
            Ok(concatenate![
                Axis(0),
                (a * ((Zip::from(&t).par_map_collect(|t| 1.0 - t) * omega).sin() / omega.sin())
                    + b * &((t * omega).sin() / omega.sin()).view()),
                b.broadcast((1, 2)).unwrap(),
            ])
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
pub fn vector_arc_crossing(
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

    let xyzs = if arcstring.closed {
        &concatenate![
            Axis(0),
            xyzs.view(),
            xyzs.slice(s![0, ..]).broadcast((1, 3)).unwrap()
        ]
    } else {
        xyzs
    };

    // iterate over individual arcs and check if the given point is collinear with their endpoints
    for arc_index in 0..xyzs.nrows() - 1 {
        let arc_0 = xyzs.slice(s![arc_index, ..]);
        let arc_1 = xyzs.slice(s![arc_index + 1, ..]);

        if crate::sphericalpoint::vectors_collinear(&arc_0.view(), xyz, &arc_1.view()) {
            return true;
        }
    }

    false
}

pub fn split_arc_at_points(arc: &ArrayView2<f64>, points: &ArrayView2<f64>) -> Vec<Array2<f64>> {
    let mut arcs = vec![arc.to_owned()];
    for point in points.rows() {
        for arc_index in 0..arcs.len() {
            let arc_0 = arcs[arc_index].slice(s![0, ..]);
            let arc_1 = arcs[arc_index].slice(s![1, ..]);
            if crate::sphericalpoint::vectors_collinear(&arc_0, &point, &arc_1) {
                // replace arc with the arc split in two at the collinear point
                arcs[arc_index] = stack![Axis(0), arcs[arc_index].slice(s![0, ..]), point.view()];
                arcs.insert(
                    arc_index + 1,
                    stack![Axis(0), point.view(), arcs[arc_index].slice(s![1, ..])],
                );
            }
        }
    }
    arcs
}

pub fn split_arcstring_at_points(
    arcstring: &ArcString,
    points: &ArrayView2<f64>,
) -> MultiArcString {
    let mut arcstrings = vec![arcstring.to_owned()];

    for point in points.rows() {
        for arcstring_index in 0..arcstrings.len() {
            let arcstring = arcstrings[arcstring_index].to_owned();
            for arc_index in 0..arcstring.points.xyz.nrows() - if arcstring.closed { 0 } else { 1 }
            {
                let arc_0 = points.slice(s![arc_index, ..]);
                let arc_1 = points.slice(s![
                    if arc_index < arcstring.points.xyz.nrows() {
                        arc_index + 1
                    } else {
                        0
                    },
                    ..
                ]);

                if crate::sphericalpoint::vectors_collinear(&arc_0, &point, &arc_1) {
                    // replace arc with the arc split in two at the collinear point
                    arcstrings[arcstring_index] = ArcString::try_from(
                        MultiSphericalPoint::try_from(concatenate![
                            Axis(0),
                            arcstring.points.xyz.slice(s![..arc_index + 1, ..]),
                            point.broadcast((1, 3)).unwrap()
                        ])
                        .unwrap(),
                    )
                    .unwrap();
                    arcstrings.insert(
                        arcstring_index + 1,
                        ArcString::try_from(
                            MultiSphericalPoint::try_from(concatenate![
                                Axis(0),
                                point.broadcast((1, 3)).unwrap(),
                                arcstring.points.xyz.slice(s![arc_index + 1.., ..]),
                            ])
                            .unwrap(),
                        )
                        .unwrap(),
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
fn arc_radians_to_point(a: &ArrayView1<f64>, b: &ArrayView1<f64>, xyz: &ArrayView1<f64>) -> f64 {
    let g = crate::sphericalpoint::cross_vector(a, b);
    let f = crate::sphericalpoint::cross_vector(xyz, &g.view());
    let t = crate::sphericalpoint::cross_vector(&g.view(), &f.view());
    vector_arc_radians(&t.view(), xyz, false)
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
        if points.xyz.nrows() < 2 {
            Err(format!(
                "cannot build an arcstring with less than 2 points (received {})",
                points.xyz.nrows()
            ))
        } else {
            let tolerance = 3e-11;
            let (points, closed) = if (&points.xyz.slice(s![-1, ..]) - &points.xyz.slice(s![0, ..]))
                .abs()
                .sum()
                < tolerance
            {
                (
                    MultiSphericalPoint::try_from(points.xyz.slice(s![..-1, ..]).to_owned())
                        .unwrap(),
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
                ArcString::try_from(
                    MultiSphericalPoint::try_from(stack(Axis(0), &[a, b]).unwrap()).unwrap(),
                )
                .unwrap()
            })
            .to_vec()
    }

    /// whether this arcstring intersects itself
    pub fn crosses_self(&self) -> bool {
        if self.points.len() >= 4 {
            let tolerance = 1e-11;

            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so instead we use brute-force and skip visited arcs
            let points = if self.closed {
                &concatenate![
                    Axis(0),
                    self.points.xyz,
                    self.points.xyz.slice(s![0, ..]).broadcast((1, 3)).unwrap()
                ]
            } else {
                &self.points.xyz
            };
            for arc_a_index in 0..points.nrows() - 1 {
                let a_0 =
                    crate::sphericalpoint::normalize_vector(&points.slice(s![arc_a_index, ..]));
                let a_1 =
                    crate::sphericalpoint::normalize_vector(&points.slice(s![arc_a_index + 1, ..]));

                for arc_b_index in arc_a_index + 2..points.nrows() - 1 {
                    let b_0 =
                        crate::sphericalpoint::normalize_vector(&points.slice(s![arc_b_index, ..]));
                    let b_1 = crate::sphericalpoint::normalize_vector(
                        &points.slice(s![arc_b_index + 1, ..]),
                    );
                    if let Some(point) =
                        vector_arc_crossing(&a_0.view(), &a_1.view(), &b_0.view(), &b_1.view())
                    {
                        let point = crate::sphericalpoint::normalize_vector(&point.view());
                        if (&point - &a_0).abs().sum() < tolerance
                            || (&point - &a_1).abs().sum() < tolerance
                            || (&point - &b_0).abs().sum() < tolerance
                            || (&point - &b_1).abs().sum() < tolerance
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
            let tolerance = 1e-11;

            let mut crossings = vec![];

            // we can't use the Bentley-Ottmann sweep-line algorithm here :/
            // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
            // so instead we use brute-force and skip visited arcs
            let points = if self.closed {
                &concatenate![
                    Axis(0),
                    self.points.xyz,
                    self.points.xyz.slice(s![0, ..]).broadcast((1, 3)).unwrap()
                ]
            } else {
                &self.points.xyz
            };
            for arc_a_index in 0..points.nrows() - 1 {
                let a_0 = points.slice(s![arc_a_index, ..]);
                let a_1 = points.slice(s![arc_a_index + 1, ..]);

                for arc_b_index in arc_a_index + 2..points.nrows() - 1 {
                    let b_0 = points.slice(s![arc_b_index, ..]);
                    let b_1 = points.slice(s![arc_b_index + 1, ..]);

                    if let Some(point) = vector_arc_crossing(&a_0, &a_1, &b_0, &b_1) {
                        let point = crate::sphericalpoint::normalize_vector(&point.view());
                        if (&point - &a_0).abs().sum() < tolerance
                            || (&point - &a_1).abs().sum() < tolerance
                            || (&point - &b_0).abs().sum() < tolerance
                            || (&point - &b_1).abs().sum() < tolerance
                        {
                            continue;
                        } else {
                            crossings.push(point);
                        }
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
        let points = if self.closed {
            &concatenate![
                Axis(0),
                self.points.xyz,
                self.points.xyz.slice(s![0, ..]).broadcast((1, 3)).unwrap()
            ]
        } else {
            &self.points.xyz
        };
        Zip::from(points.slice(s![..-1, ..]).rows())
            .and(points.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| vector_arc_radians(&a, &b, false))
            .to_degrees()
    }

    /// whether this arcstring shares endpoints with another, ignoring closed arcstrings
    pub fn adjoins(&self, other: &ArcString) -> bool {
        if !self.closed && !other.closed {
            let tolerance = 2e-8;

            let start = self.points.xyz.slice(s![0, ..]);
            let end = self.points.xyz.slice(s![-1, ..]);

            let other_start = other.points.xyz.slice(s![0, ..]);
            let other_end = other.points.xyz.slice(s![-1, ..]);

            (&end - &other_start).abs().sum() < tolerance
                || (&other_end - &start).abs().sum() < tolerance
                || (&end - &other_end).abs().sum() < tolerance
                || (&start - &other_start).abs().sum() < tolerance
        } else {
            false
        }
    }

    /// join this arcstring to another
    pub fn join(&self, other: &ArcString) -> Option<ArcString> {
        if !self.closed && !other.closed {
            let tolerance = 2e-8;

            let start = self.points.xyz.slice(s![0, ..]);
            let end = self.points.xyz.slice(s![-1, ..]);

            let other_start = other.points.xyz.slice(s![0, ..]);
            let other_end = other.points.xyz.slice(s![-1, ..]);

            if (&end - &other_start).abs().sum() < tolerance
                || (&start - &other_end).abs().sum() < tolerance
                || (&end - &other_end).abs().sum() < tolerance
                || (&start - &other_start).abs().sum() < tolerance
            {
                ArcString::try_from(
                    MultiSphericalPoint::try_from(
                        // flip arcstrings so that they match up end-to-end
                        if (&end - &other_start).abs().sum() < tolerance {
                            concatenate![
                                Axis(0),
                                self.points.xyz.view(),
                                other.points.xyz.slice(s![1.., ..]),
                            ]
                        } else if (&other_end - &start).abs().sum() < tolerance {
                            let mut xyz = self.points.xyz.to_owned();
                            let mut other_xyz = other.points.xyz.to_owned();
                            xyz.invert_axis(Axis(0));
                            other_xyz.invert_axis(Axis(0));
                            concatenate![Axis(0), xyz.view(), other_xyz.slice(s![1.., ..])]
                        } else if (&end - &other_end).abs().sum() < tolerance {
                            let mut other_xyz = other.points.xyz.to_owned();
                            other_xyz.invert_axis(Axis(0));
                            concatenate![
                                Axis(0),
                                self.points.xyz.view(),
                                other_xyz.slice(s![1.., ..]),
                            ]
                        } else {
                            let mut xyz = self.points.xyz.to_owned();
                            xyz.invert_axis(Axis(0));
                            concatenate![Axis(0), xyz.view(), other.points.xyz.slice(s![1.., ..])]
                        },
                    )
                    .unwrap(),
                )
                .ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl PartialEq for ArcString {
    fn eq(&self, other: &Self) -> bool {
        let tolerance = 2e-8;
        (&self.points.xyz - &other.points.xyz).abs().sum() < tolerance || {
            let mut xyz = self.points.xyz.to_owned();
            xyz.invert_axis(Axis(0));
            (&xyz - &other.points.xyz).abs().sum() < tolerance
        }
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

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.points.centroid()
    }

    fn boundary(&self) -> Option<MultiSphericalPoint> {
        Some(
            MultiSphericalPoint::try_from(stack![
                Axis(0),
                self.points.xyz.slice(s![0, ..]),
                self.points.xyz.slice(s![-1, ..])
            ])
            .unwrap(),
        )
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
        crate::sphericalpoint::min_1darray(
            &Zip::from(self.points.xyz.rows())
                .and(crate::sphericalpoint::shift_rows(&self.points.xyz.view(), 1).rows())
                .par_map_collect(|a, b| arc_radians_to_point(&a, &b, &other.xyz.view()))
                .to_degrees()
                .view(),
        )
        .unwrap_or(f64::NAN)
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        arcstring_contains_point(&self, &other.xyz.view())
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
        split_arcstring_at_points(&self, &other.xyz.to_shape((1, 3)).unwrap().view())
    }
}

impl GeometricOperations<MultiSphericalPoint> for ArcString {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        let mut distances = Array1::<f64>::uninit(self.points.xyz.nrows() * other.xyz.nrows());
        for (index, point) in other.xyz.rows().into_iter().enumerate() {
            Zip::from(self.points.xyz.rows())
                .and(crate::sphericalpoint::shift_rows(&self.points.xyz.view(), 1).rows())
                .par_map_collect(|a, b| arc_radians_to_point(&a, &b, &point))
                .assign_to(distances.slice_mut(s![
                    index * other.xyz.nrows()..(index + 1) * other.xyz.nrows()
                ]));
        }
        min_1darray(&unsafe { distances.assume_init() }.to_degrees().view()).unwrap_or(f64::NAN)
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        // check if points are vertices of this linestring
        if self.points.contains(other) {
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
        split_arcstring_at_points(self, &other.xyz.view())
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
        Zip::from(self.points.xyz.rows()).all(|xyz| arcstring_contains_point(other, &xyz))
    }

    fn touches(&self, other: &ArcString) -> bool {
        (Zip::from(other.points.xyz.rows()).any(|xyz| arcstring_contains_point(self, &xyz))
            || Zip::from(self.points.xyz.rows()).any(|xyz| arcstring_contains_point(other, &xyz)))
            && !self.crosses(other)
    }

    fn crosses(&self, other: &ArcString) -> bool {
        if self.within(other) || self.contains(other) {
            return false;
        }

        // we can't use the Bentley-Ottmann sweep-line algorithm here :/
        // because a sphere is an enclosed infinite space so there's no good way to sort by longitude
        // so instead we use brute-force
        let first_vertex_index = if self.closed { -1 } else { 0 };
        for arc_a_index in first_vertex_index..(self.points.xyz.nrows() - 1) as isize {
            let a_0 = self.points.xyz.slice(s![arc_a_index, ..]);
            let a_1 = self.points.xyz.slice(s![arc_a_index + 1, ..]);

            let other_first_vertex_index = if other.closed { -1 } else { 0 };
            for arc_b_index in other_first_vertex_index..(other.points.xyz.nrows() - 1) as isize {
                let b_0 = other.points.xyz.slice(s![arc_b_index, ..]);
                let b_1 = other.points.xyz.slice(s![arc_b_index + 1, ..]);

                if vector_arc_crossing(&a_0, &a_1, &b_0, &b_1).is_some() {
                    return true;
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
        let tolerance = 1e-11;

        let points = if self.closed {
            &concatenate![
                Axis(0),
                self.points.xyz,
                self.points.xyz.slice(s![-1, ..]).broadcast((1, 3)).unwrap()
            ]
        } else {
            &self.points.xyz
        };
        let other_points = if other.closed {
            &concatenate![
                Axis(0),
                other.points.xyz,
                other
                    .points
                    .xyz
                    .slice(s![-1, ..])
                    .broadcast((1, 3))
                    .unwrap()
            ]
        } else {
            &other.points.xyz
        };

        // find crossings first
        for arc_a_index in 0..points.nrows() - 1 {
            let a_0 = points.slice(s![arc_a_index, ..]);
            let a_1 = points.slice(s![arc_a_index + 1, ..]);

            for arc_b_index in 0..other_points.nrows() - 1 {
                let b_0 = other_points.slice(s![arc_b_index, ..]);
                let b_1 = other_points.slice(s![arc_b_index + 1, ..]);

                if (&a_1 - &b_0).abs().sum() < tolerance
                    || (&a_1 - &b_0).abs().sum() < tolerance
                    || crate::sphericalpoint::vectors_collinear(&b_0, &a_1, &b_1)
                {
                    intersections.push(a_1.to_owned());
                } else if let Some(point) = vector_arc_crossing(&a_0, &a_1, &b_0, &b_1) {
                    intersections.push(point);
                }
            }
        }

        if !intersections.is_empty() {
            Some(MultiSphericalPoint::try_from(&intersections).unwrap())
        } else {
            None
        }
    }

    fn split(&self, other: &ArcString) -> MultiArcString {
        if let Some(points) = self.intersection(other) {
            split_arcstring_at_points(self, &points.xyz.view())
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
            split_arcstring_at_points(self, &points.xyz.view())
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
            split_arcstring_at_points(self, &points.xyz.view())
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
                split_arcstring_at_points(self, &points.xyz.view())
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
            arcstrings.extend(
                split_arcstring_at_points(arcstring, &other.xyz.to_shape((1, 3)).unwrap().view())
                    .arcstrings,
            );
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
            arcstrings.extend(split_arcstring_at_points(arcstring, &other.xyz.view()).arcstrings);
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
                arcstrings
                    .extend(split_arcstring_at_points(arcstring, &points.xyz.view()).arcstrings);
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
                arcstrings
                    .extend(split_arcstring_at_points(arcstring, &points.xyz.view()).arcstrings);
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
                arcstrings
                    .extend(split_arcstring_at_points(arcstring, &points.xyz.view()).arcstrings);
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
                        split_arcstring_at_points(arcstring, &points.xyz.view()).arcstrings,
                    );
                }
            }
        }

        MultiArcString::try_from(arcstrings).unwrap()
    }
}
