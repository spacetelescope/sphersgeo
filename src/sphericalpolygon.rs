use crate::{
    arcstring::{vector_arc_crossing, ArcString, MultiArcString},
    edgegraph::{EdgeGraph, GeometryGraph, ToGraph},
    geometry::{GeometricOperations, Geometry, GeometryCollection, MultiGeometry},
    sphericalpoint::{
        angle_between_vectors_radians, vector_arc_radians, MultiSphericalPoint, SphericalPoint,
    },
};
use ndarray::{
    array, concatenate,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    s, stack, Array1, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use std::{cmp::Ordering, fmt::Display, iter::Sum};

/// surface area of a triangle on the sphere via Girard's theorum
///
///     θ_1 + θ_2 + θ_3 − π
///
/// References
/// ----------
/// - Klain, D. A. (2019). A probabilistic proof of the spherical excess formula (No. arXiv:1909.04505). arXiv. https://doi.org/10.48550/arXiv.1909.04505
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
///   `pdf <https://www.google.com/books/edition/Graphics_Gems_IV/CCqzMm_-WucC?hl=en&gbpv=1&dq=Graphics%20Gems%20IV.%20p132&pg=PA133&printsec=frontcover>`_
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    // let area_radians_squared = angle_between_vectors_radians(c, a, b)
    //     + angle_between_vectors_radians(a, b, c)
    //     + angle_between_vectors_radians(b, c, a)
    //     - std::f64::consts::PI;
    let ab = vector_arc_radians(a, b, false);
    let bc = vector_arc_radians(b, c, false);
    let ca = vector_arc_radians(c, a, false);
    let s = (ab + bc + ca) / 2.0;
    let area_radians_squared = 4.0
        * ((s / 2.0).tan()
            * ((s - ab) / 2.0).tan()
            * ((s - bc) / 2.0).tan()
            * ((s - ca) / 2.0).tan())
        .sqrt()
        .atan();

    area_radians_squared.sqrt().to_degrees().powi(2)
}

// use the classical even-crossings ray algorithm for point-in-polygon
pub fn point_in_polygon_boundary(
    point: &ArrayView1<f64>,
    polygon_interior_xyz: &ArrayView1<f64>,
    polygon_boundary_xyzs: &ArrayView2<f64>,
) -> bool {
    // record the number of times the ray intersects the exterior boundary arcstring
    let mut crossings = 0;
    for index in -1..polygon_boundary_xyzs.nrows() as i32 - 2 {
        if vector_arc_crossing(
            point,
            polygon_interior_xyz,
            &polygon_boundary_xyzs.slice(s![index, ..]),
            &polygon_boundary_xyzs.slice(s![index + 1, ..]),
        )
        .is_some()
        {
            crossings += 1;
        }
    }

    // if the number of crossings is even, the point is within the polygon's exterior
    crossings % 2 == 0
}

/// The normal vector to the two arcs containing a vertex points; outward
/// from the sphere if the angle is clockwise, and inward if the angle is
/// counter-clockwise. The sign of the inner product of the normal vector
/// with the vertex tells you this. The polygon is ordered clockwise if
/// the vertices are predominantly clockwise and counter-clockwise if
/// the reverse.
fn orientation(xyzs: &ArrayView2<f64>) -> Array1<f64> {
    let points = concatenate![
        Axis(0),
        xyzs.view(),
        xyzs.slice(s![1, ..]).broadcast((1, 3)).unwrap(),
    ];

    let a = xyzs.slice(s![..xyzs.nrows() - 2, ..]);
    let b = xyzs.slice(s![1..xyzs.nrows() - 1, ..]);
    let c = xyzs.slice(s![2.., ..]);

    (&b.reversed_axes()
        * &crate::sphericalpoint::cross_vectors(&(&a - &b).view(), &(&c - &b).view()).view())
        .sum_axis(Axis(1))
}

fn centroid_from_polygon_boundary(boundary: &ArcString) -> SphericalPoint {
    let xyz = if boundary.points.xyz.nrows() <= 4 {
        // the centroid of a convex polygon is the mean of its vertices
        boundary.centroid().normalized().xyz
    } else {
        let mut orient = orientation(&boundary.points.xyz.view());
        if orient.sum() < 0.0 {
            orient *= -1.0;
        }

        // the centroid of a concave polygon is the mean of its interior area
        let midpoints = Zip::from(boundary.points.xyz.slice(s![..-2, ..]).rows())
            .and(
                concatenate![
                    Axis(0),
                    boundary.points.xyz.slice(s![3.., ..]),
                    boundary
                        .points
                        .xyz
                        .slice(s![1, ..])
                        .broadcast((1, 3))
                        .unwrap()
                ]
                .rows(),
            )
            .par_map_collect(|a_xyz, c_xyz| (&a_xyz + &c_xyz) / 2.0);
        orient
            .iter()
            .zip(midpoints)
            .max_by(|x, y| x.0.partial_cmp(y.0).unwrap())
            .unwrap()
            .1
    };
    SphericalPoint { xyz }
}

/// choose an interior point from the smaller area of the two regions created by the given boundary
fn interior_point_from_polygon_boundary(boundary: &ArcString) -> Result<SphericalPoint, String> {
    // remember: the centroid of the boundary arcstring COULD be outside the polygon!
    let boundary_centroid = boundary.centroid();

    if boundary.points.xyz.nrows() <= 4 {
        Ok(boundary_centroid)
    } else {
        let points = concatenate![
            Axis(0),
            boundary.points.xyz,
            boundary.points.xyz.slice(s![..3, ..])
        ];

        // build two lists of triangle centroids, segregated by the exterior boundary of the polygon (we don't know which is which yet)
        let mut side_a = vec![];
        let mut side_b = vec![];
        for index in 0..points.nrows() - 2 {
            let triangle_centroid = SphericalPoint {
                xyz: points.slice(s![index..index + 3, ..]).sum_axis(Axis(0)) / 3.0,
            };

            let ray = ArcString::try_from(MultiSphericalPoint::from(&vec![
                triangle_centroid.to_owned(),
                boundary_centroid.to_owned(),
            ]))
            .unwrap();

            if ray.crosses(boundary) {
                side_a.push(triangle_centroid);
            } else {
                side_b.push(triangle_centroid);
            }
        }

        // assume there are always more interior triangles than exterior, due to the nature of the boundary being closed
        match side_a.len().cmp(&side_b.len()) {
            Ordering::Less => Ok(side_b[0].to_owned()),
            Ordering::Equal => Err(String::from(
                "cannot infer an interior point from two equal hemispheres!",
            )),
            Ordering::Greater => Ok(side_a[0].to_owned()),
        }
    }
}

/// polygon on the sphere, comprising:
/// 1. a non-intersecting collection of connected arcs (arcstring) that connects back to its first point (closed)
/// 2. an interior point to specify which region of the sphere the polygon represents; this is required for non-Euclidian closed geometry
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalPolygon {
    pub boundary: ArcString,
    pub interior_point: SphericalPoint,
}

impl Display for SphericalPolygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalPolygon({:?})", self.boundary)
    }
}

impl SphericalPolygon {
    /// interior point is required because a sphere is a finite space
    pub fn new(
        boundary: ArcString,
        interior_point: Option<SphericalPoint>,
    ) -> Result<Self, String> {
        if let Some(crossings_with_self) = boundary.crossings_with_self() {
            Err(format!(
                "exterior boundary crosses itself {} times",
                crossings_with_self.xyz.nrows()
            ))
        } else {
            let interior_point = if let Some(interior_point) = interior_point {
                interior_point
            } else {
                interior_point_from_polygon_boundary(&boundary)?
            };

            Ok(Self {
                boundary: if boundary.closed {
                    boundary
                } else {
                    let mut boundary = boundary.to_owned();
                    boundary.closed = true;
                    boundary
                },
                interior_point,
            })
        }
    }

    /// Create a new `SphericalPolygon` from a cone (otherwise known
    /// as a "small circle") defined using (*lon*, *lat*, *radius*).
    ///
    /// The cone is not represented as an ideal circle on the sphere,
    /// but as a series of great circle arcs.  The resolution of this
    /// conversion can be controlled using the *steps* parameter.
    pub fn from_cone(center: &SphericalPoint, radius: &f64, steps: usize) -> Self {
        // Get an arbitrary perpendicular vector.  This be be obtained
        // by crossing the center point with any unit vector that is not itself.
        let mindex = center
            .xyz
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
            .0;
        let perpendicular = center.vector_cross(&SphericalPoint::normalize(
            &if mindex == 0 {
                array![1., 0., 0.]
            } else if mindex == 1 {
                array![0., 1., 0.]
            } else {
                array![0., 0., 1.]
            }
            .view(),
        ));

        // Rotate by radius around the perpendicular vector to get the "pen"
        let xyz = center.vector_rotate_around(&perpendicular, radius);

        // Then rotate the pen around the center point all 360 degrees
        let mut spokes =
            Array1::<f64>::linspace(std::f64::consts::PI * 2.0, 0.0, steps).to_degrees();

        // Ensure that the first and last elements are exactly the same.
        // 2π should equal 0, but with rounding error that isn't always the case.
        spokes[0] = 0.0;

        // reverse the direction
        spokes.invert_axis(Axis(0));

        let vertices = Zip::from(&spokes)
            .par_map_collect(|spoke| xyz.vector_rotate_around(center, spoke).xyz)
            .to_vec();

        Self::new(
            ArcString::try_from(MultiSphericalPoint::try_from(&vertices).unwrap()).unwrap(),
            Some(center.to_owned()),
        )
        .unwrap()
    }

    /// find the point on the sphere outside of this polygon farthest from any vertex
    pub fn antipode(&self) -> SphericalPoint {
        // Compute the minimum distance between all polygon points and each antipode to a polygon point
        let points = self.boundary.points.xyz.slice(s![..-1, ..]);
        SphericalPoint {
            xyz: points
                .rows()
                .into_iter()
                .min_by(|a, b| {
                    crate::sphericalpoint::max_1darray(
                        &(&(a * -1.0).broadcast((points.nrows(), 3)).unwrap() * &points)
                            .sum_axis(Axis(1))
                            .view(),
                    )
                    .partial_cmp(&crate::sphericalpoint::max_1darray(
                        &(&(b * -1.0).broadcast((points.nrows(), 3)).unwrap() * &points)
                            .sum_axis(Axis(1))
                            .view(),
                    ))
                    .unwrap()
                })
                .unwrap()
                .to_owned(),
        }
    }

    /// invert this polygon on the sphere
    pub fn inverse(&self) -> SphericalPolygon {
        Self {
            boundary: self.boundary.to_owned(),
            interior_point: self.antipode(),
        }
    }

    /// whether the points in this polygon are in clockwise order
    pub fn is_clockwise(&self) -> bool {
        orientation(&self.boundary.points.xyz.view()).sum() > 0.0
    }
}

impl Geometry for &SphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.boundary.vertices()
    }

    /// surface area of a spherical polygon via deconstructing into triangles
    /// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
    fn area(&self) -> f64 {
        let centroid = self.centroid();

        let points = concatenate![
            Axis(0),
            self.boundary.points.xyz,
            self.boundary.points.xyz.slice(s![..3, ..])
        ];

        // calculate the interior angles of the polygon comprised of the given points
        // build two lists of triangle centroids, segregated by the exterior boundary of the polygon (we don't know which is which yet)
        let mut angles_radians = vec![];
        for index in 0..points.nrows() - 3 {
            let triangle = points.slice(s![index..index + 3, ..]);

            let triangle_centroid = SphericalPoint {
                xyz: triangle.sum_axis(Axis(0)) / 3.0,
            };

            let ray = ArcString::try_from(MultiSphericalPoint::from(&vec![
                triangle_centroid.to_owned(),
                centroid.to_owned(),
            ]))
            .unwrap();

            let angle_radians = angle_between_vectors_radians(
                &triangle.slice(s![0, ..]),
                &triangle.slice(s![1, ..]),
                &triangle.slice(s![2, ..]),
            );

            angles_radians.push(if ray.crosses(&self.boundary) {
                // invert angle if it's on the exterior of the polygon
                2.0 * std::f64::consts::PI - angle_radians
            } else {
                angle_radians
            });
        }

        let angles_radians = Array1::<f64>::from(angles_radians);
        let area_radians_squared =
            angles_radians.sum() - ((angles_radians.len() - 2) as f64 * std::f64::consts::PI);

        // convert to degrees squared
        area_radians_squared.abs().sqrt().to_degrees().powi(2)
    }

    fn length(&self) -> f64 {
        self.boundary().map_or(0.0, |boundary| boundary.length())
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.interior_point.to_owned()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        (*self).centroid()
    }

    fn boundary(&self) -> Option<ArcString> {
        Some(self.boundary.to_owned())
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.boundary.convex_hull()
    }
}

impl Geometry for SphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.boundary.vertices()
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
        centroid_from_polygon_boundary(&self.boundary)
    }

    fn boundary(&self) -> Option<ArcString> {
        (&self).boundary()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<SphericalPoint> for SphericalPolygon {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        point_in_polygon_boundary(
            &other.xyz.view(),
            &self.interior_point.xyz.view(),
            &self.boundary.points.xyz.view(),
        )
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.boundary.contains(other)
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.contains(other) || self.within(other)
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn split(&self, other: &SphericalPoint) -> MultiSphericalPolygon {
        MultiSphericalPolygon {
            polygons: vec![self.to_owned()],
        }
    }
}

impl GeometricOperations<MultiSphericalPoint> for SphericalPolygon {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        for xyz in other.xyz.rows() {
            if !point_in_polygon_boundary(
                &xyz,
                &self.interior_point.xyz.view(),
                &self.boundary.points.xyz.view(),
            ) {
                // if the point is NOT within the polygon exterior
                return false;
            }
        }

        // if none of the points returned false
        true
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.boundary.intersects(other)
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        for xyz in other.xyz.rows() {
            if point_in_polygon_boundary(
                &xyz,
                &self.interior_point.xyz.view(),
                &self.boundary.points.xyz.view(),
            ) {
                return true;
            }
        }

        // if no points returned true
        false
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        for xyz in other.xyz.rows() {
            if point_in_polygon_boundary(
                &xyz,
                &self.interior_point.xyz.view(),
                &self.boundary.points.xyz.view(),
            ) {
                intersections.push(xyz);
            }
        }

        MultiSphericalPoint::try_from(stack(Axis(0), intersections.as_slice()).unwrap()).ok()
    }

    fn split(&self, other: &MultiSphericalPoint) -> MultiSphericalPolygon {
        MultiSphericalPolygon {
            polygons: vec![self.to_owned()],
        }
    }
}

impl GeometricOperations<ArcString> for SphericalPolygon {
    fn distance(&self, other: &ArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &ArcString) -> bool {
        self.contains(
            &MultiSphericalPoint::try_from(
                other
                    .points
                    .xyz
                    .slice(s![1..other.points.xyz.nrows() - 1, ..])
                    .to_owned(),
            )
            .unwrap(),
        )
    }

    fn within(&self, _: &ArcString) -> bool {
        false
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.boundary.touches(other)
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiArcString> {
        let mut arcstrings = vec![];

        if other.within(self) {
            arcstrings.push(other.to_owned());
        } else if other.crosses(self) {
            // split arcstring by the polygon boundary
            for arcstring in other.split(self).arcstrings {
                // only include arcstrings inside the polygon
                if arcstring.within(self) {
                    arcstrings.push(arcstring);
                }
            }
        }

        MultiArcString::try_from(arcstrings).ok()
    }

    fn split(&self, other: &ArcString) -> MultiSphericalPolygon {
        let tolerance = 3e-11;

        // split this polygon into several pieces
        let mut polygons = MultiSphericalPolygon {
            polygons: vec![self.to_owned()],
        };
        if let Some(arcstrings) = self.intersection(other) {
            for arcstring in arcstrings.arcstrings {
                // each of these arcstrings splits each polygon in two
                let mut polygon_removal_indices = vec![];
                let mut new_polygons = vec![];
                for (index, polygon) in polygons.polygons.iter().enumerate() {
                    if let Some(crossing_segments) = polygon.intersection(&arcstring) {
                        for crossing_segment in crossing_segments.arcstrings {
                            // an arcstring only splits the polygon if it touches the boundary twice
                            if crate::arcstring::arcstring_contains_point(
                                &self.boundary,
                                &crossing_segment.points.xyz.slice(s![0, ..]),
                            ) && crate::arcstring::arcstring_contains_point(
                                &self.boundary,
                                &crossing_segment.points.xyz.slice(s![-1, ..]),
                            ) {
                                // this polygon will be split into two
                                polygon_removal_indices.push(index);

                                let mut boundary_segments: Vec<ArcString> =
                                    polygon.boundary.split(&crossing_segment).arcstrings;

                                // stitch the pieces back together; they should form two complete boundaries
                                let mut pieces: Vec<ArcString> =
                                    vec![crossing_segment.to_owned(), crossing_segment.to_owned()];
                                let mut latest_segments: Vec<ArcString> =
                                    vec![crossing_segment.to_owned(), crossing_segment.to_owned()];

                                // each segment should be used!
                                while !boundary_segments.is_empty() {
                                    let mut found = false;
                                    let mut segment_removal_indices = vec![];
                                    for (segment_index, segment) in
                                        boundary_segments.iter().enumerate()
                                    {
                                        for piece_index in 0..pieces.len() {
                                            let piece = &pieces[piece_index];
                                            if segment.adjoins(piece)
                                                && segment.adjoins(&latest_segments[piece_index])
                                            {
                                                // if a segment fits onto an existing piece AND is on the working side,
                                                // attach it to that piece
                                                latest_segments[piece_index] =
                                                    boundary_segments[segment_index].to_owned();
                                                segment_removal_indices.push(segment_index);
                                                pieces[piece_index] = pieces[piece_index]
                                                    .join(&latest_segments[piece_index])
                                                    .unwrap();
                                                found = true;
                                            }
                                        }
                                    }

                                    // remove attached segments
                                    for segment_index in segment_removal_indices {
                                        boundary_segments.swap_remove(segment_index);
                                    }

                                    if !found {
                                        panic!("cannot fit split segments back together");
                                    }
                                }

                                for piece in pieces {
                                    new_polygons.push(SphericalPolygon::new(piece, None).unwrap());
                                }
                            }
                        }
                    }
                }

                for polygon_index in polygon_removal_indices {
                    polygons.polygons.swap_remove(polygon_index);
                }

                polygons.polygons.extend(new_polygons);
            }
        }

        polygons
    }
}

impl GeometricOperations<MultiArcString> for SphericalPolygon {
    fn distance(&self, other: &MultiArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(&self, _: &MultiArcString) -> bool {
        false
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        self.boundary.touches(other)
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }

    fn intersection(&self, other: &MultiArcString) -> Option<MultiArcString> {
        let mut arcstrings = vec![];

        for arcstring in &other.arcstrings {
            if let Some(intersection) = self.intersection(arcstring) {
                arcstrings.extend(intersection.arcstrings);
            }
        }

        MultiArcString::try_from(arcstrings).ok()
    }

    fn split(&self, other: &MultiArcString) -> MultiSphericalPolygon {
        let mut polygons = vec![];

        for arcstring in &other.arcstrings {
            polygons.extend(self.split(arcstring).polygons);
        }

        MultiSphericalPolygon { polygons }
    }
}

impl GeometricOperations<SphericalPolygon> for SphericalPolygon {
    fn distance(&self, other: &SphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &SphericalPolygon) -> bool {
        self.contains(&other.vertices())
    }

    fn within(&self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &SphericalPolygon) -> bool {
        self.boundary.touches(&other.boundary)
    }

    fn crosses(&self, other: &SphericalPolygon) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(&self, other: &SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other) || self.within(other)
    }

    fn intersection(&self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        let mut polygons = vec![];
        if self.intersects(other) {
            for polygon in self.split(other).polygons {
                if polygon.within(other) {
                    polygons.push(polygon);
                }
            }
        }

        MultiSphericalPolygon::try_from(polygons).ok()
    }

    fn split(&self, other: &SphericalPolygon) -> MultiSphericalPolygon {
        let mut polygons = vec![];
        if self.intersects(other) {
            let mut segments: Vec<ArcString> = self.boundary.split(&other.boundary).arcstrings;
            if let Some(other_intersection) = self.intersection(&other.boundary) {
                segments.extend(other_intersection.arcstrings);
            }

            // each of these segments connects to two of the other's segments, and vice versa
            while !segments.is_empty() {
                let mut segment_removal_indices = vec![];
                let mut joined_segments = vec![];
                for (segment_index, segment) in segments.iter().enumerate() {
                    for (other_segment_index, other_segment) in segments.iter().enumerate() {
                        if segment_index != other_segment_index && segment.adjoins(other_segment) {
                            let joined = segment.join(other_segment).unwrap();
                            if joined.closed {
                                polygons.push(SphericalPolygon::new(joined, None).unwrap());
                            } else {
                                joined_segments.push(joined);
                                segment_removal_indices.push(segment_index);
                                segment_removal_indices.push(other_segment_index);
                            }
                        }
                    }
                }

                // if no joins were found, break out of the loop
                if joined_segments.is_empty() {
                    break;
                }

                // remove comprising parts of joined segments
                segment_removal_indices.sort_unstable();
                segment_removal_indices.reverse();
                segment_removal_indices.dedup();
                println!("{segment_removal_indices:?}");
                for segment_index in segment_removal_indices {
                    segments.swap_remove(segment_index);
                }

                // add joined segments
                segments.extend(joined_segments);
            }
        } else {
            polygons.push(self.to_owned());
        }

        MultiSphericalPolygon { polygons }
    }
}

impl GeometricOperations<MultiSphericalPolygon> for SphericalPolygon {
    fn distance(&self, other: &MultiSphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn contains(&self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(&self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn touches(&self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.touches(other)
    }

    fn crosses(&self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(&self, other: &MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self) || self.within(other)
    }

    fn intersection(&self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        other
            .polygons
            .par_iter()
            .map(|other_polygon| self.intersection(other_polygon))
            .sum()
    }

    fn split(&self, other: &MultiSphericalPolygon) -> MultiSphericalPolygon {
        other
            .polygons
            .par_iter()
            .map(|other_polygon| self.split(other_polygon))
            .sum()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiSphericalPolygon {
    pub polygons: Vec<SphericalPolygon>,
}

impl TryFrom<Vec<SphericalPolygon>> for MultiSphericalPolygon {
    type Error = String;

    fn try_from(polygons: Vec<SphericalPolygon>) -> Result<Self, Self::Error> {
        if !polygons.is_empty() {
            Ok(Self { polygons })
        } else {
            Err(String::from("no polygons provided"))
        }
    }
}

impl ToGraph<SphericalPolygon> for MultiSphericalPolygon {
    fn to_graph(&self) -> EdgeGraph<SphericalPolygon> {
        let mut graph = EdgeGraph::<SphericalPolygon>::default();
        for polygon in self.polygons.iter() {
            graph.push(polygon);
        }
        graph
    }
}

impl Display for MultiSphericalPolygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiAngularPolygon({:?})", self.polygons)
    }
}

impl PartialEq<MultiSphericalPolygon> for MultiSphericalPolygon {
    fn eq(&self, other: &MultiSphericalPolygon) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for polygon in &self.polygons {
            if !other.polygons.contains(polygon) {
                return false;
            }
        }

        true
    }
}

impl PartialEq<Vec<SphericalPolygon>> for MultiSphericalPolygon {
    fn eq(&self, other: &Vec<SphericalPolygon>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for polygon in &self.polygons {
            if !other.contains(polygon) {
                return false;
            }
        }

        true
    }
}

impl Geometry for &MultiSphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.polygons
            .par_iter()
            .map(|geometry| geometry.vertices())
            .sum()
    }

    fn area(&self) -> f64 {
        self.polygons.par_iter().map(|polygon| polygon.area()).sum()
    }

    fn length(&self) -> f64 {
        self.boundary().map_or(0.0, |boundary| boundary.length())
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.polygons[0].representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        let centroids: Vec<SphericalPoint> = self
            .polygons
            .par_iter()
            .map(|polygon| polygon.centroid())
            .collect();
        MultiSphericalPoint::from(&centroids).centroid()
    }

    fn boundary(&self) -> Option<MultiArcString> {
        let arcstrings: Vec<ArcString> = self
            .polygons
            .par_iter()
            .filter_map(|polygon| polygon.boundary())
            .collect();
        MultiArcString::try_from(arcstrings).ok()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.vertices().convex_hull()
    }
}

impl Geometry for MultiSphericalPolygon {
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

    fn boundary(&self) -> Option<MultiArcString> {
        (&self).boundary()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }
}

// impl MultiGeometry for &MultiSphericalPolygon {
//     fn len(&self) -> usize {
//         self.polygons.len()
//     }
// }

impl MultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn len(&self) -> usize {
        self.polygons.len()
    }
    // }

    // impl ExtendableMultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: SphericalPolygon) {
        self.polygons.push(value);
    }
}

impl Sum for MultiSphericalPolygon {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut polygons = vec![];
        for multipolygon in iter {
            polygons.extend(multipolygon.polygons);
        }
        Self { polygons }
    }
}

impl GeometricOperations<SphericalPoint, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(&self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn split(&self, other: &SphericalPoint) -> MultiSphericalPolygon {
        self.to_owned()
    }
}

impl GeometricOperations<MultiSphericalPoint, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        other.within(self)
    }

    fn within(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.touches(other)
            || self.crosses(other)
            || self
                .polygons
                .par_iter()
                .any(|polygon| polygon.intersects(other))
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn split(&self, other: &MultiSphericalPoint) -> MultiSphericalPolygon {
        self.to_owned()
    }
}

impl GeometricOperations<ArcString, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &ArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(&self, _: &ArcString) -> bool {
        false
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(&self, other: &ArcString) -> Option<MultiArcString> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn split(&self, other: &ArcString) -> MultiSphericalPolygon {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.split(other))
            .sum()
    }
}

impl GeometricOperations<MultiArcString, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other
            .arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(self))
    }

    fn within(&self, _: &MultiArcString) -> bool {
        false
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }

    fn intersection(&self, other: &MultiArcString) -> Option<MultiArcString> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn split(&self, other: &MultiArcString) -> MultiSphericalPolygon {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.split(other))
            .sum()
    }
}

impl GeometricOperations<SphericalPolygon, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &SphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .all(|polygon| polygon.within(other))
    }

    fn touches(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(&self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn split(&self, other: &SphericalPolygon) -> MultiSphericalPolygon {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.split(other))
            .sum()
    }
}

impl GeometricOperations<MultiSphericalPolygon, SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiSphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn contains(&self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .all(|polygon| polygon.within(other))
    }

    fn touches(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }

    fn crosses(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(&self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn split(&self, other: &MultiSphericalPolygon) -> MultiSphericalPolygon {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.split(other))
            .sum()
    }
}

impl GeometryCollection<SphericalPolygon> for MultiSphericalPolygon {
    fn join(&self) -> Self {
        let mut graph = self.to_graph();
        graph.split_edges();
        graph.prune_overlapping_edges();
        graph.prune_degenerate_edges();

        MultiSphericalPolygon::try_from(graph.find_disjoint_geometries()).unwrap()
    }

    fn overlap(&self) -> Option<Self> {
        let mut graph = self.to_graph();
        graph.split_edges();
        graph.prune_nonoverlapping_edges();
        graph.prune_degenerate_edges();

        MultiSphericalPolygon::try_from(graph.find_disjoint_geometries()).ok()
    }

    fn symmetric_split(&self) -> Self {
        let mut split_graph = self.to_graph();
        split_graph.split_edges();

        let mut overlap_graph = self.to_graph();
        overlap_graph.split_edges();
        overlap_graph.prune_nonoverlapping_edges();
        overlap_graph.prune_degenerate_edges();

        let mut polygons = split_graph.find_disjoint_geometries();
        polygons.extend(overlap_graph.find_disjoint_geometries());

        MultiSphericalPolygon::try_from(polygons).unwrap()
    }
}
