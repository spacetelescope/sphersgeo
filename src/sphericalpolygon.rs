use crate::{
    arcstring::{xyz_two_arc_crossing, ArcString, MultiArcString},
    edgegraph::{EdgeGraph, GeometryGraph, ToGraph},
    geometry::{GeometricOperations, Geometry, GeometryCollection, MultiGeometry},
    sphericalpoint::{
        xyz_add_xyz, xyz_cross, xyz_div_f64, xyz_mul_xyz, xyz_radians_over_sphere_between,
        xyz_sub_xyz, xyz_sum, xyz_two_arc_angle_radians, xyzs_mean, xyzs_sum, MultiSphericalPoint,
        SphericalPoint,
    },
};
use ndarray::{
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    Array1, Axis, Zip,
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
pub fn spherical_triangle_area(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    // let area_radians_squared = angle_between_vectors_radians(c, a, b)
    //     + angle_between_vectors_radians(a, b, c)
    //     + angle_between_vectors_radians(b, c, a)
    //     - std::f64::consts::PI;
    let ab = xyz_radians_over_sphere_between(a, b);
    let bc = xyz_radians_over_sphere_between(b, c);
    let ca = xyz_radians_over_sphere_between(c, a);
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
    point: &[f64; 3],
    polygon_interior_xyz: &[f64; 3],
    polygon_boundary_xyzs: &[[f64; 3]],
) -> bool {
    // record the number of times the ray intersects the exterior boundary arcstring
    let mut crossings = 0;
    for index in 0..polygon_boundary_xyzs.len() - 1 {
        if xyz_two_arc_crossing(
            point,
            polygon_interior_xyz,
            &polygon_boundary_xyzs[index],
            &polygon_boundary_xyzs[index + 1],
        )
        .is_some()
        {
            crossings += 1;
        }
    }

    // if the number of crossings is even, the point is within the polygon's exterior
    crossings % 2 == 0
}

fn polygon_boundary_is_convex(xyzs: &[[f64; 3]]) -> bool {
    // if all orientations are positive, the polygon is convex
    let orient = orientation(xyzs);

    let positive_dominant = orient.iter().sum::<f64>() > 0.0;
    orient.iter().all(|orientation| {
        if positive_dominant {
            orientation > &0.0
        } else {
            orientation < &0.0
        }
    })
}

/// The normal vector to the two arcs containing a vertex points; outward
/// from the sphere if the angle is clockwise, and inward if the angle is
/// counter-clockwise. The sign of the inner product of the normal vector
/// with the vertex tells you this. The polygon is ordered clockwise if
/// the vertices are predominantly clockwise and counter-clockwise if
/// the reverse.
fn orientation(xyzs: &[[f64; 3]]) -> Vec<f64> {
    (0..xyzs.len() - 2)
        .map(|index| {
            let a = xyzs[index];
            let b = xyzs[index + 1];
            let c = xyzs[index + 2];

            xyz_sum(&xyz_mul_xyz(
                &b,
                &xyz_cross(&xyz_sub_xyz(&a, &b), &xyz_sub_xyz(&c, &b)),
            ))
        })
        .collect()
}

fn centroid_from_polygon_boundary(xyzs: &Vec<[f64; 3]>) -> SphericalPoint {
    let mut orient = orientation(xyzs);

    // make positive orientation the dominant
    if orient.iter().sum::<f64>() < 0.0 {
        orient = orient.iter().map(|value| value * -1.0).collect();
    }

    // if all orientations are positive, the polygon is convex
    SphericalPoint::from(if orient.iter().all(|orientation| orientation > &0.0) {
        // the centroid of a convex polygon is the simple mean of its vertices
        xyzs_mean(xyzs)
    } else {
        // get the centroid of all midpoints between the endpoints of each convex triad (acute interior angle)
        xyzs_mean(
            &orient
                .iter()
                .enumerate()
                .filter_map(|(index, orientation)| {
                    if orientation > &0.0 {
                        // midpoint between endpoint of convex triad (acute interior angle)
                        Some(xyz_div_f64(
                            &xyz_add_xyz(
                                &xyzs[index],
                                &xyzs[if index >= xyzs.len() - 2 {
                                    index - xyzs.len()
                                } else {
                                    index
                                } + 2],
                            ),
                            &2.0,
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
    })
}

/// choose an interior point from the smaller area of the two regions created by the given boundary
fn interior_point_from_polygon_boundary(boundary: &ArcString) -> Result<SphericalPoint, String> {
    let boundary_centroid = boundary.centroid();

    if boundary.points.len() <= 4 {
        Ok(boundary_centroid)
    } else {
        // the centroid of the boundary arcstring COULD be outside the polygon if the boundary is not convex

        // build two lists of triangle centroids, segregated by the exterior boundary of the polygon (we don't know which is which yet)
        let mut side_a = vec![];
        let mut side_b = vec![];
        for index in 0..boundary.points.len() - 2 {
            let triangle_centroid = SphericalPoint {
                xyz: xyz_div_f64(
                    &xyzs_sum(&boundary.points.xyzs[index..index + 3].to_vec()),
                    &3.0,
                ),
            };

            let ray = ArcString::try_from(
                MultiSphericalPoint::try_from(vec![
                    triangle_centroid.to_owned(),
                    boundary_centroid.to_owned(),
                ])
                .unwrap(),
            )
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
    pub fn try_new(
        boundary: ArcString,
        interior_point: Option<SphericalPoint>,
    ) -> Result<Self, String> {
        if let Some(crossings_with_self) = boundary.crossings_with_self() {
            Err(format!(
                "exterior boundary crosses itself {} times",
                crossings_with_self.len()
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
        let perpendicular = center.vector_cross(&SphericalPoint::from(if mindex == 0 {
            [1., 0., 0.]
        } else if mindex == 1 {
            [0., 1., 0.]
        } else {
            [0., 0., 1.]
        }));

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

        Self::try_new(
            ArcString::try_from(MultiSphericalPoint::try_from(vertices).unwrap()).unwrap(),
            Some(center.to_owned()),
        )
        .unwrap()
    }

    /// invert this polygon on the sphere
    pub fn inverse(&self) -> SphericalPolygon {
        Self::try_new(self.boundary.to_owned(), Some(&self.centroid() * &-1.0)).unwrap()
    }

    /// whether all interior angles are clockwise
    pub fn is_convex(&self) -> bool {
        polygon_boundary_is_convex(&self.boundary.points.xyzs)
    }

    /// whether the points in this polygon are in clockwise order
    pub fn is_clockwise(&self) -> bool {
        orientation(&self.boundary.points.xyzs).iter().sum::<f64>() > 0.0
    }
}

impl Geometry for SphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.boundary.vertices()
    }

    /// surface area of a spherical polygon via deconstructing into triangles
    /// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
    fn area(&self) -> f64 {
        // calculate the interior angles of the polygon comprised of the given points
        let angles = (0..self.boundary.points.len() - 3)
            .map(|index| {
                let triangle = vec![
                    self.boundary.points.xyzs[index],
                    self.boundary.points.xyzs[if index >= self.boundary.points.xyzs.len() - 1 {
                        index - self.boundary.points.xyzs.len()
                    } else {
                        index
                    } + 1],
                    self.boundary.points.xyzs[if index >= self.boundary.points.xyzs.len() - 2 {
                        index - self.boundary.points.xyzs.len()
                    } else {
                        index
                    } + 2],
                ];

                let triangle_centroid = SphericalPoint {
                    xyz: xyzs_mean(&triangle),
                };

                let ray = ArcString::try_from(
                    MultiSphericalPoint::try_from(vec![
                        triangle_centroid.to_owned(),
                        self.interior_point.to_owned(),
                    ])
                    .unwrap(),
                )
                .unwrap();

                let angle_radians =
                    xyz_two_arc_angle_radians(&triangle[0], &triangle[1], &triangle[2]);

                if ray.crosses(&self.boundary) {
                    // invert angle if it's on the exterior of the polygon
                    2.0 * std::f64::consts::PI - angle_radians
                } else {
                    angle_radians
                }
                .to_degrees()
            })
            .collect::<Vec<f64>>();

        // sum the interior angles and subtract to get the spherical excess (area)
        angles.iter().sum::<f64>() - ((angles.len() - 2) as f64 * std::f64::consts::PI.to_degrees())
    }

    fn length(&self) -> f64 {
        self.boundary().map_or(0.0, |boundary| boundary.length())
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.interior_point.to_owned()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        centroid_from_polygon_boundary(&self.boundary.points.xyzs)
    }

    fn boundary(&self) -> Option<ArcString> {
        Some(self.boundary.to_owned())
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.boundary.convex_hull()
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
            &other.xyz,
            &self.interior_point.xyz,
            &self.boundary.points.xyzs,
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
        for xyz in &other.xyzs {
            if !point_in_polygon_boundary(xyz, &self.interior_point.xyz, &self.boundary.points.xyzs)
            {
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
        for xyz in &other.xyzs {
            if point_in_polygon_boundary(xyz, &self.interior_point.xyz, &self.boundary.points.xyzs)
            {
                return true;
            }
        }

        // if no points returned true
        false
    }

    fn intersection(&self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        MultiSphericalPoint::try_from(
            other
                .xyzs
                .iter()
                .filter_map(|xyz| {
                    if point_in_polygon_boundary(
                        xyz,
                        &self.interior_point.xyz,
                        &self.boundary.points.xyzs,
                    ) {
                        Some(*xyz)
                    } else {
                        None
                    }
                })
                .collect::<Vec<[f64; 3]>>(),
        )
        .ok()
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
        self.touches(other)
            || self.crosses(other)
            || self.contains(
                &MultiSphericalPoint::try_from(
                    // endpoints of an arcstring are not part of the set
                    other.points.xyzs[1..other.points.len() - 1].to_owned(),
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
                                &crossing_segment.points.xyzs[0],
                            ) && crate::arcstring::arcstring_contains_point(
                                &self.boundary,
                                &crossing_segment.points.xyzs[crossing_segment.points.len() - 1],
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
                                    new_polygons
                                        .push(SphericalPolygon::try_new(piece, None).unwrap());
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
                                polygons.push(SphericalPolygon::try_new(joined, None).unwrap());
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

impl Geometry for MultiSphericalPolygon {
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
        MultiSphericalPoint::try_from(
            self.polygons
                .par_iter()
                .map(|polygon| polygon.centroid().xyz)
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
        .centroid()
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
        graph.assign_polygons_to_edges();
        graph.remove_multisourced_edges();
        graph.remove_degenerate_edges();

        MultiSphericalPolygon::try_from(Vec::<SphericalPolygon>::from(graph)).unwrap()
    }

    fn overlap(&self) -> Option<Self> {
        let mut graph = self.to_graph();
        graph.split_edges();
        graph.assign_polygons_to_edges();
        graph.remove_unisourced_edges();
        graph.remove_degenerate_edges();

        MultiSphericalPolygon::try_from(Vec::<SphericalPolygon>::from(graph)).ok()
    }

    fn symmetric_split(&self) -> Self {
        let mut split_graph = self.to_graph();
        split_graph.split_edges();
        split_graph.assign_polygons_to_edges();

        let mut overlap_graph = self.to_graph();
        overlap_graph.split_edges();
        overlap_graph.assign_polygons_to_edges();
        overlap_graph.remove_unisourced_edges();
        overlap_graph.remove_degenerate_edges();

        let mut polygons = Vec::<SphericalPolygon>::from(split_graph);
        polygons.extend(Vec::<SphericalPolygon>::from(overlap_graph));

        MultiSphericalPolygon::try_from(polygons).unwrap()
    }
}
