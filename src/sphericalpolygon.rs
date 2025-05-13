use crate::{
    arcstring::{vector_arc_crossings, ArcString, MultiArcString},
    geometry::{ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry},
    sphericalpoint::{
        angle_between_vectors, shift_rows, vector_arc_length, MultiSphericalPoint, SphericalPoint,
    },
};
use ndarray::{
    array, concatenate,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    s, stack, Array1, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use std::{cmp::Ordering, collections::VecDeque, fmt::Display, iter::Sum};

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
    // angle_between_vectors(c, a, b, false)
    //     + angle_between_vectors(a, b, c, false)
    //     + angle_between_vectors(b, c, a, false)
    //     - std::f64::consts::PI
    let ab = vector_arc_length(a, b, false);
    let bc = vector_arc_length(b, c, false);
    let ca = vector_arc_length(c, a, false);
    let s = (ab + bc + ca) / 2.0;
    4.0 * ((s / 2.0).tan()
        * ((s - ab) / 2.0).tan()
        * ((s - bc) / 2.0).tan()
        * ((s - ca) / 2.0).tan())
    .sqrt()
    .atan()
}

// calculate the interior angles of the polygon comprised of the given points
pub fn spherical_polygon_interior_angles(points: &ArrayView2<f64>, degrees: bool) -> Array1<f64> {
    Zip::from(shift_rows(points, -1).rows())
        .and(points.rows())
        .and(shift_rows(points, 1).rows())
        .par_map_collect(|a, b, c| angle_between_vectors(&a, &b, &c, degrees))
}

/// surface area of a spherical polygon via deconstructing into triangles
/// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
pub fn spherical_polygon_area(points: &ArrayView2<f64>) -> f64 {
    spherical_polygon_interior_angles(points, false).sum()
        - ((points.nrows() - 1) as f64 * std::f64::consts::PI)
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
        if vector_arc_crossings(
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
    // let points = concatenate(
    //     Axis(0),
    //     &[
    //         xyzs.view(),
    //         xyzs.slice(s![1, ..]).to_shape((1, 3)).unwrap().view(),
    //     ],
    // )
    // .unwrap();

    let a = xyzs.slice(s![..xyzs.nrows() - 2, ..]);
    let b = xyzs.slice(s![1..xyzs.nrows() - 1, ..]);
    let c = xyzs.slice(s![2.., ..]);

    (&b.reversed_axes()
        * &crate::sphericalpoint::cross_vectors(&(&a - &b).view(), &(&c - &b).view()).view())
        .sum_axis(Axis(1))
}

fn centroid_from_polygon_boundary(boundary: &ArcString) -> SphericalPoint {
    let xyz = if boundary.points.xyz.nrows() <= 4 {
        // a simple polygon centroid is the mean of its vertices
        boundary.centroid().normalized().xyz
    } else {
        let mut orient = orientation(&boundary.points.xyz.view());
        if orient.sum() < 0.0 {
            orient *= -1.0;
        }
        let midpoints = Zip::from(boundary.points.xyz.slice(s![..-2, ..]).rows())
            .and(
                concatenate(
                    Axis(0),
                    &[
                        boundary.points.xyz.slice(s![2.., ..]),
                        boundary
                            .points
                            .xyz
                            .slice(s![1, ..])
                            .broadcast((1, 3))
                            .unwrap()
                            .view(),
                    ],
                )
                .unwrap()
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
        // build two lists of triangle centroids, segregated by the exterior boundary of the polygon (we don't know which is which yet)
        let mut side_a = vec![];
        let mut side_b = vec![];
        for index in -2..boundary.points.xyz.nrows() as i32 - 3 {
            let triangle_centroid = SphericalPoint {
                xyz: boundary
                    .points
                    .xyz
                    .slice(s![index..index + 3, ..])
                    .sum_axis(Axis(0))
                    / 3.0,
            };

            if (ArcString {
                points: MultiSphericalPoint::try_from(
                    stack(
                        Axis(0),
                        &[triangle_centroid.xyz.view(), boundary_centroid.xyz.view()],
                    )
                    .unwrap(),
                )
                .unwrap(),
                closed: false,
            })
            .crosses(boundary)
            {
                side_a.push(triangle_centroid);
            } else {
                side_b.push(triangle_centroid);
            }
        }

        // assume there are always more interior triangles than exterior, due to the nature of the boundary being closed
        if side_a.len() > side_b.len() {
            Ok(side_a[0].to_owned())
        } else if side_a.len() > side_b.len() {
            Ok(side_b[0].to_owned())
        } else {
            Err(String::from("polygon boundary is a perfect great circle; cannot infer an interior point from two hemispheres!"))
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
    pub fn from_cone(center: &SphericalPoint, radius: &f64, degrees: bool, steps: usize) -> Self {
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
        let xyz = center.vector_rotate_around(&perpendicular, radius, false);

        // Then rotate the pen around the center point all 360 degrees
        let mut spokes = Array1::<f64>::linspace(std::f64::consts::PI * 2.0, 0.0, steps);

        // Ensure that the first and last elements are exactly the same.
        // 2π should equal 0, but with rounding error that isn't always the case.
        spokes[0] = 0.0;

        // reverse the direction
        spokes.invert_axis(Axis(0));

        let vertices = Zip::from(&spokes)
            .par_map_collect(|spoke| xyz.vector_rotate_around(center, spoke, false).xyz)
            .to_vec();

        Self::new(
            ArcString::from(MultiSphericalPoint::try_from(&vertices).unwrap()),
            Some(center.to_owned()),
        )
        .unwrap()
    }

    fn interior_angles(&self, degrees: bool) -> Array1<f64> {
        spherical_polygon_interior_angles(&self.boundary.points.xyz.view(), degrees)
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

    /// we can calculate the surface area of a spherical polygon by summing its interior angles on the sphere
    /// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
    fn area(&self) -> f64 {
        self.interior_angles(false).sum()
            - ((self.boundary.points.len() - 1) as f64 * std::f64::consts::PI)
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

impl GeometricOperations<&SphericalPoint> for &SphericalPolygon {
    fn distance(self, other: &SphericalPoint, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        point_in_polygon_boundary(
            &other.xyz.view(),
            &self.interior_point.xyz.view(),
            &self.boundary.points.xyz.view(),
        )
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.contains(other) || self.within(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.boundary.contains(other)
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &SphericalPolygon {
    fn distance(self, other: &MultiSphericalPoint, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
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

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
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

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
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

        if !intersections.is_empty() {
            Some(
                MultiSphericalPoint::try_from(stack(Axis(0), intersections.as_slice()).unwrap())
                    .unwrap(),
            )
        } else {
            None
        }
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.boundary.intersects(other)
    }
}

impl GeometricOperations<&ArcString> for &SphericalPolygon {
    fn distance(self, other: &ArcString, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &ArcString) -> bool {
        self.contains(&other.points)
    }

    fn within(self, _: &ArcString) -> bool {
        false
    }

    fn crosses(self, other: &ArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn touches(self, other: &ArcString) -> bool {
        self.boundary.touches(other)
    }
}

impl GeometricOperations<&MultiArcString> for &SphericalPolygon {
    fn distance(self, other: &MultiArcString, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, _: &MultiArcString) -> bool {
        false
    }

    fn crosses(self, other: &MultiArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn touches(self, other: &MultiArcString) -> bool {
        self.boundary.touches(other)
    }
}

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &SphericalPolygon, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.contains(&other.vertices())
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &SphericalPolygon) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other) || self.within(other)
    }

    fn intersection(self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn touches(self, other: &SphericalPolygon) -> bool {
        self.boundary.touches(&other.boundary)
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.crosses(other)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self) || self.within(other)
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        other.intersection(self)
    }

    fn touches(self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.touches(other)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiSphericalPolygon {
    pub polygons: VecDeque<SphericalPolygon>,
}

impl From<Vec<SphericalPolygon>> for MultiSphericalPolygon {
    fn from(polygons: Vec<SphericalPolygon>) -> Self {
        Self {
            polygons: polygons.into(),
        }
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
        Some(MultiArcString::from(arcstrings))
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

impl MultiGeometry for &MultiSphericalPolygon {
    fn len(&self) -> usize {
        self.polygons.len()
    }
}

impl MultiGeometry for MultiSphericalPolygon {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl Sum for MultiSphericalPolygon {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut polygons = vec![];
        for multipolygon in iter {
            polygons.extend(multipolygon.polygons);
        }
        Self::from(polygons)
    }
}

impl ExtendMultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: SphericalPolygon) {
        self.polygons.push_back(value);
    }
}

impl GeometricOperations<&SphericalPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &SphericalPoint, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPoint, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        other.within(self)
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        self.touches(other)
            || self.crosses(other)
            || self
                .polygons
                .par_iter()
                .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

impl GeometricOperations<&ArcString> for &MultiSphericalPolygon {
    fn distance(self, other: &ArcString, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &ArcString) -> bool {
        false
    }

    fn crosses(self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn touches(self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

impl GeometricOperations<&MultiArcString> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiArcString, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other
            .arcstrings
            .par_iter()
            .all(|arcstring| arcstring.within(self))
    }

    fn within(self, _: &MultiArcString) -> bool {
        false
    }

    fn crosses(self, other: &MultiArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn touches(self, other: &MultiArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &SphericalPolygon, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .all(|polygon| polygon.within(other))
    }

    fn crosses(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn touches(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon, degrees: bool) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other, degrees)
        }
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .all(|polygon| polygon.within(other))
    }

    fn crosses(self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.crosses(other))
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn touches(self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.touches(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_cone() {
        let lonlat = array![0., 21.];
        let polygon = SphericalPolygon::from_cone(
            &crate::sphericalpoint::SphericalPoint::try_from_lonlat(&lonlat.view(), true).unwrap(),
            &8.0,
            true,
            64,
        );
        assert!(polygon.area() > 0.0);
    }
}
