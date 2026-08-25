use crate::{
    arcstring::{ArcString, MultiArcString, points_are_on_same_side, xyzs_turn_orientations},
    edgegraph::EdgeGraph,
    geometry::{
        GeometricOperations, GeometricRelationships, Geometry, GeometryCollection, MultiGeometry,
        MultiGeometryUnaryOperations,
    },
    sphericalpoint::{MultiSphericalPoint, SphericalPoint, xyz_div_f64, xyzs_sum},
};
use std::{
    cmp::Ordering,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign},
};

#[cfg(feature = "py")]
use pyo3::prelude::*;

/// solid angle of a triangle on the sphere via Oosterom-Strackee formula:
///
///     2 * arctan((a · (b ⨯ c)) / (1 + (a · b) + (b · c) + (c · a)))
///
/// References
/// ----------
/// - A. Van Oosterom and J. Strackee, "The Solid Angle of a Plane Triangle," in IEEE Transactions on Biomedical Engineering, vol. BME-30, no. 2, pp. 125-126, Feb. 1983, doi: 10.1109/TBME.1983.325207.
pub fn solid_angle_of_spherical_triangle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    2.0 * crate::sphericalpoint::xyz_dot(a, &crate::sphericalpoint::xyz_cross(b, c)).atan2(
        1.0 + crate::sphericalpoint::xyz_dot(a, b)
            + crate::sphericalpoint::xyz_dot(b, c)
            + crate::sphericalpoint::xyz_dot(c, a),
    )
}

/// whether this polygon is convex, that is, all possible arcs between points inside the polygon never leave the enclosed space
fn polygon_boundary_is_convex(boundary: &ArcString) -> bool {
    // if all orientations are positive (left turns), the polygon is convex
    boundary.points.xyzs.len() <= 4
        || xyzs_turn_orientations(&boundary.points.xyzs, true)
            .iter()
            .all(|orientation| orientation < &0.0)
}

fn centroid_of_polygon_boundary(boundary: &ArcString) -> SphericalPoint {
    // see here https://www.javaspring.net/blog/how-can-you-find-the-centroid-of-a-concave-irregular-polygon-in-javascript/#step-by-step-calculation-in-javascript
    todo!()
}

/// vertex angles in radians to the left of the given polygon boundary
fn vertex_angles_inside_polygon_boundary(boundary: &ArcString) -> Vec<f64> {
    crate::arcstring::xyzs_turn_angles(&boundary.points.xyzs, true)
        .into_iter()
        .zip(xyzs_turn_orientations(&boundary.points.xyzs, true).iter())
        .map(|(angle, orientation)| {
            if orientation < &0.0 {
                angle
            } else {
                // invert the angle to ensure it's the one on the left
                (2.0 * std::f64::consts::PI) - angle
            }
        })
        .collect::<Vec<f64>>()
}

/// surface area of the polygon in square degrees
///
/// References
/// ----------
/// - https://en.wikipedia.org/wiki/Polygon_triangulation
/// - Toddhunter, I. (1886). Article 99. In Spherical Trigonometry: For the Use of Colleges and Schools (pp. 73–74). print.
fn area_inside_polygon_boundary(boundary: &ArcString) -> f64 {
    let xyzs = &boundary.points.xyzs;
    // if the polygon is convex, we can do fan triangulation from any arbitrary point
    // if the polygon is concave, the area given by Oosterom-Strackee is signed
    // such that exterior triangles cancel out interior triangles
    let a = xyzs[0];
    let solid_angle = (1..xyzs.len() - 1)
        .map(|index| {
            let b = xyzs[index];
            let c = xyzs[index + 1];

            solid_angle_of_spherical_triangle(&a, &b, &c)
        })
        .sum::<f64>();

    // convert from steradians to square degrees
    solid_angle * 3282.8065632
}

/// choose an interior point from the region to the left of the given boundary
///
/// assumes the boundary is closed
fn find_point_inside_polygon_boundary(boundary: &ArcString) -> Result<SphericalPoint, String> {
    let centroid_of_vertices = boundary.centroid();

    let orientations = xyzs_turn_orientations(&boundary.points.xyzs, true);
    if boundary.points.xyzs.len() <= 4 || orientations.iter().all(|orientation| orientation < &0.0)
    {
        // if the polygon has 4 or less points,
        // or is only comprised of left turns,
        // then we can assume the centroid of vertices is inside, ezpz
        Ok(centroid_of_vertices)
    } else if orientations.iter().all(|orientation| orientation > &0.0) {
        // if the polygon is ENTIRELY right turns
        // (comprising an area of more than half the sphere)
        // then we know that the antipode of the vertex centroid is inside, so again ezpz gg
        Ok(centroid_of_vertices.antipode())
    } else if orientations.iter().all(|orientation| orientation == &0.0) {
        // if the polygon is a great circle on the sphere, bisecting it in two,
        // then we can pick any point from the hemisphere on the left
        todo!()
    } else {
        // otherwise this is a concave polygon and we have to do some WORK
        // to find a point guaranteed to be inside

        // iterate over groups of three vertices on the boundary to form triangles,
        // the centroids of which can be either inside or outside the polygon.
        // Keep track which side of the boundary each one is on
        // (IMPORTANT: remember we don't know yet if the centroid of vertices is inside or outside)
        let mut centroids_of_triangles = vec![];
        let mut with_centroid_of_vertices = vec![];
        for index in 0..boundary.points.len() {
            let a = boundary.points.xyzs[if index > 0 {
                index - 1
            } else {
                boundary.points.len() - 1
            }];
            let b = boundary.points.xyzs[index];
            let c = boundary.points.xyzs[if index < boundary.points.len() - 1 {
                index + 1
            } else {
                0
            }];

            let centroid_of_triangle = SphericalPoint {
                xyz: xyz_div_f64(&xyzs_sum(&vec![a, b, c]), &3.0),
            };

            with_centroid_of_vertices.push(points_are_on_same_side(
                &centroid_of_triangle.xyz,
                &centroid_of_vertices.xyz,
                boundary,
            ));
            centroids_of_triangles.push(centroid_of_triangle)
        }

        // If the polygon is majorly left turns (counterclockwise)
        // then we need a triangle centroid from a left turn.
        // The opposite (from a right turn) for majorly right turns (clockwise).
        let is_clockwise = orientations.iter().sum::<f64>() > 0.0;

        // Then we'll simply need to compare the areas of each side.
        // This is where it gets dodgy;
        // I figure we can ROUGHLY tell which side has the greater area
        // by comparing the interior angles of triangles with centroids on either side...
        //
        // TODO validate this by testing with a polygon in the shape of a gulper eel
        let interior_angles = vertex_angles_inside_polygon_boundary(boundary);
        let sum_angles_with_centroid_of_vertices = interior_angles
            .iter()
            .zip(with_centroid_of_vertices.iter())
            .filter_map(|(angle, on_same_side_as_centroid_of_vertices)| {
                if on_same_side_as_centroid_of_vertices == &true {
                    Some(angle)
                } else {
                    None
                }
            })
            .sum::<f64>();
        let sum_angles_without_centroid_of_vertices = interior_angles
            .iter()
            .zip(with_centroid_of_vertices.iter())
            .filter_map(|(angle, on_same_side_as_centroid_of_vertices)| {
                if on_same_side_as_centroid_of_vertices == &false {
                    Some(angle)
                } else {
                    None
                }
            })
            .sum::<f64>();

        // For a counterclockwise polygon,
        // we need to pick a triangle centroid from the side with the LESSER area.
        // The opposite (greater area) for a clockwise polygon.
        //
        // here we're making the assumption that
        // there will always be a triangle whose centroid is inside the polygon
        let assumption_is_wrong_message = String::from(
            "apparently no triangles exist along the boundary with a centroid inside the polygon.",
        );
        let cannot_infer_error_message = if sum_angles_with_centroid_of_vertices
            > sum_angles_without_centroid_of_vertices
        {
            if !is_clockwise {
                return Ok(centroid_of_vertices);
            } else {
                for (centroid_of_triangle, same_side_as_centroid_of_vertices) in
                    centroids_of_triangles
                        .iter()
                        .zip(with_centroid_of_vertices.iter())
                {
                    if same_side_as_centroid_of_vertices == &false {
                        return Ok(centroid_of_triangle.to_owned());
                    }
                }

                assumption_is_wrong_message
            }
        } else if sum_angles_without_centroid_of_vertices > sum_angles_with_centroid_of_vertices {
            if is_clockwise {
                return Ok(centroid_of_vertices);
            } else {
                for (centroid_of_triangle, same_side_as_centroid_of_vertices) in
                    centroids_of_triangles
                        .iter()
                        .zip(with_centroid_of_vertices.iter())
                {
                    if same_side_as_centroid_of_vertices == &false {
                        return Ok(centroid_of_triangle.to_owned());
                    }
                }

                assumption_is_wrong_message
            }
        } else {
            format!(
                "triangles along the boundary of this polygon have equal turn angles ({sum_angles_with_centroid_of_vertices} == {sum_angles_without_centroid_of_vertices})"
            )
        };

        Err(format!(
            "Cannot infer an interior point automatically; {cannot_infer_error_message}. Please provide a point known to be inside this polygon."
        ))
    }
}

/// polygon on the sphere, comprised of a closed boundary arcstring and a point guaranteed to be inside the polygon (inferred if not provided)
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalPolygon {
    pub boundary: ArcString,
    pub interior_point: SphericalPoint,
}

impl TryFrom<ArcString> for SphericalPolygon {
    type Error = String;

    fn try_from(boundary: ArcString) -> Result<Self, Self::Error> {
        Self::try_new(boundary, None)
    }
}

impl Display for SphericalPolygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalPolygon({})", self.boundary)
    }
}

impl SphericalPolygon {
    /// Providing an interior point is recommended because a sphere is a finite space and the boundary of a polygon divides it into two regions.
    /// If not provided, smaller of the two spaces be inferred as "inside" the polygon.
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
                find_point_inside_polygon_boundary(&boundary)?
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

    /// Create a new roughly-circular polygon from a center point and a radius.
    pub fn from_cone(center: &SphericalPoint, radius: &f64, steps: usize) -> Self {
        // Get an arbitrary perpendicular vector by crossing the center point with any unit vector that is not itself.
        let min_index = center
            .xyz
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal))
            .unwrap()
            .0;
        let mut unit_vector = [0., 0., 0.];
        unit_vector[min_index] = 1.;

        let perpendicular = center.vector_cross(&SphericalPoint::from(unit_vector));

        // Rotate by radius around the perpendicular vector to get the "pen"
        let pen = center.vector_rotate_around(&perpendicular, radius);

        // Then rotate the pen around the center point all 360 degrees
        let mut spokes = crate::sphericalpoint::linspace(0.0, std::f64::consts::PI * 2.0, steps);

        // Ensure that the first and last elements are exactly the same.
        // 2π should equal 0, but with rounding error that isn't always the case.
        let num_spokes = spokes.len();
        spokes[num_spokes - 1] = 0.0;

        // iterate over spokes in reverse and calculate the vertices
        let vertices = spokes
            .iter()
            .rev()
            .map(|spoke| pen.vector_rotate_around(center, &spoke.to_degrees()).xyz)
            .collect::<Vec<[f64; 3]>>();

        Self::try_new(
            ArcString::try_from(MultiSphericalPoint::try_from(vertices).unwrap()).unwrap(),
            Some(center.to_owned()),
        )
        .unwrap()
    }

    /// whether this polygon is convex, that is, all possible arcs between points inside the polygon never leave the enclosed space
    pub fn is_convex(&self) -> bool {
        polygon_boundary_is_convex(&self.boundary)
    }

    /// remove redundant vertices that already lie along the boundary
    pub fn simplify(&mut self) {
        self.boundary.simplify();
    }
}

impl Add<Self> for &SphericalPolygon {
    type Output = MultiSphericalPolygon;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::try_from(vec![self.to_owned(), rhs.to_owned()]).unwrap()
    }
}

impl Geometry for SphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.boundary.vertices()
    }

    fn boundary(&self) -> Option<ArcString> {
        Some(self.boundary.to_owned())
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.interior_point.to_owned()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        centroid_of_polygon_boundary(&self.boundary)
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.boundary.convex_hull()
    }

    fn area(&self) -> f64 {
        area_inside_polygon_boundary(&self.boundary)
    }

    fn length(&self) -> f64 {
        (0..self.boundary.points.len())
            .map(|index| {
                // due to the nature of this search, we can skip all previous indices
                (index + 1..self.boundary.points.len())
                    .filter_map(|other_index| {
                        if index != other_index {
                            Some(crate::sphericalpoint::arc_distance_over_sphere(
                                &self.boundary.points.xyzs[index],
                                &self.boundary.points.xyzs[other_index],
                            ))
                        } else {
                            None
                        }
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .to_degrees()
    }

    fn to_wkt(&self, angular: bool) -> String {
        // no holes
        format!(
            "POLYGON ({})",
            self.boundary.to_wkt(angular).replace("LINESTRING ", "")
        )
    }
}

impl GeometricRelationships<SphericalPoint> for SphericalPolygon {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        self.boundary.contains(other)
            || points_are_on_same_side(&other.xyz, &self.interior_point.xyz, &self.boundary)
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        !self.boundary.contains(other)
            && points_are_on_same_side(&other.xyz, &self.interior_point.xyz, &self.boundary)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.boundary.touches(other)
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.contains(other) || self.within(other)
    }
}

impl GeometricRelationships<MultiSphericalPoint> for SphericalPolygon {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        other.xyzs.iter().all(|point| {
            crate::arcstring::point_is_along_arcstring(point, &self.boundary)
                || points_are_on_same_side(point, &self.interior_point.xyz, &self.boundary)
        })
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        other.xyzs.iter().all(|point| {
            !crate::arcstring::point_is_along_arcstring(point, &self.boundary)
                && points_are_on_same_side(point, &self.interior_point.xyz, &self.boundary)
        })
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.boundary.touches(other)
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        return other
            .xyzs
            .iter()
            .any(|point| points_are_on_same_side(point, &self.interior_point.xyz, &self.boundary));
    }
}

impl GeometricRelationships<ArcString> for SphericalPolygon {
    fn distance(&self, other: &ArcString) -> f64 {
        other.distance(self)
    }

    fn covers(&self, other: &ArcString) -> bool {
        !self.boundary.crosses(other) && self.covers(&other.points)
    }
    fn contains(&self, other: &ArcString) -> bool {
        self.covers(other) && !self.boundary.contains(other)
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.boundary.touches(other)
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }
}

impl GeometricRelationships<MultiArcString> for SphericalPolygon {
    fn distance(&self, other: &MultiArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn covers(&self, other: &MultiArcString) -> bool {
        todo!()
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        self.boundary.crosses(other)
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        self.boundary.touches(other)
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }
}

impl GeometricRelationships<Self> for SphericalPolygon {
    fn distance(&self, other: &Self) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn equals(&self, other: &Self) -> bool {
        self.boundary == other.boundary
            && !self
                .interior_point
                .to(&other.interior_point)
                .crosses(&self.boundary)
    }

    fn covers(&self, other: &Self) -> bool {
        self.contains(other) || self == other
    }

    fn contains(&self, other: &Self) -> bool {
        self.contains(&other.vertices())
    }

    fn within(&self, other: &Self) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &Self) -> bool {
        self.boundary.crosses(other)
    }
    fn touches(&self, other: &Self) -> bool {
        self.boundary.touches(other)
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.crosses(other)
    }

    fn intersects(&self, other: &Self) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other) || self.within(other)
    }
}

impl GeometricRelationships<MultiSphericalPolygon> for SphericalPolygon {
    fn distance(&self, other: &MultiSphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary.distance(other)
        }
    }

    fn covers(&self, other: &MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn contains(&self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(&self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(&self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.crosses(other)
    }

    fn touches(&self, other: &MultiSphericalPolygon) -> bool {
        self.boundary.touches(other)
    }
    fn overlaps(&self, other: &MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn intersects(&self, other: &MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self) || self.within(other)
    }
}

impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for SphericalPolygon {
    fn intersection(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpoint::SphericalPoint,
    ) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpoint::MultiSphericalPoint> for SphericalPolygon {
    fn intersection(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::MultiSphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::ArcString> for SphericalPolygon {
    fn intersection(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::ArcString) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString> for SphericalPolygon {
    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(
        &self,
        other: &crate::arcstring::MultiArcString,
    ) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<Self> for SphericalPolygon {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &Self) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &Self) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<MultiSphericalPolygon> for SphericalPolygon {
    fn intersection(&self, other: &MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn union(&self, other: &MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

/// collection of polygons on the sphere
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone, Debug)]
pub struct MultiSphericalPolygon {
    pub polygons: Vec<SphericalPolygon>,
}

impl From<SphericalPolygon> for MultiSphericalPolygon {
    fn from(polygon: SphericalPolygon) -> Self {
        Self::try_from(vec![polygon]).unwrap()
    }
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

impl From<MultiSphericalPolygon> for Vec<SphericalPolygon> {
    fn from(polygons: MultiSphericalPolygon) -> Self {
        polygons.polygons
    }
}

impl Display for MultiSphericalPolygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiSphericalPolygon({:?})", self.polygons)
    }
}

impl PartialEq for MultiSphericalPolygon {
    fn eq(&self, other: &Self) -> bool {
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

impl Add<Self> for &MultiSphericalPolygon {
    type Output = MultiSphericalPolygon;

    fn add(self, rhs: Self) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&Self> for MultiSphericalPolygon {
    fn add_assign(&mut self, other: &Self) {
        self.extend(other.to_owned());
    }
}

impl Add<&SphericalPolygon> for &MultiSphericalPolygon {
    type Output = MultiSphericalPolygon;

    fn add(self, rhs: &SphericalPolygon) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&SphericalPolygon> for MultiSphericalPolygon {
    fn add_assign(&mut self, other: &SphericalPolygon) {
        self.push(other.to_owned());
    }
}

impl Geometry for MultiSphericalPolygon {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.polygons
            .iter()
            .map(|geometry| geometry.vertices())
            .sum()
    }

    fn boundary(&self) -> Option<MultiArcString> {
        let arcstrings: Vec<ArcString> = self
            .polygons
            .iter()
            .filter_map(|polygon| polygon.boundary())
            .collect();
        MultiArcString::try_from(arcstrings).ok()
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.polygons[0].representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        MultiSphericalPoint::try_from(
            self.polygons
                .iter()
                .map(|polygon| polygon.centroid().xyz)
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
        .centroid()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.vertices().convex_hull()
    }

    fn area(&self) -> f64 {
        self.polygons.iter().map(|polygon| polygon.area()).sum()
    }

    fn length(&self) -> f64 {
        self.boundary().map_or(0.0, |boundary| boundary.length())
    }

    fn to_wkt(&self, angular: bool) -> String {
        format!(
            "MULTIPOLYGON ({})",
            self.polygons
                .iter()
                .map(|polygon| polygon.to_wkt(angular).replace("POLYGON ", ""))
                .collect::<Vec<String>>()
                .join("), (")
        )
    }
}

impl MultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn len(&self) -> usize {
        self.polygons.len()
    }

    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: SphericalPolygon) {
        self.polygons.push(value);
    }
}

impl MultiGeometryUnaryOperations<SphericalPolygon> for MultiSphericalPolygon {
    fn unary_union(&self) -> Self {
        let mut graph = EdgeGraph::<SphericalPolygon>::from(self);
        graph.split_edges();
        graph.assign_polygons_to_edges();
        graph.remove_degenerate_edges();

        Self::try_from(Vec::<SphericalPolygon>::from(graph)).unwrap()
    }

    fn unary_intersection(&self) -> Option<Self> {
        let mut graph = EdgeGraph::<SphericalPolygon>::from(self);
        graph.split_edges();
        graph.assign_polygons_to_edges();
        graph.remove_unisourced_edges();
        graph.remove_degenerate_edges();

        Self::try_from(Vec::<SphericalPolygon>::from(graph)).ok()
    }

    fn unary_symmetric_difference(&self) -> Option<Self> {
        let mut graph = EdgeGraph::<SphericalPolygon>::from(self);
        graph.split_edges();
        graph.assign_polygons_to_edges();
        graph.remove_multisourced_edges();
        graph.remove_degenerate_edges();

        Self::try_from(Vec::<SphericalPolygon>::from(graph)).ok()
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

impl GeometricRelationships<SphericalPoint> for MultiSphericalPolygon {
    fn distance(&self, other: &SphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn intersects(&self, other: &SphericalPoint) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }

    fn touches(&self, other: &SphericalPoint) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn contains(&self, other: &SphericalPoint) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn covers(&self, other: &SphericalPoint) -> bool {
        todo!()
    }
}

impl GeometricRelationships<MultiSphericalPoint> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiSphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn covers(&self, other: &MultiSphericalPoint) -> bool {
        todo!()
    }

    fn contains(&self, other: &MultiSphericalPoint) -> bool {
        other.within(self)
    }

    fn touches(&self, other: &MultiSphericalPoint) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn intersects(&self, other: &MultiSphericalPoint) -> bool {
        self.touches(other)
            || self.crosses(other)
            || self
                .polygons
                .iter()
                .any(|polygon| polygon.intersects(other))
    }
}

impl GeometricRelationships<ArcString> for MultiSphericalPolygon {
    fn distance(&self, other: &ArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn covers(&self, other: &ArcString) -> bool {
        todo!()
    }

    fn contains(&self, other: &ArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn crosses(&self, other: &ArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.crosses(other))
    }

    fn touches(&self, other: &ArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn intersects(&self, other: &ArcString) -> bool {
        self.touches(other) || self.crosses(other) || self.contains(other)
    }
}

impl GeometricRelationships<MultiArcString> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn covers(&self, other: &MultiArcString) -> bool {
        todo!()
    }

    fn contains(&self, other: &MultiArcString) -> bool {
        other
            .arcstrings
            .iter()
            .all(|arcstring| arcstring.within(self))
    }

    fn crosses(&self, other: &MultiArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.crosses(other))
    }

    fn touches(&self, other: &MultiArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn intersects(&self, other: &MultiArcString) -> bool {
        self.touches(other) || self.crosses(other) || other.intersects(self)
    }
}

impl GeometricRelationships<SphericalPolygon> for MultiSphericalPolygon {
    fn distance(&self, other: &SphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn equals(&self, other: &SphericalPolygon) -> bool {
        todo!()
    }

    fn covers(&self, other: &SphericalPolygon) -> bool {
        todo!()
    }

    fn contains(&self, other: &SphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn within(&self, other: &SphericalPolygon) -> bool {
        self.polygons.iter().all(|polygon| polygon.within(other))
    }

    fn crosses(&self, other: &SphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.crosses(other))
    }

    fn touches(&self, other: &SphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn overlaps(&self, other: &SphericalPolygon) -> bool {
        todo!()
    }

    fn intersects(&self, other: &SphericalPolygon) -> bool {
        self.polygons
            .iter()
            .any(|polygon| polygon.intersects(other))
    }
}

impl GeometricRelationships<Self> for MultiSphericalPolygon {
    fn distance(&self, other: &MultiSphericalPolygon) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.boundary().unwrap().distance(other)
        }
    }

    fn equals(&self, other: &Self) -> bool {
        self == other
    }

    fn covers(&self, other: &Self) -> bool {
        todo!()
    }

    fn contains(&self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons.iter().all(|polygon| polygon.within(other))
    }

    fn crosses(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.crosses(other))
    }

    fn touches(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.touches(other))
    }

    fn overlaps(&self, other: &Self) -> bool {
        todo!()
    }

    fn intersects(&self, other: &MultiSphericalPolygon) -> bool {
        self.polygons
            .iter()
            .any(|polygon| polygon.intersects(other))
    }
}

impl GeometricOperations<crate::sphericalpoint::SphericalPoint, SphericalPolygon>
    for MultiSphericalPolygon
{
    fn intersection(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::sphericalpoint::SphericalPoint) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::SphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::sphericalpoint::MultiSphericalPoint, SphericalPolygon>
    for MultiSphericalPolygon
{
    fn intersection(
        &self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::sphericalpoint::MultiSphericalPoint) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &crate::sphericalpoint::MultiSphericalPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::ArcString, SphericalPolygon> for MultiSphericalPolygon {
    fn intersection(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::ArcString) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<crate::arcstring::MultiArcString, SphericalPolygon>
    for MultiSphericalPolygon
{
    fn intersection(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &crate::arcstring::MultiArcString) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &crate::arcstring::MultiArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<SphericalPolygon, SphericalPolygon> for MultiSphericalPolygon {
    fn intersection(&self, other: &SphericalPolygon) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &SphericalPolygon) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<Self, SphericalPolygon> for MultiSphericalPolygon {
    fn intersection(&self, other: &Self) -> GeometryCollection {
        todo!()
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        todo!()
    }

    fn union(&self, other: &MultiSphericalPolygon) -> GeometryCollection {
        todo!()
    }
}
