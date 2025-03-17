use crate::{
    angularbounds::AngularBounds,
    arcstring::{
        vector_arcs_angle_between, vector_arcs_angles_between, vector_arcs_intersection, ArcString,
        MultiArcString,
    },
    geometry::{ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry},
    sphericalpoint::{min_1darray, shift_rows, MultiSphericalPoint, SphericalPoint},
};
use ndarray::{
    array, concatenate,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    s, stack, Array1, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use rayon::iter::IntoParallelIterator;
use std::{cmp::Ordering, collections::VecDeque, iter::Sum};

/// surface area of a spherical triangle via Girard's theorum
///
///     θ_1 + θ_2 + θ_3 − π
///
/// References
/// ----------
/// - Klain, D. A. (2019). A probabilistic proof of the spherical excess formula (No. arXiv:1909.04505). arXiv. https://doi.org/10.48550/arXiv.1909.04505
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    vector_arcs_angle_between(c, a, b, false)
        + vector_arcs_angle_between(a, b, c, false)
        + vector_arcs_angle_between(b, c, a, false)
        - std::f64::consts::PI
}

// calculate the interior angles of the polygon comprised of the given points
pub fn spherical_polygon_interior_angles(points: &ArrayView2<f64>, degrees: bool) -> Array1<f64> {
    vector_arcs_angles_between(
        &points.slice(s![-2..-2, ..]),
        &shift_rows(points, -2).view(),
        &shift_rows(points, -1).view(),
        degrees,
    )
}

/// surface area of a spherical polygon via deconstructing into triangles
/// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
pub fn spherical_polygon_area(points: &ArrayView2<f64>) -> f64 {
    spherical_polygon_interior_angles(points, false).sum()
        - ((points.nrows() - 1) as f64 * std::f64::consts::PI)
}

// use the classical even-crossings ray algorithm for point-in-polygon
fn point_in_polygon_exterior(
    point: &ArrayView1<f64>,
    polygon_interior_point: &ArrayView1<f64>,
    polygon_exterior_points: &ArrayView2<f64>,
) -> bool {
    // include the final connection back to the first point
    let exterior_arcstring_points = concatenate(
        Axis(0),
        &[
            polygon_exterior_points.view(),
            polygon_exterior_points
                .slice(s![0, ..])
                .broadcast((1, 3))
                .unwrap(),
        ],
    )
    .unwrap();

    // record the number of times the ray intersects the exterior arcstring
    let mut crossings = 0;
    for arc_index in 0..exterior_arcstring_points.nrows() - 1 {
        let arc_0 = exterior_arcstring_points.slice(s![arc_index, ..]);
        let arc_1 = exterior_arcstring_points.slice(s![arc_index + 1, ..]);

        if vector_arcs_intersection(&point, &polygon_interior_point, &arc_0, &arc_1).is_some() {
            crossings += 1;
        }
    }

    // if the number of crossings is even, the point is within the polygon's exterior
    crossings % 2 == 0
}

/// polygon on the sphere, comprising:
/// 1. a non-intersecting collection of connected arcs (arcstring) that connects back to its first point (closed)
/// 2. an interior point to specify which region of the sphere the polygon represents; this is required for non-Euclidian geometry
/// 3. a series of closed arcstrings representing holes inside the exterior polygon
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalPolygon {
    pub exterior: ArcString,
    pub interior_point: SphericalPoint,
    pub holes: Option<MultiSphericalPolygon>,
}

impl ToString for SphericalPolygon {
    fn to_string(&self) -> String {
        format!("AngularPolygon({:?})", self.exterior)
    }
}

impl SphericalPolygon {
    /// interior point is required because a sphere is a finite space
    pub fn new(
        exterior: ArcString,
        interior_point: SphericalPoint,
        holes: Option<MultiArcString>,
    ) -> Result<Self, String> {
        if let Some(intersections) = exterior.intersection_with_self() {
            Err(format!(
                "arcstring intersects itself: {}",
                intersections.xyz
            ))
        } else {
            if let Some(holes) = &holes {
                for hole in &holes.arcstrings {
                    if let Some(intersections) = hole.intersection_with_self() {
                        return Err(format!(
                            "arcstring intersects itself: {}",
                            intersections.xyz
                        ));
                    }

                    if point_in_polygon_exterior(
                        &interior_point.xyz.view(),
                        &hole.centroid().xyz.view(),
                        &hole.points.xyz.view(),
                    ) {
                        return Err(format!(
                            "interior point {} is within hole {}",
                            interior_point.xyz, hole.points.xyz
                        ));
                    }
                }
            }

            let holes = holes.map(|holes| {
                let points: Vec<SphericalPolygon> = holes
                    .arcstrings
                    .into_par_iter()
                    .map(|arcstring| {
                        let centroid = arcstring.centroid();
                        SphericalPolygon::new(arcstring, centroid, None).unwrap()
                    })
                    .collect();
                MultiSphericalPolygon::from(points)
            });

            Ok(Self {
                exterior: exterior.closed(),
                interior_point,
                holes,
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
            .par_map_collect(|spoke| xyz.vector_rotate_around(&center, spoke, false).xyz)
            .to_vec();

        Self::new(
            ArcString::from(MultiSphericalPoint::try_from(&vertices).unwrap()),
            center.to_owned(),
            None,
        )
        .unwrap()
    }

    fn interior_angles(&self, degrees: bool) -> Array1<f64> {
        spherical_polygon_interior_angles(&self.exterior.points.xyz.view(), degrees)
    }
}

impl Geometry for &SphericalPolygon {
    /// surface area of a spherical polygon via deconstructing into triangles
    /// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
    fn area(&self) -> f64 {
        self.interior_angles(false).sum()
            - ((self.exterior.points.len() - 1) as f64 * std::f64::consts::PI)
    }

    /// length of this polygon
    fn length(&self) -> f64 {
        todo!()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.exterior.convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.exterior.coords()
    }

    fn boundary(&self) -> Option<ArcString> {
        Some(self.exterior.to_owned())
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        self.interior_point.to_owned()
    }
}

impl Geometry for SphericalPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).exterior.coords()
    }

    fn boundary(&self) -> Option<ArcString> {
        (&self).boundary()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
    }
}

impl GeometricOperations<&SphericalPoint> for &SphericalPolygon {
    fn distance(self, other: &SphericalPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.exterior.distance(other)
        }
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        if point_in_polygon_exterior(
            &other.xyz.view(),
            &self.interior_point.xyz.view(),
            &self.exterior.points.xyz.view(),
        ) {
            // check against holes, if any
            if let Some(holes) = &self.holes {
                // if the point is within a hole, it is not within the polygon
                !holes.polygons.par_iter().any(|hole| {
                    point_in_polygon_exterior(
                        &other.xyz.view(),
                        // calculate the centroid of the hole's exterior
                        &hole.interior_point.xyz.view(),
                        &hole.exterior.points.xyz.view(),
                    )
                })
            } else {
                // if there are no holes
                true
            }
        } else {
            // if the point is NOT within the polygon exterior
            false
        }
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.exterior.intersects(other) || self.intersects(other)
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &SphericalPolygon {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        min_1darray(
            &Zip::from(other.xyz.rows())
                .par_map_collect(|point| {
                    unsafe { SphericalPoint::try_from(point.to_owned()).unwrap_unchecked() }
                        .distance(&self.exterior)
                })
                .view(),
        )
        .unwrap_or(0.0)
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        // use the classical even-crossings ray algorithm for point-in-polygon
        for point in other.xyz.rows() {
            if point_in_polygon_exterior(
                &point,
                &self.interior_point.xyz.view(),
                &self.exterior.points.xyz.view(),
            ) {
                // check against holes, if any
                if let Some(holes) = &self.holes {
                    if holes.polygons.par_iter().any(|hole| {
                        point_in_polygon_exterior(
                            &point,
                            // calculate the centroid of the hole's exterior
                            &hole.interior_point.xyz.view(),
                            &hole.exterior.points.xyz.view(),
                        )
                    }) {
                        // if the point is within a hole, it is not within the polygon
                        return false;
                    }
                }
            } else {
                // if the point is NOT within the polygon exterior
                return false;
            }
        }

        // if none of the points returned false
        return true;
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        // use the classical even-crossings ray algorithm for point-in-polygon
        for point in other.xyz.rows() {
            if point_in_polygon_exterior(
                &point,
                &self.interior_point.xyz.view(),
                &self.exterior.points.xyz.view(),
            ) {
                // check against holes, if any
                if let Some(holes) = &self.holes {
                    // if the point is within a hole, it is not within the polygon
                    if !holes.polygons.par_iter().any(|hole| {
                        point_in_polygon_exterior(
                            &point,
                            // calculate the centroid of the hole's exterior
                            &hole.interior_point.xyz.view(),
                            &hole.exterior.points.xyz.view(),
                        )
                    }) {
                        // if the point is not within any holes
                        return true;
                    }
                } else {
                    // if there are no holes
                    return true;
                }
            }
        }

        // if no points returned true
        return false;
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        let mut intersections = vec![];

        // use the classical even-crossings ray algorithm for point-in-polygon
        for point in other.xyz.rows() {
            if point_in_polygon_exterior(
                &point,
                &self.interior_point.xyz.view(),
                &self.exterior.points.xyz.view(),
            ) {
                // check against holes, if any
                if let Some(holes) = &self.holes {
                    // if the point is within a hole, it is not within the polygon
                    if !holes.polygons.par_iter().any(|hole| {
                        point_in_polygon_exterior(
                            &point,
                            // calculate the centroid of the hole's exterior
                            &hole.interior_point.xyz.view(),
                            &hole.exterior.points.xyz.view(),
                        )
                    }) {
                        // if the point is not within any holes
                        intersections.push(point);
                    }
                } else {
                    // if there are no holes
                    intersections.push(point);
                }
            }
        }

        if intersections.len() > 0 {
            Some(unsafe {
                MultiSphericalPoint::try_from(
                    stack(Axis(0), intersections.as_slice()).unwrap_unchecked(),
                )
                .unwrap_unchecked()
            })
        } else {
            None
        }
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.exterior.intersects(other) || self.intersects(other)
    }
}

impl GeometricOperations<&ArcString> for &SphericalPolygon {
    fn distance(self, other: &ArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &ArcString) -> bool {
        self.contains(&other.points)
    }

    fn within(self, _: &ArcString) -> bool {
        false
    }

    fn intersects(self, other: &ArcString) -> bool {
        if self.contains(other) {
            if let Some(holes) = &self.holes {
                // make sure the arcstring does not fully reside within a hole
                !holes.polygons.par_iter().any(|hole| hole.contains(other))
            } else {
                true
            }
        } else {
            self.exterior.intersects(other)
        }
    }

    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn touches(self, other: &ArcString) -> bool {
        self.exterior.intersects(&other.points) || self.intersects(other)
    }
}

impl GeometricOperations<&MultiArcString> for &SphericalPolygon {
    fn distance(self, other: &MultiArcString) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.exterior.distance(other)
        }
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, _: &MultiArcString) -> bool {
        false
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }

    fn touches(self, other: &MultiArcString) -> bool {
        self.exterior.intersects(&other) || self.intersects(other)
    }
}

impl GeometricOperations<&AngularBounds> for &SphericalPolygon {
    fn distance(self, other: &AngularBounds) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularBounds) -> bool {
        self.contains(&other.coords())
    }

    fn within(self, _: &AngularBounds) -> bool {
        false
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        other
            .convex_hull()
            .map_or(false, |convex_hull| self.intersects(&convex_hull))
    }

    fn intersection(self, other: &AngularBounds) -> Option<MultiSphericalPolygon> {
        if let Some(convex_hull) = other.convex_hull() {
            self.intersection(&convex_hull)
        } else {
            None
        }
    }

    fn touches(self, other: &AngularBounds) -> bool {
        self.exterior.touches(&other.boundary().unwrap())
    }
}

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.contains(&other.coords())
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.exterior.intersects(&other.exterior)
    }

    fn intersection(self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
    }

    fn touches(self, other: &SphericalPolygon) -> bool {
        self.exterior.touches(&other.exterior)
            || self
                .holes
                .as_ref()
                .is_some_and(|holes| holes.touches(&other.exterior))
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        other.intersection(self)
    }

    fn touches(self, other: &MultiSphericalPolygon) -> bool {
        todo!()
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

impl ToString for MultiSphericalPolygon {
    fn to_string(&self) -> String {
        format!("MultiAngularPolygon({:?})", self.polygons)
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

        return true;
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

        return true;
    }
}

impl Geometry for &MultiSphericalPolygon {
    fn area(&self) -> f64 {
        self.polygons.par_iter().map(|polygon| polygon.area()).sum()
    }

    fn length(&self) -> f64 {
        todo!()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.coords().convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.polygons
            .par_iter()
            .map(|geometry| geometry.coords())
            .sum()
    }

    fn boundary(&self) -> Option<MultiArcString> {
        let arcstrings: Vec<ArcString> = self
            .polygons
            .par_iter()
            .filter_map(|polygon| polygon.boundary())
            .collect();
        Some(MultiArcString::from(arcstrings))
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        self.polygons[0].representative_point()
    }
}

impl Geometry for MultiSphericalPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).coords()
    }

    fn boundary(&self) -> Option<MultiArcString> {
        (&self).boundary()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
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
    fn distance(self, other: &SphericalPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &SphericalPoint) -> Option<SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &SphericalPoint) -> bool {
        self.intersects(other) || self.boundary().unwrap().touches(&other.boundary().unwrap())
    }
}

impl GeometricOperations<&MultiSphericalPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiSphericalPoint) -> bool {
        other.within(self)
    }

    fn within(self, _: &MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiSphericalPoint) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiSphericalPoint) -> Option<MultiSphericalPoint> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn touches(self, other: &MultiSphericalPoint) -> bool {
        self.intersects(other) || self.boundary().unwrap().touches(&other.boundary().unwrap())
    }
}

impl GeometricOperations<&ArcString> for &MultiSphericalPolygon {
    fn distance(self, other: &ArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &ArcString) -> bool {
        false
    }

    fn intersects(self, other: &ArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }

    fn touches(self, other: &ArcString) -> bool {
        self.intersects(other) || self.boundary().unwrap().touches(&other.boundary().unwrap())
    }
}

impl GeometricOperations<&MultiArcString> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiArcString) -> bool {
        todo!()
    }

    fn within(self, other: &MultiArcString) -> bool {
        todo!()
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| self.intersects(polygon))
    }

    fn intersection(self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &MultiSphericalPolygon {
    fn distance(self, other: &AngularBounds) -> f64 {
        if let Some(convex_hull) = other.convex_hull() {
            self.distance(&convex_hull)
        } else {
            0.0
        }
    }

    fn contains(self, other: &AngularBounds) -> bool {
        self.contains(&other.coords())
    }

    fn within(self, other: &AngularBounds) -> bool {
        self.polygons
            .par_iter()
            .all(|polygon| polygon.within(other))
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &AngularBounds) -> Option<MultiSphericalPolygon> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.intersects(other))
    }

    fn intersection(self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
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
}
