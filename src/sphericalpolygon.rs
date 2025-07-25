use crate::{
    angularbounds::AngularBounds,
    arcstring::{
        vector_arc_angle, vector_arc_angles, vector_arcs_intersection, ArcString, MultiArcString,
    },
    geometry::{
        ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        MultiGeometryIntoIterator, MultiGeometryIterator,
    },
    vectorpoint::{min_1darray, shift_rows, MultiVectorPoint, VectorPoint},
};
use ndarray::{
    concatenate,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    s, stack, Array1, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use rayon::iter::IntoParallelIterator;
use std::{collections::VecDeque, iter::Sum};

/// surface area of a spherical triangle via Girard's theorum
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    vector_arc_angle(c, a, b, false)
        + vector_arc_angle(a, b, c, false)
        + vector_arc_angle(b, c, a, false)
        - std::f64::consts::PI
}

// calculate the interior angles of the polygon comprised of the given points
pub fn spherical_polygon_interior_angles(points: &ArrayView2<f64>, degrees: bool) -> Array1<f64> {
    vector_arc_angles(
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
    pub interior_point: VectorPoint,
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
        interior: VectorPoint,
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
                }
            }

            Ok(Self {
                exterior: exterior.closed(),
                interior_point: interior,
                holes: holes.map(|multiarcstring| {
                    let points: Vec<SphericalPolygon> = multiarcstring
                        .arcstrings
                        .into_par_iter()
                        .map(|arcstring| {
                            let centroid = arcstring.centroid();
                            SphericalPolygon::new(arcstring, centroid, None).unwrap()
                        })
                        .collect();
                    MultiSphericalPolygon::from(points)
                }),
            })
        }
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

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.exterior.points()
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

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).exterior.points()
    }
}

impl GeometricOperations<&VectorPoint> for &SphericalPolygon {
    fn distance(self, other: &VectorPoint) -> f64 {
        if self.contains(other) {
            0.0
        } else {
            self.exterior.distance(other)
        }
    }

    fn contains(self, other: &VectorPoint) -> bool {
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

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &VectorPoint) -> Option<VectorPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiVectorPoint> for &SphericalPolygon {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        min_1darray(
            &Zip::from(other.xyz.rows())
                .par_map_collect(|point| {
                    unsafe { VectorPoint::try_from(point.to_owned()).unwrap_unchecked() }
                        .distance(&self.exterior)
                })
                .view(),
        )
        .unwrap_or(0.0)
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
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

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiVectorPoint) -> Option<MultiVectorPoint> {
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
                MultiVectorPoint::try_from(
                    stack(Axis(0), intersections.as_slice()).unwrap_unchecked(),
                )
                .unwrap_unchecked()
            })
        } else {
            None
        }
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        todo!()
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiArcString) -> Option<MultiArcString> {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &SphericalPolygon {
    fn distance(self, other: &AngularBounds) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularBounds) -> bool {
        self.contains(&other.points())
    }

    fn within(self, _: &AngularBounds) -> bool {
        false
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        other
            .convex_hull()
            .map_or(false, |convex_hull| self.intersects(&convex_hull))
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularBounds) -> Option<MultiSphericalPolygon> {
        if let Some(convex_hull) = other.convex_hull() {
            self.intersection(&convex_hull)
        } else {
            None
        }
    }
}

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.contains(&other.points())
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.intersection(other).is_some()
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> Option<MultiSphericalPolygon> {
        todo!()
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        other.intersection(self)
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
        self.points().convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.polygons
            .par_iter()
            .map(|geometry| geometry.points())
            .sum()
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

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).points()
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

impl GeometricOperations<&VectorPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &VectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &VectorPoint) -> bool {
        self.polygons
            .par_iter()
            .any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &VectorPoint) -> Option<VectorPoint> {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiVectorPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        other.within(self)
    }

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        other.intersects(self)
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiVectorPoint) -> Option<MultiVectorPoint> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &ArcString) -> Option<MultiArcString> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
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

    #[allow(refining_impl_trait)]
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
        self.contains(&other.points())
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

    #[allow(refining_impl_trait)]
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

    #[allow(refining_impl_trait)]
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

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        self.polygons
            .par_iter()
            .map(|polygon| polygon.intersection(other))
            .sum()
    }
}

impl<'a> Iterator for MultiGeometryIterator<'a, MultiSphericalPolygon> {
    type Item = &'a SphericalPolygon;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.multi.len() {
            Some(&self.multi.polygons[self.index])
        } else {
            None
        }
    }
}

impl MultiSphericalPolygon {
    #[allow(dead_code)]
    fn iter(&self) -> MultiGeometryIterator<MultiSphericalPolygon> {
        MultiGeometryIterator::<MultiSphericalPolygon> {
            multi: self,
            index: 0,
        }
    }
}

impl Iterator for MultiGeometryIntoIterator<MultiSphericalPolygon> {
    type Item = SphericalPolygon;

    fn next(&mut self) -> Option<Self::Item> {
        self.multi.polygons.pop_front()
    }
}

impl IntoIterator for MultiSphericalPolygon {
    type Item = SphericalPolygon;

    type IntoIter = MultiGeometryIntoIterator<MultiSphericalPolygon>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            multi: self,
            index: 0,
        }
    }
}
