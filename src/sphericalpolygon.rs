use crate::{
    angularbounds::AngularBounds,
    arcstring::{arc_angle, arc_angles, ArcString, MultiArcString},
    geometry::{
        ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        MultiGeometryIntoIterator, MultiGeometryIterator,
    },
    geometrycollection::GeometryCollection,
    vectorpoint::{MultiVectorPoint, VectorPoint},
};
use kiddo::ImmutableKdTree;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// surface area of a spherical triangle via Girard's theorum
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    arc_angle(c, a, b, false) + arc_angle(a, b, c, false) + arc_angle(b, c, a, false)
        - std::f64::consts::PI
}

fn shift_rows(from: &ArrayView2<f64>, by: i32) -> Array2<f64> {
    let mut to = Array2::uninit(from.dim());
    from.slice(s![-by.., ..])
        .assign_to(to.slice_mut(s![..by, ..]));
    from.slice(s![..-by, ..])
        .assign_to(to.slice_mut(s![by.., ..]));

    unsafe { to.assume_init() }
}

// calculate the interior angles of the polygon comprised of the given points
pub fn spherical_polygon_interior_angles(points: &ArrayView2<f64>, degrees: bool) -> Array1<f64> {
    arc_angles(
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

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalPolygon {
    pub exterior: ArcString,
    pub interior: VectorPoint,
    pub holes: Option<MultiArcString>,
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
        // TODO: check for self-intersections
        Ok(Self {
            exterior,
            interior,
            holes,
        })
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
        todo!();
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
        self.exterior.distance(other)
    }

    fn contains(self, other: &VectorPoint) -> bool {
        todo!()
    }

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &VectorPoint) -> GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiVectorPoint> for &SphericalPolygon {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn intersection(self, other: &MultiVectorPoint) -> GeometryCollection {
        todo!()
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
        todo!()
    }

    fn intersection(self, other: &ArcString) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiArcString> for &SphericalPolygon {
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
        todo!()
    }

    fn intersection(self, other: &MultiArcString) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &SphericalPolygon {
    fn distance(self, other: &AngularBounds) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn within(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn intersection(self, other: &AngularBounds) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &SphericalPolygon {
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
        other.intersects(self)
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> GeometryCollection {
        other.intersection(self)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiSphericalPolygon {
    pub polygons: VecDeque<SphericalPolygon>,
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl From<Vec<SphericalPolygon>> for MultiSphericalPolygon {
    fn from(polygons: Vec<SphericalPolygon>) -> Self {
        let mut points = vec![];
        for polygon in &polygons {
            points.extend(
                polygon
                    .exterior
                    .points
                    .xyz
                    .rows()
                    .into_iter()
                    .map(|point| [point[0], point[1], point[2]]),
            );
        }

        Self {
            polygons: polygons.into(),
            kdtree: points.as_slice().into(),
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

        for polygon in other.iter() {
            if !self.contains(polygon) {
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

        for polygon in other {
            if !self.contains(polygon) {
                return false;
            }
        }

        return true;
    }
}

impl Geometry for &MultiSphericalPolygon {
    fn area(&self) -> f64 {
        todo!();
    }

    fn length(&self) -> f64 {
        todo!();
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.polygons.iter().map(|geometry| geometry.points()).sum()
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
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &VectorPoint) -> GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiVectorPoint> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn intersection(self, other: &MultiVectorPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&ArcString> for &MultiSphericalPolygon {
    fn distance(self, other: &ArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &ArcString) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &ArcString) -> bool {
        false
    }

    fn intersects(self, other: &ArcString) -> bool {
        todo!()
    }

    fn intersection(self, other: &ArcString) -> GeometryCollection {
        todo!()
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
        todo!()
    }

    fn intersection(self, other: &MultiArcString) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &MultiSphericalPolygon {
    fn distance(self, other: &AngularBounds) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn within(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn intersection(self, other: &AngularBounds) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        todo!()
    }

    fn intersection(self, other: &SphericalPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        todo!();
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        todo!();
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> GeometryCollection {
        todo!();
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
