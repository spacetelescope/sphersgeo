use std::collections::VecDeque;

use crate::arcstring::angle;
use crate::{
    arcstring::{angles, ArcString},
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

/// surface area of a spherical triangle via Girard's theorum
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    angle(c, a, b, false) + angle(a, b, c, false) + angle(b, c, a, false) - std::f64::consts::PI
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
    angles(
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
#[derive(Clone, Debug)]
pub struct AngularPolygon {
    pub arcstring: ArcString,
    pub interior: VectorPoint,
    pub holes: Vec<ArcString>,
}

impl TryFrom<Array2<f64>> for AngularPolygon {
    type Error = String;

    fn try_from(xyz: Array2<f64>) -> Result<Self, Self::Error> {
        Self::try_from(ArcString::try_from(MultiVectorPoint::try_from(xyz)?)?)
    }
}

impl TryFrom<MultiVectorPoint> for AngularPolygon {
    type Error = String;

    fn try_from(multipoint: MultiVectorPoint) -> Result<Self, Self::Error> {
        Self::try_from(ArcString::try_from(multipoint)?)
    }
}

impl TryFrom<ArcString> for AngularPolygon {
    type Error = String;

    /// create a simple polygon without holes, and assume the interior point is within the lower-area
    fn try_from(arcstring: ArcString) -> Result<Self, Self::Error> {
        // TODO: check for self-intersections
        let interior = todo!();

        Ok(Self {
            arcstring,
            interior,
            holes: vec![],
        })
    }
}

impl ToString for AngularPolygon {
    fn to_string(&self) -> String {
        format!("Polygon({:?})", self.arcstring)
    }
}

impl AngularPolygon {
    fn interior_angles(&self, degrees: bool) -> Array1<f64> {
        spherical_polygon_interior_angles(&self.arcstring.points.xyz.view(), degrees)
    }
}

impl Geometry for &AngularPolygon {
    /// surface area of a spherical polygon via deconstructing into triangles
    /// https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf
    fn area(&self) -> f64 {
        self.interior_angles(false).sum()
            - ((self.arcstring.points.len() - 1) as f64 * std::f64::consts::PI)
    }

    /// length of this polygon
    fn length(&self) -> f64 {
        todo!();
    }

    fn convex_hull(&self) -> Option<AngularPolygon> {
        self.arcstring.convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.arcstring.points()
    }
}

impl Geometry for AngularPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<AngularPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).arcstring.points()
    }
}

impl GeometricOperations<&VectorPoint> for &AngularPolygon {
    fn distance(self, other: &VectorPoint) -> f64 {
        self.arcstring.distance(other)
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

impl GeometricOperations<&MultiVectorPoint> for &AngularPolygon {
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

impl GeometricOperations<&ArcString> for &AngularPolygon {
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

impl GeometricOperations<&AngularPolygon> for &AngularPolygon {
    fn distance(self, other: &AngularPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &AngularPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &AngularPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &AngularPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiAngularPolygon> for &AngularPolygon {
    fn distance(self, other: &MultiAngularPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiAngularPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &MultiAngularPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiAngularPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiAngularPolygon) -> GeometryCollection {
        other.intersection(self)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiAngularPolygon {
    pub polygons: VecDeque<AngularPolygon>,
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl From<Vec<AngularPolygon>> for MultiAngularPolygon {
    fn from(polygons: Vec<AngularPolygon>) -> Self {
        let mut points = vec![];
        for polygon in &polygons {
            points.extend(
                polygon
                    .arcstring
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

impl Geometry for &MultiAngularPolygon {
    fn area(&self) -> f64 {
        todo!();
    }

    fn length(&self) -> f64 {
        todo!();
    }

    fn convex_hull(&self) -> Option<AngularPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.polygons.iter().map(|geometry| geometry.points()).sum()
    }
}

impl Geometry for MultiAngularPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<AngularPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).points()
    }
}

impl MultiGeometry for &MultiAngularPolygon {
    fn len(&self) -> usize {
        self.polygons.len()
    }
}

impl MultiGeometry for MultiAngularPolygon {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl ExtendMultiGeometry<AngularPolygon> for MultiAngularPolygon {
    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: AngularPolygon) {
        self.polygons.push_back(value);
    }
}

impl GeometricOperations<&VectorPoint> for &MultiAngularPolygon {
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

impl GeometricOperations<&MultiVectorPoint> for &MultiAngularPolygon {
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

impl GeometricOperations<&ArcString> for &MultiAngularPolygon {
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

impl GeometricOperations<&AngularPolygon> for &MultiAngularPolygon {
    fn distance(self, other: &AngularPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &AngularPolygon) -> bool {
        self.polygons.iter().any(|polygon| polygon.contains(other))
    }

    fn within(self, _: &AngularPolygon) -> bool {
        false
    }

    fn intersects(self, other: &AngularPolygon) -> bool {
        todo!()
    }

    fn intersection(self, other: &AngularPolygon) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiAngularPolygon> for &MultiAngularPolygon {
    fn distance(self, other: &MultiAngularPolygon) -> f64 {
        todo!();
    }

    fn contains(self, other: &MultiAngularPolygon) -> bool {
        todo!();
    }

    fn within(self, other: &MultiAngularPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiAngularPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    fn intersection(self, other: &MultiAngularPolygon) -> GeometryCollection {
        todo!();
    }
}

impl<'a> Iterator for MultiGeometryIterator<'a, MultiAngularPolygon> {
    type Item = AngularPolygon;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.multi.len() {
            Some(self.multi.polygons[self.index].to_owned())
        } else {
            None
        }
    }
}

impl MultiAngularPolygon {
    fn iter(&self) -> MultiGeometryIterator<MultiAngularPolygon> {
        MultiGeometryIterator::<MultiAngularPolygon> {
            multi: self,
            index: 0,
        }
    }
}

impl Iterator for MultiGeometryIntoIterator<MultiAngularPolygon> {
    type Item = AngularPolygon;

    fn next(&mut self) -> Option<Self::Item> {
        self.multi.polygons.pop_front()
    }
}

impl IntoIterator for MultiAngularPolygon {
    type Item = AngularPolygon;

    type IntoIter = MultiGeometryIntoIterator<MultiAngularPolygon>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            multi: self,
            index: 0,
        }
    }
}
