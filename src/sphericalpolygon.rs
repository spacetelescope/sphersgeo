use crate::{
    arcstring::ArcString,
    geometry::{ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry},
    geometrycollection::GeometryCollection,
    vectorpoint::{MultiVectorPoint, VectorPoint},
};
use kiddo::ImmutableKdTree;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SphericalPolygon {
    arcstring: ArcString,
    interior: VectorPoint,
}

impl SphericalPolygon {
    fn new(arcstring: ArcString, interior: VectorPoint) -> SphericalPolygon {
        Self {
            arcstring,
            interior,
        }
    }
}

impl Geometry for &SphericalPolygon {
    fn area(&self) -> f64 {
        todo!();
    }

    fn length(&self) -> f64 {
        todo!();
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        self.arcstring.bounds(degrees)
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.arcstring.convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.arcstring.points()
    }
}

impl Geometry for SphericalPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        (&self).bounds(degrees)
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).arcstring.points()
    }
}

impl GeometricOperations<&VectorPoint> for &SphericalPolygon {
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
    polygons: Vec<SphericalPolygon>,
    kdtree: ImmutableKdTree<f64, 3>,
}

impl From<Vec<SphericalPolygon>> for MultiSphericalPolygon {
    fn from(polygons: Vec<SphericalPolygon>) -> Self {
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
            polygons,
            kdtree: points.as_slice().into(),
        }
    }
}

impl Geometry for &MultiSphericalPolygon {
    fn area(&self) -> f64 {
        todo!();
    }

    fn length(&self) -> f64 {
        todo!();
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        self.points().bounds(degrees)
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

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        (&self).bounds(degrees)
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

impl ExtendMultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: SphericalPolygon) {
        self.polygons.push(value);
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
