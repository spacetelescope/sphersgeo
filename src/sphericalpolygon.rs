use crate::{
    arcstring::ArcString,
    collection::GeometryCollection,
    geometry::{GeometricOperations, Geometry, MultiGeometry, MutableMultiGeometry},
    vectorpoint::VectorPoint,
};
use kiddo::ImmutableKdTree;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SphericalPolygon {
    arcstring: ArcString,
    interior: VectorPoint,
    kdtree: ImmutableKdTree<f64, 3>,
}

impl SphericalPolygon {
    fn new(arcstring: ArcString, interior: VectorPoint) -> SphericalPolygon {
        let points = arcstring
            .points
            .xyz
            .rows()
            .into_iter()
            .map(|point| [point[0], point[1], point[2]])
            .collect::<Vec<[f64; 3]>>();

        Self {
            arcstring,
            interior,
            kdtree: points.as_slice().into(),
        }
    }
}

impl Geometry for &SphericalPolygon {
    fn area(&self) -> f64 {
        // TODO: implement
        std::f64::NAN
    }

    fn length(&self) -> f64 {
        // TODO: implement
        std::f64::NAN
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

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        // TODO: implement
        std::f64::NAN
    }

    fn contains(self, other: &SphericalPolygon) -> bool {
        // TODO: implement
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &SphericalPolygon) -> GeometryCollection {
        // TODO: implement
        GeometryCollection::empty()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MultiSphericalPolygon {
    polygons: Vec<SphericalPolygon>,
    kdtree: ImmutableKdTree<f64, 3>,
}

impl MultiSphericalPolygon {
    fn new(polygons: Vec<SphericalPolygon>) -> MultiSphericalPolygon {
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
        // TODO: implement
        std::f64::NAN
    }

    fn length(&self) -> f64 {
        // TODO: implement
        std::f64::NAN
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

impl MutableMultiGeometry<SphericalPolygon> for MultiSphericalPolygon {
    fn extend(&mut self, other: Self) {
        self.polygons.extend(other.polygons);
    }

    fn push(&mut self, value: SphericalPolygon) {
        self.polygons.push(value);
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiSphericalPolygon {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        // TODO: implement
        std::f64::NAN
    }

    fn contains(self, other: &MultiSphericalPolygon) -> bool {
        // TODO: implement
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.intersection(other).len() > 0
    }

    #[allow(refining_impl_trait)]
    fn intersection(self, other: &MultiSphericalPolygon) -> GeometryCollection {
        // TODO: implement
        GeometryCollection::empty()
    }
}
