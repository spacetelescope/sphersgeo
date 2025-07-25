use crate::{geometrycollection::GeometryCollection, vectorpoint::MultiVectorPoint};
use kiddo::traits::DistanceMetric;
use pyo3::prelude::*;

pub trait Geometry {
    /// area of this geometry
    fn area(&self) -> f64;

    /// length of this geometry
    fn length(&self) -> f64;

    /// bounding box [minX,minY,maxX,maxY]
    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        self.points().bounds(degrees)
    }

    /// convex hull of this geometry
    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint;
}

pub trait MultiGeometry {
    /// number of elements in this collection
    fn len(&self) -> usize;
}

pub struct MultiGeometryIterator<'a, M: MultiGeometry> {
    pub multi: &'a M,
    pub index: usize,
}

pub struct MultiGeometryIntoIterator<M: MultiGeometry> {
    pub multi: M,
    pub index: usize,
}

pub trait ExtendMultiGeometry<T: Geometry> {
    /// extend this collection with geometries from the other collection
    fn extend(&mut self, other: Self);

    /// append the geometry to this collection
    fn push(&mut self, other: T);
}

pub trait GeometricOperations<O: Geometry = Self> {
    /// distance between this geometry and another
    fn distance(self, other: O) -> f64;

    /// whether this geometry contains another
    fn contains(self, other: O) -> bool;

    /// whether this geometry is within another
    fn within(self, other: O) -> bool;

    /// whether this geometry and another given geometry intersect
    fn intersects(self, other: O) -> bool;

    /// intersection between this geometry and another given geometry
    fn intersection(self, other: O) -> crate::geometrycollection::GeometryCollection;
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum AnyGeometry {
    VectorPoint(crate::vectorpoint::VectorPoint),
    MultiVectorPoint(crate::vectorpoint::MultiVectorPoint),
    ArcString(crate::arcstring::ArcString),
    MultiArcString(crate::arcstring::MultiArcString),
    AngularBounds(crate::angularbounds::AngularBounds),
    AngularPolygon(crate::angularpolygon::AngularPolygon),
    MultiAngularPolygon(crate::angularpolygon::MultiAngularPolygon),
}

impl Geometry for AnyGeometry {
    fn area(&self) -> f64 {
        match self {
            Self::VectorPoint(point) => point.area(),
            Self::MultiVectorPoint(multipoint) => multipoint.area(),
            Self::ArcString(arcstring) => arcstring.area(),
            Self::MultiArcString(multiarcstring) => multiarcstring.area(),
            Self::AngularBounds(bounding_box) => bounding_box.area(),
            Self::AngularPolygon(polygon) => polygon.area(),
            Self::MultiAngularPolygon(multipolygon) => multipolygon.area(),
        }
    }

    fn length(&self) -> f64 {
        match self {
            Self::VectorPoint(point) => point.length(),
            Self::MultiVectorPoint(multipoint) => multipoint.length(),
            Self::ArcString(arcstring) => arcstring.length(),
            Self::MultiArcString(multiarcstring) => multiarcstring.length(),
            Self::AngularBounds(bounding_box) => bounding_box.length(),
            Self::AngularPolygon(polygon) => polygon.length(),
            Self::MultiAngularPolygon(multipolygon) => multipolygon.length(),
        }
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        match self {
            Self::VectorPoint(point) => point.bounds(degrees),
            Self::MultiVectorPoint(multipoint) => multipoint.bounds(degrees),
            Self::ArcString(arcstring) => arcstring.bounds(degrees),
            Self::MultiArcString(multiarcstring) => multiarcstring.bounds(degrees),
            Self::AngularBounds(bounding_box) => bounding_box.bounds(degrees),
            Self::AngularPolygon(polygon) => polygon.bounds(degrees),
            Self::MultiAngularPolygon(multipolygon) => multipolygon.bounds(degrees),
        }
    }

    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        match self {
            Self::VectorPoint(point) => point.convex_hull(),
            Self::MultiVectorPoint(multipoint) => multipoint.convex_hull(),
            Self::ArcString(arcstring) => arcstring.convex_hull(),
            Self::MultiArcString(multiarcstring) => multiarcstring.convex_hull(),
            Self::AngularBounds(bounding_box) => bounding_box.convex_hull(),
            Self::AngularPolygon(polygon) => polygon.convex_hull(),
            Self::MultiAngularPolygon(multipolygon) => multipolygon.convex_hull(),
        }
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        match self {
            Self::VectorPoint(point) => point.points(),
            Self::MultiVectorPoint(multipoint) => multipoint.points(),
            Self::ArcString(arcstring) => arcstring.points(),
            Self::MultiArcString(multiarcstring) => multiarcstring.points(),
            Self::AngularBounds(bounding_box) => bounding_box.points(),
            Self::AngularPolygon(polygon) => polygon.points(),
            Self::MultiAngularPolygon(multipolygon) => multipolygon.points(),
        }
    }
}

/// define angular separation between 3D vectors
pub struct AngularSeparation {}

#[inline]
fn normalize(vector: &[f64; 3]) -> [f64; 3] {
    let l = (vector[0].powi(2) + vector[1].powi(2) + vector[2].powi(2)).sqrt();
    [vector[0] / l, vector[1] / l, vector[2] / l]
}

impl DistanceMetric<f64, 3> for AngularSeparation {
    #[inline]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let a = normalize(a);
        let b = normalize(b);
        // radians subtended
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).acos()
    }

    #[inline]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}
