use crate::{geometrycollection::GeometryCollection, vectorpoint::MultiVectorPoint};
use kiddo::traits::DistanceMetric;
use pyo3::prelude::*;

pub trait Geometry {
    /// area of this geometry
    fn area(&self) -> f64;

    /// length of this geometry
    fn length(&self) -> f64;

    /// bounding box [minX,minY,maxX,maxY]
    fn bounds(&self, degrees: bool) -> [f64; 4];

    /// convex hull of this geometry
    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon>;

    fn points(&self) -> MultiVectorPoint;
}

pub trait MultiGeometry {
    /// number of elements in this collection
    fn len(&self) -> usize;
}

pub trait ExtendMultiGeometry<T: Geometry> {
    /// extend this collection with geometries from the other collection
    fn extend(&mut self, other: Self);

    /// append the geometry to this collection
    fn push(&mut self, other: T);
}

pub trait GeometricOperations<OtherGeometry: Geometry = Self> {
    /// distance between this geometry and another
    fn distance(self, other: OtherGeometry) -> f64;

    /// whether this geometry contains another
    fn contains(self, other: OtherGeometry) -> bool;

    /// whether this geometry is within another
    fn within(self, other: OtherGeometry) -> bool;

    /// whether this geometry and another given geometry intersect
    fn intersects(self, other: OtherGeometry) -> bool;

    /// intersection between this geometry and another given geometry
    fn intersection(self, other: OtherGeometry) -> GeometryCollection;
}

#[pyclass]
#[derive(Debug)]
pub enum AnyGeometry {
    VectorPoint(crate::vectorpoint::VectorPoint),
    MultiVectorPoint(crate::vectorpoint::MultiVectorPoint),
    ArcString(crate::arcstring::ArcString),
    SphericalPolygon(crate::sphericalpolygon::SphericalPolygon),
    MultiSphericalPolygon(crate::sphericalpolygon::MultiSphericalPolygon),
}

impl Geometry for AnyGeometry {
    fn area(&self) -> f64 {
        match self {
            Self::VectorPoint(point) => point.area(),
            Self::MultiVectorPoint(multipoint) => multipoint.area(),
            Self::ArcString(arcstring) => arcstring.area(),
            Self::SphericalPolygon(polygon) => polygon.area(),
            Self::MultiSphericalPolygon(multipolygon) => multipolygon.area(),
        }
    }

    fn length(&self) -> f64 {
        match self {
            Self::VectorPoint(point) => point.length(),
            Self::MultiVectorPoint(multipoint) => multipoint.length(),
            Self::ArcString(arcstring) => arcstring.length(),
            Self::SphericalPolygon(polygon) => polygon.length(),
            Self::MultiSphericalPolygon(multipolygon) => multipolygon.length(),
        }
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        match self {
            Self::VectorPoint(point) => point.bounds(degrees),
            Self::MultiVectorPoint(multipoint) => multipoint.bounds(degrees),
            Self::ArcString(arcstring) => arcstring.bounds(degrees),
            Self::SphericalPolygon(polygon) => polygon.bounds(degrees),
            Self::MultiSphericalPolygon(multipolygon) => multipolygon.bounds(degrees),
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        match self {
            Self::VectorPoint(point) => point.convex_hull(),
            Self::MultiVectorPoint(multipoint) => multipoint.convex_hull(),
            Self::ArcString(arcstring) => arcstring.convex_hull(),
            Self::SphericalPolygon(polygon) => polygon.convex_hull(),
            Self::MultiSphericalPolygon(multipolygon) => multipolygon.convex_hull(),
        }
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        match self {
            Self::VectorPoint(point) => point.points(),
            Self::MultiVectorPoint(multipoint) => multipoint.points(),
            Self::ArcString(arcstring) => arcstring.points(),
            Self::SphericalPolygon(polygon) => polygon.points(),
            Self::MultiSphericalPolygon(multipolygon) => multipolygon.points(),
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
