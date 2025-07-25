use kiddo::traits::DistanceMetric;
use numpy::ndarray::Axis;
use pyo3::prelude::*;

pub trait Geometry {
    fn area(&self) -> f64;

    fn length(&self) -> f64;

    fn centroid(&self) -> crate::vectorpoint::VectorPoint {
        crate::vectorpoint::VectorPoint::try_from(self.points().xyz.mean_axis(Axis(0)).unwrap())
            .unwrap()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        self.points().bounds(degrees)
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
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
    fn distance(self, other: O) -> f64;

    fn contains(self, other: O) -> bool;

    fn within(self, other: O) -> bool;

    fn intersects(self, other: O) -> bool;

    fn intersection(self, other: O) -> crate::geometrycollection::GeometryCollection;
}

#[derive(FromPyObject, Debug, Clone, PartialEq)]
pub enum AnyGeometry {
    #[pyo3(transparent)]
    VectorPoint(crate::vectorpoint::VectorPoint),
    #[pyo3(transparent)]
    MultiVectorPoint(crate::vectorpoint::MultiVectorPoint),
    #[pyo3(transparent)]
    ArcString(crate::arcstring::ArcString),
    #[pyo3(transparent)]
    MultiArcString(crate::arcstring::MultiArcString),
    #[pyo3(transparent)]
    AngularBounds(crate::angularbounds::AngularBounds),
    #[pyo3(transparent)]
    SphericalPolygon(crate::sphericalpolygon::SphericalPolygon),
    #[pyo3(transparent)]
    MultiSphericalPolygon(crate::sphericalpolygon::MultiSphericalPolygon),
}

impl Geometry for AnyGeometry {
    fn area(&self) -> f64 {
        match self {
            AnyGeometry::VectorPoint(point) => point.area(),
            AnyGeometry::MultiVectorPoint(multipoint) => multipoint.area(),
            AnyGeometry::ArcString(arcstring) => arcstring.area(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.area(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.area(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.area(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.area(),
        }
    }

    fn length(&self) -> f64 {
        match self {
            AnyGeometry::VectorPoint(point) => point.length(),
            AnyGeometry::MultiVectorPoint(multipoint) => multipoint.length(),
            AnyGeometry::ArcString(arcstring) => arcstring.length(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.length(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.length(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.length(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.length(),
        }
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        match self {
            AnyGeometry::VectorPoint(point) => point.bounds(degrees),
            AnyGeometry::MultiVectorPoint(multipoint) => multipoint.bounds(degrees),
            AnyGeometry::ArcString(arcstring) => arcstring.bounds(degrees),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.bounds(degrees),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.bounds(degrees),
            AnyGeometry::SphericalPolygon(polygon) => polygon.bounds(degrees),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.bounds(degrees),
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        match self {
            AnyGeometry::VectorPoint(point) => point.convex_hull(),
            AnyGeometry::MultiVectorPoint(multipoint) => multipoint.convex_hull(),
            AnyGeometry::ArcString(arcstring) => arcstring.convex_hull(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.convex_hull(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.convex_hull(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.convex_hull(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.convex_hull(),
        }
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        match self {
            AnyGeometry::VectorPoint(point) => point.points(),
            AnyGeometry::MultiVectorPoint(multipoint) => multipoint.points(),
            AnyGeometry::ArcString(arcstring) => arcstring.points(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.points(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.points(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.points(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.points(),
        }
    }
}

impl From<crate::vectorpoint::VectorPoint> for AnyGeometry {
    fn from(value: crate::vectorpoint::VectorPoint) -> Self {
        AnyGeometry::VectorPoint(value)
    }
}

impl From<crate::vectorpoint::MultiVectorPoint> for AnyGeometry {
    fn from(value: crate::vectorpoint::MultiVectorPoint) -> Self {
        AnyGeometry::MultiVectorPoint(value)
    }
}

impl From<crate::arcstring::ArcString> for AnyGeometry {
    fn from(value: crate::arcstring::ArcString) -> Self {
        AnyGeometry::ArcString(value)
    }
}

impl From<crate::arcstring::MultiArcString> for AnyGeometry {
    fn from(value: crate::arcstring::MultiArcString) -> Self {
        AnyGeometry::MultiArcString(value)
    }
}

impl From<crate::angularbounds::AngularBounds> for AnyGeometry {
    fn from(value: crate::angularbounds::AngularBounds) -> Self {
        AnyGeometry::AngularBounds(value)
    }
}

impl From<crate::sphericalpolygon::SphericalPolygon> for AnyGeometry {
    fn from(value: crate::sphericalpolygon::SphericalPolygon) -> Self {
        AnyGeometry::SphericalPolygon(value)
    }
}

impl From<crate::sphericalpolygon::MultiSphericalPolygon> for AnyGeometry {
    fn from(value: crate::sphericalpolygon::MultiSphericalPolygon) -> Self {
        AnyGeometry::MultiSphericalPolygon(value)
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
