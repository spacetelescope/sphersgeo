use kiddo::traits::DistanceMetric;
use numpy::ndarray::Axis;
use pyo3::prelude::*;

pub trait Geometry {
    fn area(&self) -> f64;

    fn length(&self) -> f64;

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        crate::sphericalpoint::SphericalPoint::try_from(
            self.coords().xyz.mean_axis(Axis(0)).unwrap(),
        )
        .unwrap()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        self.coords().bounds(degrees)
    }

    /// lower dimension geometry that bounds the object
    /// The boundary of a polygon is a line, the boundary of a line is a collection of points. The boundary of a point is an empty (null) collection.
    fn boundary(&self) -> Option<impl Geometry>;

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.coords().convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint;

    /// point guaranteed to be within the object
    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint;
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

pub trait GeometricOperations<O: Geometry = Self> {
    /// shortest great-circle distance over the sphere from any part of this geometry to another
    fn distance(self, other: O) -> f64;

    /// if this geometry completely envelops another
    fn contains(self, other: O) -> bool;

    /// if this entire geometry is completely within another
    fn within(self, other: O) -> bool;

    /// if any part of this geometry is within another
    fn intersects(self, other: O) -> bool;

    /// any part of this geometry that is within another
    ///
    /// NOTE: this function is NOT rigorous;
    /// it will only return the lower order of geometry being compared
    /// and will NOT handle degenerate cases or cases of touching vertices
    fn intersection(self, other: O) -> Option<impl Geometry>;

    /// if any part of this geometry (including the boundary) overlaps another
    fn touches(self, other: O) -> bool;
}

#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq)]
pub enum AnyGeometry {
    #[pyo3(transparent)]
    SphericalPoint(crate::sphericalpoint::SphericalPoint),
    #[pyo3(transparent)]
    MultiSphericalPoint(crate::sphericalpoint::MultiSphericalPoint),
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
            AnyGeometry::SphericalPoint(point) => point.area(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.area(),
            AnyGeometry::ArcString(arcstring) => arcstring.area(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.area(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.area(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.area(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.area(),
        }
    }

    fn length(&self) -> f64 {
        match self {
            AnyGeometry::SphericalPoint(point) => point.length(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.length(),
            AnyGeometry::ArcString(arcstring) => arcstring.length(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.length(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.length(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.length(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.length(),
        }
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        match self {
            AnyGeometry::SphericalPoint(point) => point.bounds(degrees),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.bounds(degrees),
            AnyGeometry::ArcString(arcstring) => arcstring.bounds(degrees),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.bounds(degrees),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.bounds(degrees),
            AnyGeometry::SphericalPolygon(polygon) => polygon.bounds(degrees),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.bounds(degrees),
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        match self {
            AnyGeometry::SphericalPoint(point) => point.convex_hull(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.convex_hull(),
            AnyGeometry::ArcString(arcstring) => arcstring.convex_hull(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.convex_hull(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.convex_hull(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.convex_hull(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.convex_hull(),
        }
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        match self {
            AnyGeometry::SphericalPoint(point) => point.coords(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.coords(),
            AnyGeometry::ArcString(arcstring) => arcstring.coords(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.coords(),
            AnyGeometry::AngularBounds(bounding_box) => bounding_box.coords(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.coords(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.coords(),
        }
    }

    fn boundary(&self) -> Option<AnyGeometry> {
        todo!()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        todo!()
    }
}

impl From<crate::sphericalpoint::SphericalPoint> for AnyGeometry {
    fn from(value: crate::sphericalpoint::SphericalPoint) -> Self {
        AnyGeometry::SphericalPoint(value)
    }
}

impl From<crate::sphericalpoint::MultiSphericalPoint> for AnyGeometry {
    fn from(value: crate::sphericalpoint::MultiSphericalPoint) -> Self {
        AnyGeometry::MultiSphericalPoint(value)
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

impl DistanceMetric<f64, 3> for AngularSeparation {
    #[inline]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        #[inline]
        fn normalize(vector: &[f64; 3]) -> [f64; 3] {
            let l = (vector[0].powi(2) + vector[1].powi(2) + vector[2].powi(2)).sqrt();
            [vector[0] / l, vector[1] / l, vector[2] / l]
        }

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
