use kiddo::traits::DistanceMetric;
use pyo3::prelude::*;

pub trait Geometry {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint;

    fn area(&self) -> f64;

    fn length(&self) -> f64;

    /// point guaranteed to be within the object
    fn representative(&self) -> crate::sphericalpoint::SphericalPoint;

    // mean position of all possible points within the geometry
    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint;

    /// lower dimension geometry that bounds the object
    /// The boundary of a polygon is a line, the boundary of a line is a collection of endpoints, and the boundary of a point is null.
    fn boundary(&self) -> Option<impl Geometry>;

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.vertices().convex_hull()
    }
}

pub trait MultiGeometry<G: Geometry> {
    /// number of elements in this collection
    fn len(&self) -> usize;

    /// extend this collection with geometries from the other collection
    fn extend(&mut self, other: Self);

    /// append the geometry to this collection
    fn push(&mut self, other: G);
}

pub trait GeometricOperations<O: Geometry = Self, S: Geometry = Self> {
    /// shortest great-circle distance over the sphere from any part of this geometry to another
    fn distance(&self, other: &O) -> f64;

    /// One geometry contains another if the other geometry is a subset of it and their interiors have at least one point in common.
    /// Contains is the inverse of Within.
    /// https://esri.github.io/geometry-api-java/doc/Contains.html
    fn contains(&self, other: &O) -> bool;

    /// One geometry is within another if it is a subset of the other geometry and their interiors have at least one point in common. Within is the inverse of Contains.
    /// https://esri.github.io/geometry-api-java/doc/Within.html
    fn within(&self, other: &O) -> bool;

    /// An object is said to touch other if it has at least one point in common with other and its interior does not intersect with any part of the other.
    /// Overlapping features therefore do not touch.
    /// https://esri.github.io/geometry-api-java/doc/Touches.html
    fn touches(&self, other: &O) -> bool;

    /// the geometries have some, but not all interior points in common
    ///
    /// Two polylines cross if they meet at (a) point/s only, and at least one of the shared points is internal to both polylines.
    /// A polyline and polygon cross if a connected part of the polyline is partly inside and partly outside the polygon.
    /// https://esri.github.io/geometry-api-java/doc/Crosses.html
    fn crosses(&self, other: &O) -> bool;

    /// Two geometries intersect if they share at least one point in common.
    /// https://esri.github.io/geometry-api-java/doc/Intersects.html
    fn intersects(&self, other: &O) -> bool;

    /// any part of this geometry that is within another
    ///
    /// NOTE: this function is NOT rigorous;
    /// it will ONLY return the lower order of geometry being compared
    /// and will NOT handle touching, collinear overlap, or degenerate cases
    fn intersection(&self, other: &O) -> Option<impl Geometry>;

    /// split this geometry into a multi-geometry, at the crossing with the given geometry
    fn split(&self, other: &O) -> impl MultiGeometry<S>;
}

pub trait GeometryCollection<G: Geometry, M: MultiGeometry<G>> {
    /// join geometries into one; errors if any of the geometries are disjoint
    fn join(&self) -> Result<G, String>;

    /// find overlapping regions between geometries, if any
    fn overlap(&self) -> Option<G>;

    /// split geometries so none are overlapping
    fn split(&self) -> M;
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
    SphericalPolygon(crate::sphericalpolygon::SphericalPolygon),
    #[pyo3(transparent)]
    MultiSphericalPolygon(crate::sphericalpolygon::MultiSphericalPolygon),
}

impl Geometry for AnyGeometry {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        match self {
            AnyGeometry::SphericalPoint(point) => point.vertices(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.vertices(),
            AnyGeometry::ArcString(arcstring) => arcstring.vertices(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.vertices(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.vertices(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.vertices(),
        }
    }

    fn area(&self) -> f64 {
        match self {
            AnyGeometry::SphericalPoint(point) => point.area(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.area(),
            AnyGeometry::ArcString(arcstring) => arcstring.area(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.area(),
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
            AnyGeometry::SphericalPolygon(polygon) => polygon.length(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.length(),
        }
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        match self {
            AnyGeometry::SphericalPoint(point) => point.representative(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.representative(),
            AnyGeometry::ArcString(arcstring) => arcstring.representative(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.representative(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.representative(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.representative(),
        }
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        match self {
            AnyGeometry::SphericalPoint(point) => point.centroid(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.centroid(),
            AnyGeometry::ArcString(arcstring) => arcstring.centroid(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.centroid(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.centroid(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.centroid(),
        }
    }

    fn boundary(&self) -> Option<AnyGeometry> {
        match self {
            AnyGeometry::SphericalPoint(point) => point.boundary().map(AnyGeometry::SphericalPoint),
            AnyGeometry::MultiSphericalPoint(multipoint) => {
                multipoint.boundary().map(AnyGeometry::MultiSphericalPoint)
            }
            AnyGeometry::ArcString(arcstring) => {
                arcstring.boundary().map(AnyGeometry::MultiSphericalPoint)
            }
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring
                .boundary()
                .map(AnyGeometry::MultiSphericalPoint),
            AnyGeometry::SphericalPolygon(polygon) => {
                polygon.boundary().map(AnyGeometry::ArcString)
            }
            AnyGeometry::MultiSphericalPolygon(multipolygon) => {
                multipolygon.boundary().map(AnyGeometry::MultiArcString)
            }
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        match self {
            AnyGeometry::SphericalPoint(point) => point.convex_hull(),
            AnyGeometry::MultiSphericalPoint(multipoint) => multipoint.convex_hull(),
            AnyGeometry::ArcString(arcstring) => arcstring.convex_hull(),
            AnyGeometry::MultiArcString(multiarcstring) => multiarcstring.convex_hull(),
            AnyGeometry::SphericalPolygon(polygon) => polygon.convex_hull(),
            AnyGeometry::MultiSphericalPolygon(multipolygon) => multipolygon.convex_hull(),
        }
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
