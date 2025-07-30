use kiddo::traits::DistanceMetric;
use pyo3::prelude::*;

pub trait Geometry {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint;

    /// lower dimension geometry that bounds the object
    /// The boundary of a polygon is a line, the boundary of a line is a collection of endpoints, and the boundary of a point is null.
    fn boundary(&self) -> Option<impl Geometry>;

    /// point guaranteed to be within the object
    fn representative(&self) -> crate::sphericalpoint::SphericalPoint;

    // mean position of all possible points within the geometry
    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint;

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.vertices().convex_hull()
    }

    fn area(&self) -> f64;

    fn length(&self) -> f64;
}

pub trait MultiGeometry<G: Geometry> {
    /// number of elements in this collection
    fn len(&self) -> usize;

    /// extend this collection with geometries from the other collection
    fn extend(&mut self, other: Self);

    /// append the geometry to this collection
    fn push(&mut self, other: G);
}

pub trait GeometricPredicates<O: Geometry = Self> {
    /// Two geometries intersect if they share at least one point in common.
    /// https://esri.github.io/geometry-api-java/doc/Intersects.html
    fn intersects(&self, other: &O) -> bool;

    /// An object is said to touch other if it has at least one point in common with other and its interior does not intersect with any part of the other.
    /// Overlapping features therefore do not touch.
    /// https://esri.github.io/geometry-api-java/doc/Touches.html
    fn touches(&self, other: &O) -> bool;

    /// Two geometries are disjoint if they don’t have any points in common.
    /// Disjoint is the inverse of Intersects. Disjoint is the most efficient operator and is guaranteed to work even on non-simple geometries.
    /// https://esri.github.io/geometry-api-java/doc/Disjoint.html
    fn disjoint(&self, other: &O) -> bool {
        !self.intersects(other)
    }

    /// the geometries have some, but not all interior points in common
    ///
    /// Two polylines cross if they meet at (a) point/s only, and at least one of the shared points is internal to both polylines.
    /// A polyline and polygon cross if a connected part of the polyline is partly inside and partly outside the polygon.
    /// https://esri.github.io/geometry-api-java/doc/Crosses.html
    fn crosses(&self, other: &O) -> bool;

    /// One geometry is within another if it is a subset of the other geometry and their interiors have at least one point in common. Within is the inverse of Contains.
    /// https://esri.github.io/geometry-api-java/doc/Within.html
    fn within(&self, other: &O) -> bool;

    /// One geometry contains another if the other geometry is a subset of it and their interiors have at least one point in common.
    /// Contains is the inverse of Within.
    /// https://esri.github.io/geometry-api-java/doc/Contains.html
    fn contains(&self, other: &O) -> bool;

    /// Two geometries overlap if they have the same dimension, and their intersection also has the same dimension but is different from both of them.
    /// https://esri.github.io/geometry-api-java/doc/Overlaps.html
    fn overlaps(&self, other: &O) -> bool {
        false
    }

    /// Whether every point of other is a point on the interior or boundary of object.
    /// This is similar to object.contains(other) except that this does not require any interior points of other to lie in the interior of object.
    fn covers(&self, other: &O) -> bool;
}

pub trait GeometricOperations<O: Geometry = Self, S: Geometry = Self> {
    fn union(&self, other: &O) -> Option<impl MultiGeometry<S>>;

    /// shortest great-circle distance over the sphere from any part of this geometry to another
    fn distance(&self, other: &O) -> f64;

    /// any part of this geometry that is within another
    ///
    /// NOTE: this function is NOT rigorous;
    /// it will ONLY return the lower order of geometry being compared
    /// and will NOT handle touching, collinear overlap, or degenerate cases
    fn intersection(&self, other: &O) -> Option<impl Geometry>;

    /// split this geometry into a multi-geometry, at the crossing with the given geometry
    fn symmetric_difference(&self, other: &O) -> impl MultiGeometry<S>;
}

pub trait GeometryCollection<G: Geometry, M: MultiGeometry<G> = Self> {
    /// join geometries into one
    fn join_self(&self) -> M;

    /// find overlapping regions between geometries, if any
    fn overlap_self(&self) -> Option<M>;

    /// only return non-overlapping regions between geometries
    fn symmetric_difference_self(&self) -> Option<M>;
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

/// define angular separation between 3D vectors
pub struct AngularSeparation {}

impl DistanceMetric<f64, 3> for AngularSeparation {
    #[inline]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        // radians subtended
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).acos()
    }

    #[inline]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}
