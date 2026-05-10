use kiddo::traits::DistanceMetric;

pub trait Geometry {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint;

    /// lower dimension geometry that bounds this geometry's interior
    ///
    /// The boundary of a polygon is a closed arcstring,
    /// the boundary of an arcstring is two endpoints (unless closed),
    /// and the boundary of a point (and a closed arcstring) is null.
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

/// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/
pub trait GeometricRelationships<O: Geometry = Self> {
    /// Whether this and the other geometry's interiors are identical and the geometry types are the same.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#equals
    fn equals(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share ANY point(s).
    /// If this geometries contains, is within, crosses, touches, or overlaps the other geometry, they intersect.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#intersects
    fn intersects(&self, other: &O) -> bool;

    /// Whether the other geometry is a subset of this geometry
    /// (every point of the other geometry is a point on the interior OR boundary of this geometry).
    fn covers(&self, other: &O) -> bool {
        false
    }

    /// Whether this geometry covers the other geometry AND the interiors share at least one point.
    ///
    /// Contains is the inverse of Within.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains
    fn contains(&self, other: &O) -> bool {
        false
    }

    /// Whether the other geometry covers this geometry AND the interiors share at least one point.
    ///
    /// Within is the inverse of Contains.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains
    fn within(&self, other: &O) -> bool {
        false
    }

    /// Whether this arcstring / polygon and the other arcstring / polygon share only SOME (not all) interior points, but do NOT overlap.
    ///
    /// Two arcstrings cross if they meet at point(s) only, and at least one of the shared points is internal to both arcstrings.
    /// An arcstring and polygon cross if they share an arcstring on the interior of the polygon, which is NOT equal to the entire arcstring.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#crosses
    fn crosses(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share any vertices but do not overlap.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#touches
    fn touches(&self, other: &O) -> bool;

    /// Whether this and the other geometry are of the same geometry type,
    /// AND their intersection is also of the same geometry type BUT is not equal to either.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#overlaps
    fn overlaps(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry do NOT share ANY point(s).
    ///
    /// Disjoint is the inverse of Intersects.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#disjoint
    fn disjoint(&self, other: &O) -> bool {
        !self.intersects(other)
    }
}

pub trait GeometricOperations<O: Geometry = Self, S: Geometry = Self> {
    fn union(&self, other: &O) -> Option<impl MultiGeometry<S>>;

    /// shortest great-circle distance over the sphere from any part of this geometry to another
    fn distance(&self, other: &O) -> f64;

    /// any part of this geometry that is within another
    ///
    /// NOTE: this function is NOT rigorous;
    /// it will ONLY return the lower order of geometry being compared
    /// and will NOT handle touching, colinear overlap, or degenerate cases
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
