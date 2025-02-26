use kiddo::traits::DistanceMetric;

pub trait Geometry {
    /// area of this geometry
    fn area(&self) -> f64;

    /// length of this geometry
    fn length(&self) -> f64;

    /// bounding box [minX,minY,maxX,maxY]
    fn bounds(&self, degrees: bool) -> [f64; 4];

    /// convex hull of this geometry
    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon>;
}

pub trait SingleGeometry {}

pub trait MultiGeometry<T: SingleGeometry> {
    /// elements of this collection
    fn parts(&self) -> Vec<T>;

    /// number of elements in this collection
    fn len(&self) -> usize;
}

pub trait GeometryCollection<T: SingleGeometry> {
    /// extend this collection with elements from the other collection
    fn extend(&mut self, other: Self);

    /// extend this collection with elements from the other collection
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
    fn intersection(self, other: OtherGeometry) -> Option<impl Geometry>;
}

pub struct SphericalDistanceMetric {}

#[inline]
fn normalize(vector: &[f64; 3]) -> [f64; 3] {
    let l = (vector[0].powi(2) + vector[1].powi(2) + vector[2].powi(2)).sqrt();
    [vector[0] / l, vector[1] / l, vector[2] / l]
}

impl DistanceMetric<f64, 3> for SphericalDistanceMetric {
    #[inline]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let a = normalize(a);
        let b = normalize(b);
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[inline]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}
