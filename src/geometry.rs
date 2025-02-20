pub trait BoundingBox {
    /// bounding box [minX,minY,maxX,maxY]
    fn bounds(&self, degrees: bool) -> [f64; 4];
}

pub trait Distance {
    /// distance between this geometry and another
    fn distance(&self, other: Self) -> f64;
}
