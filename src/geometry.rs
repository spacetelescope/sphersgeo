pub trait BoundingBox {
    fn bounds(&self, degrees: bool) -> [f64; 4];
}

