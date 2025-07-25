use crate::{
    arcstring::ArcString,
    geometry::{GeometricOperations, Geometry},
    vectorpoint::{MultiVectorPoint, VectorPoint},
};
use kiddo::ImmutableKdTree;
use pyo3::prelude::*;

#[pyclass]
pub struct SphericalPolygon {
    arcstring: ArcString,
    interior: VectorPoint,
    kdtree: ImmutableKdTree<f64, 3>,
}

impl SphericalPolygon {
    fn new(arcstring: ArcString, interior: VectorPoint) -> SphericalPolygon {
        let points = arcstring
            .points
            .xyz
            .rows()
            .into_iter()
            .map(|point| [point[0], point[1], point[2]])
            .collect::<Vec<[f64; 3]>>();

        Self {
            arcstring,
            interior,
            kdtree: points.as_slice().into(),
        }
    }
}

impl Geometry for &SphericalPolygon {
    fn area(&self) -> f64 {
        // TODO: implement
        -1.
    }

    fn length(&self) -> f64 {
        // TODO: implement
        -1.
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        // TODO: implement
        [-1., -1., -1., -1.]
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        // TODO: implement
        None
    }
}

impl Geometry for SphericalPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        (&self).bounds(degrees)
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<&SphericalPolygon> for &SphericalPolygon {
    fn distance(&self, other: &SphericalPolygon) -> f64 {
        // TODO: implement
        -1.
    }

    fn contains(&self, other: &SphericalPolygon) -> bool {
        // TODO: implement
        false
    }

    fn within(&self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(&self, other: &SphericalPolygon) -> bool {
        self.intersection(other).is_some()
    }

    fn intersection(&self, other: &SphericalPolygon) -> Option<SphericalPolygon> {
        // TODO: implement
        None
    }
}

#[pyclass]
pub struct MultiSphericalPolygon {
    polygons: Vec<SphericalPolygon>,
    kdtree: ImmutableKdTree<f64, 3>,
}

impl MultiSphericalPolygon {
    fn new(polygons: Vec<SphericalPolygon>) -> MultiSphericalPolygon {
        let mut points = vec![];
        for polygon in &polygons {
            points.extend(
                polygon
                    .arcstring
                    .points
                    .xyz
                    .rows()
                    .into_iter()
                    .map(|point| [point[0], point[1], point[2]]),
            );
        }

        Self {
            polygons,
            kdtree: points.as_slice().into(),
        }
    }
}

impl Geometry for &MultiSphericalPolygon {
    fn area(&self) -> f64 {
        // TODO: implement
        -1.
    }

    fn length(&self) -> f64 {
        // TODO: implement
        -1.
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        // TODO: implement
        [-1., -1., -1., -1.]
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        // TODO: implement
        None
    }
}

impl Geometry for MultiSphericalPolygon {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        (&self).bounds(degrees)
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiSphericalPolygon {
    fn distance(&self, other: &MultiSphericalPolygon) -> f64 {
        // TODO: implement
        -1.
    }

    fn contains(&self, other: &MultiSphericalPolygon) -> bool {
        // TODO: implement
        false
    }

    fn within(&self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(&self, other: &MultiSphericalPolygon) -> bool {
        self.intersection(other).is_some()
    }

    fn intersection(&self, other: &MultiSphericalPolygon) -> Option<MultiSphericalPolygon> {
        // TODO: implement
        None
    }
}
