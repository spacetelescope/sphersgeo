use crate::geometry::{ExtendMultiGeometry, Geometry, MultiGeometry};
use ndarray::array;
use pyo3::prelude::*;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

use crate::geometry::AnyGeometry;
/// collection of assorted geometries
#[pyclass]
#[derive(Debug, Clone)]
pub struct GeometryCollection {
    pub geometries: Vec<AnyGeometry>,
}

impl From<Vec<AnyGeometry>> for GeometryCollection {
    fn from(geometries: Vec<AnyGeometry>) -> Self {
        Self { geometries }
    }
}

impl GeometryCollection {
    pub fn empty() -> Self {
        Self { geometries: vec![] }
    }
}

impl Geometry for &GeometryCollection {
    fn area(&self) -> f64 {
        self.geometries.iter().map(|geometry| geometry.area()).sum()
    }

    fn length(&self) -> f64 {
        self.geometries
            .iter()
            .map(|geometry| geometry.length())
            .sum()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        if self.len() > 0 {
            self.points().bounds(degrees)
        } else {
            crate::angularbounds::AngularBounds::empty(degrees)
        }
    }

    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        if self.len() > 0 {
            self.geometries
                .iter()
                .map(|geometry| geometry.points())
                .sum()
        } else {
            unsafe {
                crate::vectorpoint::MultiVectorPoint::try_from(array![[
                    std::f64::NAN,
                    std::f64::NAN,
                    std::f64::NAN
                ]])
                .unwrap_unchecked()
            }
        }
    }
}

impl Geometry for GeometryCollection {
    fn area(&self) -> f64 {
        todo!()
    }

    fn length(&self) -> f64 {
        todo!()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        todo!()
    }

    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        (&self).convex_hull()
    }
}

impl MultiGeometry for GeometryCollection {
    fn len(&self) -> usize {
        self.geometries.len()
    }
}

impl ExtendMultiGeometry<AnyGeometry> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: AnyGeometry) {
        self.geometries.push(other.into());
    }
}

impl ExtendMultiGeometry<crate::vectorpoint::VectorPoint> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::vectorpoint::VectorPoint) {
        self.geometries.push(AnyGeometry::VectorPoint(other));
    }
}

impl ExtendMultiGeometry<crate::vectorpoint::MultiVectorPoint> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::vectorpoint::MultiVectorPoint) {
        self.geometries.push(AnyGeometry::MultiVectorPoint(other));
    }
}

impl ExtendMultiGeometry<crate::arcstring::ArcString> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::arcstring::ArcString) {
        self.geometries.push(AnyGeometry::ArcString(other));
    }
}

impl ExtendMultiGeometry<crate::arcstring::MultiArcString> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::arcstring::MultiArcString) {
        self.geometries.push(AnyGeometry::MultiArcString(other));
    }
}

impl ExtendMultiGeometry<crate::angularbounds::AngularBounds> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::angularbounds::AngularBounds) {
        self.geometries.push(AnyGeometry::AngularBounds(other));
    }
}

impl ExtendMultiGeometry<crate::angularpolygon::AngularPolygon> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::angularpolygon::AngularPolygon) {
        self.geometries.push(AnyGeometry::AngularPolygon(other));
    }
}

impl ExtendMultiGeometry<crate::angularpolygon::MultiAngularPolygon> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::angularpolygon::MultiAngularPolygon) {
        self.geometries
            .push(AnyGeometry::MultiAngularPolygon(other));
    }
}

impl Sum for GeometryCollection {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut geometries = vec![];
        for collection in iter {
            geometries.extend(collection.geometries);
        }

        Self::from(geometries)
    }
}

impl Add<&GeometryCollection> for &GeometryCollection {
    type Output = GeometryCollection;

    fn add(self, rhs: &GeometryCollection) -> Self::Output {
        let mut local = self.to_owned();
        local += rhs;
        local
    }
}

impl AddAssign<&GeometryCollection> for GeometryCollection {
    fn add_assign(&mut self, other: &GeometryCollection) {
        self.geometries.extend(other.geometries.to_owned());
    }
}
