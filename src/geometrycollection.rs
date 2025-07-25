use crate::geometry::AnyGeometry;
use crate::geometry::{ExtendMultiGeometry, Geometry, MultiGeometry};
use ndarray::array;
use pyo3::prelude::*;
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

/// collection of assorted geometries
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryCollection {
    // TODO: figure out how to make this dyn Geometry
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
            self.coords().bounds(degrees)
        } else {
            crate::angularbounds::AngularBounds::empty(degrees)
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.coords().convex_hull()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        if self.len() > 0 {
            self.geometries
                .iter()
                .map(|geometry| geometry.coords())
                .sum()
        } else {
            crate::sphericalpoint::MultiSphericalPoint::try_from(array![[
                std::f64::NAN,
                std::f64::NAN,
                std::f64::NAN
            ]])
            .unwrap()
        }
    }

    fn boundary(&self) -> Option<GeometryCollection> {
        todo!()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        todo!()
    }
}

impl Geometry for GeometryCollection {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).coords()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn boundary(&self) -> Option<GeometryCollection> {
        (&self).boundary()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
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
        self.geometries.push(other);
    }
}

impl ExtendMultiGeometry<crate::sphericalpoint::SphericalPoint> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::sphericalpoint::SphericalPoint) {
        self.geometries.push(AnyGeometry::SphericalPoint(other));
    }
}

impl ExtendMultiGeometry<crate::sphericalpoint::MultiSphericalPoint> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::sphericalpoint::MultiSphericalPoint) {
        self.geometries
            .push(AnyGeometry::MultiSphericalPoint(other));
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

impl ExtendMultiGeometry<crate::sphericalpolygon::SphericalPolygon> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::sphericalpolygon::SphericalPolygon) {
        self.geometries.push(AnyGeometry::SphericalPolygon(other));
    }
}

impl ExtendMultiGeometry<crate::sphericalpolygon::MultiSphericalPolygon> for GeometryCollection {
    fn extend(&mut self, other: GeometryCollection) {
        self.geometries.extend(other.geometries);
    }

    fn push(&mut self, other: crate::sphericalpolygon::MultiSphericalPolygon) {
        self.geometries
            .push(AnyGeometry::MultiSphericalPolygon(other));
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
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&GeometryCollection> for GeometryCollection {
    fn add_assign(&mut self, other: &GeometryCollection) {
        self.geometries.extend(other.geometries.to_owned());
    }
}

impl ToString for GeometryCollection {
    fn to_string(&self) -> String {
        format!("GeometryCollection({:?})", self.geometries)
    }
}
