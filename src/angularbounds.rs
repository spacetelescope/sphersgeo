use crate::geometry::{GeometricOperations, Geometry};
use numpy::ndarray::{array, s};
use pyo3::prelude::*;

/// an ortholinear rectangle aligned with the angular axes of the sphere
#[pyclass]
#[derive(Debug, Clone)]
pub struct AngularBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub degrees: bool,
}

impl From<[f64; 4]> for AngularBounds {
    fn from(degrees: [f64; 4]) -> Self {
        Self {
            min_x: degrees[0],
            min_y: degrees[1],
            max_x: degrees[2],
            max_y: degrees[3],
            degrees: true,
        }
    }
}

impl Into<[f64; 4]> for AngularBounds {
    fn into(self) -> [f64; 4] {
        [self.min_x, self.min_y, self.max_x, self.max_y]
    }
}

impl AngularBounds {
    pub fn empty(degrees: bool) -> Self {
        Self {
            min_x: std::f64::NAN,
            min_y: std::f64::NAN,
            max_x: std::f64::NAN,
            max_y: std::f64::NAN,
            degrees,
        }
    }
}

impl Geometry for &AngularBounds {
    fn area(&self) -> f64 {
        let points = self.points();
        crate::angularpolygon::spherical_triangle_area(
            &points.xyz.slice(s![0, ..]),
            &points.xyz.slice(s![1, ..]),
            &points.xyz.slice(s![2, ..]),
        ) + crate::angularpolygon::spherical_triangle_area(
            &points.xyz.slice(s![1, ..]),
            &points.xyz.slice(s![2, ..]),
            &points.xyz.slice(s![3, ..]),
        )
    }

    fn length(&self) -> f64 {
        crate::arcstring::arc_length(
            &array![self.min_x, self.min_y].view(),
            &array![self.max_x, self.max_y].view(),
        )
    }

    fn bounds(&self, degrees: bool) -> AngularBounds {
        let min_x;
        let min_y;
        let max_x;
        let max_y;

        if degrees != self.degrees {
            if degrees {
                min_x = self.min_x.to_degrees();
                min_y = self.min_y.to_degrees();
                max_x = self.max_x.to_degrees();
                max_y = self.max_y.to_degrees();
            } else {
                min_x = self.min_x.to_radians();
                min_y = self.min_y.to_radians();
                max_x = self.max_x.to_radians();
                max_y = self.max_y.to_radians();
            }
        } else {
            min_x = self.min_x;
            min_y = self.min_y;
            max_x = self.max_x;
            max_y = self.max_y;
        }

        AngularBounds {
            min_x,
            min_y,
            max_x,
            max_y,
            degrees,
        }
    }

    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        Some(crate::angularpolygon::AngularPolygon::try_from(self.points()).unwrap())
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        crate::vectorpoint::MultiVectorPoint::try_from_lonlats(
            &array![
                [self.min_x, self.min_y],
                [self.min_x, self.max_y],
                [self.max_x, self.max_y],
                [self.max_x, self.min_y]
            ]
            .view(),
            self.degrees,
        )
        .unwrap()
    }
}

impl Geometry for AngularBounds {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn bounds(&self, degrees: bool) -> crate::angularbounds::AngularBounds {
        (&self).bounds(degrees)
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        (&self).points()
    }

    fn convex_hull(&self) -> Option<crate::angularpolygon::AngularPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<&crate::vectorpoint::VectorPoint> for &AngularBounds {
    fn distance(self, other: &crate::vectorpoint::VectorPoint) -> f64 {
        self.convex_hull().unwrap().distance(other)
    }

    fn contains(self, other: &crate::vectorpoint::VectorPoint) -> bool {
        let point = other.to_lonlat(self.degrees);

        point[0] >= self.min_x
            && point[1] >= self.min_y
            && point[0] <= self.max_x
            && point[1] <= self.max_y
    }

    fn within(self, _: &crate::vectorpoint::VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &crate::vectorpoint::VectorPoint) -> bool {
        self.contains(other)
    }

    fn intersection(
        self,
        other: &crate::vectorpoint::VectorPoint,
    ) -> crate::geometrycollection::GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&crate::vectorpoint::MultiVectorPoint> for &AngularBounds {
    fn distance(self, other: &crate::vectorpoint::MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::vectorpoint::MultiVectorPoint) -> bool {
        other.within(self)
    }

    fn within(self, _: &crate::vectorpoint::MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &crate::vectorpoint::MultiVectorPoint) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::vectorpoint::MultiVectorPoint,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&crate::arcstring::ArcString> for &AngularBounds {
    fn distance(self, other: &crate::arcstring::ArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::arcstring::ArcString) -> bool {
        todo!()
    }

    fn within(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn intersects(self, other: &crate::arcstring::ArcString) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::arcstring::ArcString,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&crate::vectorpoint::MultiVectorPoint> for &AngularBounds {
    fn distance(self, other: &crate::vectorpoint::MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::vectorpoint::MultiVectorPoint) -> bool {
        todo!()
    }

    fn within(self, _: &crate::vectorpoint::MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &crate::vectorpoint::MultiVectorPoint) -> bool {
        other.intersects(self)
    }

    fn intersection(
        self,
        other: &crate::vectorpoint::MultiVectorPoint,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&crate::angularpolygon::AngularPolygon> for &AngularBounds {
    fn distance(self, other: &crate::angularpolygon::AngularPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::angularpolygon::AngularPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &crate::angularpolygon::AngularPolygon) -> bool {
        todo!()
    }

    fn intersects(self, other: &crate::angularpolygon::AngularPolygon) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::angularpolygon::AngularPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&crate::angularpolygon::MultiAngularPolygon> for &AngularBounds {
    fn distance(self, other: &crate::angularpolygon::MultiAngularPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::angularpolygon::MultiAngularPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &crate::angularpolygon::MultiAngularPolygon) -> bool {
        todo!()
    }

    fn intersects(self, other: &crate::angularpolygon::MultiAngularPolygon) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::angularpolygon::MultiAngularPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}
