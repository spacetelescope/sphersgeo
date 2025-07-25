use crate::geometry::{GeometricOperations, Geometry};
use numpy::ndarray::{array, s, Array1};
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

impl Into<[f64; 4]> for &AngularBounds {
    fn into(self) -> [f64; 4] {
        [self.min_x, self.min_y, self.max_x, self.max_y]
    }
}

impl Into<Array1<f64>> for &AngularBounds {
    fn into(self) -> Array1<f64> {
        array![self.min_x, self.min_y, self.max_x, self.max_y]
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

    pub fn to_radians(&self) -> Self {
        if self.degrees {
            Self {
                min_x: self.min_x.to_radians(),
                min_y: self.min_y.to_radians(),
                max_x: self.max_x.to_radians(),
                max_y: self.max_y.to_radians(),
                degrees: false,
            }
        } else {
            self.to_owned()
        }
    }

    pub fn to_degrees(&self) -> Self {
        if !self.degrees {
            Self {
                min_x: self.min_x.to_radians(),
                min_y: self.min_y.to_radians(),
                max_x: self.max_x.to_radians(),
                max_y: self.max_y.to_radians(),
                degrees: false,
            }
        } else {
            self.to_owned()
        }
    }
}

impl ToString for AngularBounds {
    fn to_string(&self) -> String {
        let this: [f64; 4] = self.into();
        format!("AngularBounds({:?}, degrees: {})", this, self.degrees)
    }
}

impl PartialEq for AngularBounds {
    fn eq(&self, other: &AngularBounds) -> bool {
        let tolerance = 3e-11;
        let this: Array1<f64> = self.into();
        let other: Array1<f64> = other.into();
        (this - other).abs().sum() < tolerance
    }
}

impl Geometry for &AngularBounds {
    fn area(&self) -> f64 {
        let xyz = self.points().xyz;
        crate::sphericalpolygon::spherical_triangle_area(
            &xyz.slice(s![0, ..]),
            &xyz.slice(s![1, ..]),
            &xyz.slice(s![2, ..]),
        ) + crate::sphericalpolygon::spherical_triangle_area(
            &xyz.slice(s![2, ..]),
            &xyz.slice(s![3, ..]),
            &xyz.slice(s![0, ..]),
        )
    }

    fn length(&self) -> f64 {
        crate::arcstring::arc_length_from_vectors(
            &array![self.min_x, self.min_y].view(),
            &array![self.max_x, self.max_y].view(),
        )
    }

    fn bounds(&self, degrees: bool) -> AngularBounds {
        if degrees == self.degrees {
            (*self).to_owned()
        } else if degrees {
            self.to_degrees()
        } else {
            self.to_radians()
        }
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        crate::sphericalpolygon::SphericalPolygon::try_from(self.points()).ok()
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

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }
}

impl GeometricOperations<&crate::vectorpoint::VectorPoint> for &AngularBounds {
    fn distance(self, other: &crate::vectorpoint::VectorPoint) -> f64 {
        match self.convex_hull() {
            Some(hull) => hull.distance(other),
            None => self.points().distance(other),
        }
    }

    fn contains(self, other: &crate::vectorpoint::VectorPoint) -> bool {
        other.within(self)
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

impl GeometricOperations<&crate::arcstring::MultiArcString> for &AngularBounds {
    fn distance(self, other: &crate::arcstring::MultiArcString) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::arcstring::MultiArcString) -> bool {
        todo!()
    }

    fn within(self, other: &crate::arcstring::MultiArcString) -> bool {
        todo!()
    }

    fn intersects(self, other: &crate::arcstring::MultiArcString) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::arcstring::MultiArcString,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &AngularBounds {
    fn distance(self, other: &AngularBounds) -> f64 {
        self.points().distance(&other.points())
    }

    fn contains(self, other: &AngularBounds) -> bool {
        other.within(self)
    }

    fn within(self, other: &AngularBounds) -> bool {
        other.min_x > self.min_x
            && other.min_y > self.min_y
            && other.max_x < self.max_x
            && other.max_y < self.max_y
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        let other = if self.degrees == other.degrees {
            other
        } else if other.degrees {
            &other.to_radians()
        } else {
            &other.to_degrees()
        };

        other.min_x < self.max_x
            || other.max_x > self.min_x
            || other.min_y < self.max_y
            || other.max_y > self.min_y
    }

    fn intersection(self, other: &AngularBounds) -> crate::geometrycollection::GeometryCollection {
        let other = if self.degrees == other.degrees {
            other
        } else if other.degrees {
            &other.to_radians()
        } else {
            &other.to_degrees()
        };

        if self.intersects(other) {
            let min_x = if other.min_x > self.min_x {
                other.min_x
            } else {
                self.min_x
            };
            let min_y = if other.min_y > self.min_y {
                other.min_y
            } else {
                self.min_y
            };
            let max_x = if other.max_x < self.max_x {
                other.max_x
            } else {
                self.max_x
            };
            let max_y = if other.max_y < self.max_y {
                other.max_y
            } else {
                self.max_y
            };

            crate::geometrycollection::GeometryCollection::from(vec![
                crate::geometry::AnyGeometry::AngularBounds(AngularBounds {
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    degrees: self.degrees,
                }),
            ])
        } else {
            crate::geometrycollection::GeometryCollection::empty()
        }
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        todo!()
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        todo!()
    }

    fn contains(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}
