use crate::geometry::{GeometricOperations, Geometry};
use ndarray::Array2;
use numpy::ndarray::{array, s, Array1, Axis};
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

    pub fn contains_lonlat(&self, lon: f64, lat: f64) -> bool {
        lon > self.min_x && lat > self.min_y && lon < self.max_x && lat < self.max_y
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
        let xyz = self.coords().xyz;
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
        crate::sphericalpoint::vector_arc_length(
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
        crate::sphericalpolygon::SphericalPolygon::new(
            crate::arcstring::ArcString {
                points: self.coords(),
            },
            self.centroid(),
            None,
        )
        .ok()
    }

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        crate::sphericalpoint::MultiSphericalPoint::try_from_lonlats(
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

    fn boundary(&self) -> Option<crate::arcstring::ArcString> {
        let mut boundary = crate::arcstring::ArcString {
            points: self.coords(),
        };
        boundary.close();
        Some(boundary)
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        self.centroid()
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

    fn coords(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        (&self).coords()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn boundary(&self) -> Option<crate::arcstring::ArcString> {
        (&self).boundary()
    }

    fn representative_point(&self) -> crate::sphericalpoint::SphericalPoint {
        (&self).representative_point()
    }
}

impl GeometricOperations<&crate::sphericalpoint::SphericalPoint> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpoint::SphericalPoint) -> f64 {
        match self.convex_hull() {
            Some(hull) => hull.distance(other),
            None => self.coords().distance(other),
        }
    }

    fn contains(self, other: &crate::sphericalpoint::SphericalPoint) -> bool {
        let point = other.to_lonlat(self.degrees);

        self.contains_lonlat(point[0], point[1])
    }

    fn within(self, _: &crate::sphericalpoint::SphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &crate::sphericalpoint::SphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpoint::SphericalPoint) -> bool {
        self.contains(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpoint::SphericalPoint,
    ) -> Option<crate::sphericalpoint::SphericalPoint> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpoint::SphericalPoint) -> bool {
        let lonlat = other.to_lonlat(self.degrees);
        let tolerance = 3e-11;
        ((lonlat[0] - self.min_x).abs() < tolerance || (lonlat[0] - self.max_x).abs() < tolerance)
            && ((lonlat[1] - self.min_y).abs() < tolerance
                || (lonlat[1] - self.max_y).abs() < tolerance)
    }
}

impl GeometricOperations<&crate::sphericalpoint::MultiSphericalPoint> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpoint::MultiSphericalPoint) -> f64 {
        self.boundary().unwrap().distance(other)
    }

    fn contains(self, other: &crate::sphericalpoint::MultiSphericalPoint) -> bool {
        other
            .to_lonlats(self.degrees)
            .rows()
            .into_iter()
            .all(|lonlat| {
                lonlat[0] > self.min_x
                    && lonlat[1] > self.min_y
                    && lonlat[0] < self.max_x
                    && lonlat[1] < self.max_y
            })
    }

    fn within(self, _: &crate::sphericalpoint::MultiSphericalPoint) -> bool {
        false
    }

    fn crosses(self, _: &crate::sphericalpoint::MultiSphericalPoint) -> bool {
        false
    }

    fn intersects(self, other: &crate::sphericalpoint::MultiSphericalPoint) -> bool {
        other
            .to_lonlats(self.degrees)
            .rows()
            .into_iter()
            .any(|lonlat| {
                lonlat[0] > self.min_x
                    && lonlat[1] > self.min_y
                    && lonlat[0] < self.max_x
                    && lonlat[1] < self.max_y
            })
    }

    fn intersection(
        self,
        other: &crate::sphericalpoint::MultiSphericalPoint,
    ) -> Option<crate::sphericalpoint::MultiSphericalPoint> {
        let lonlats: Vec<Array1<f64>> = other
            .to_lonlats(self.degrees)
            .rows()
            .into_iter()
            .filter_map(|lonlat| {
                if lonlat[0] > self.min_x
                    && lonlat[1] > self.min_y
                    && lonlat[0] < self.max_x
                    && lonlat[1] < self.max_y
                {
                    Some(lonlat.to_owned())
                } else {
                    None
                }
            })
            .collect();

        let mut array = Array2::<f64>::uninit((lonlats.len(), 2));
        for (index, row) in array.axis_iter_mut(Axis(0)).enumerate() {
            lonlats[index].assign_to(row);
        }

        if !lonlats.is_empty() {
            Some(
                crate::sphericalpoint::MultiSphericalPoint::try_from_lonlats(
                    &unsafe { array.assume_init() }.view(),
                    self.degrees,
                )
                .unwrap(),
            )
        } else {
            None
        }
    }

    fn touches(self, other: &crate::sphericalpoint::MultiSphericalPoint) -> bool {
        self.boundary().unwrap().touches(other)
    }
}

impl GeometricOperations<&crate::arcstring::ArcString> for &AngularBounds {
    fn distance(self, other: &crate::arcstring::ArcString) -> f64 {
        self.boundary().unwrap().distance(other)
    }

    fn contains(self, other: &crate::arcstring::ArcString) -> bool {
        other.within(self)
    }

    fn within(self, _: &crate::arcstring::ArcString) -> bool {
        false
    }

    fn crosses(self, other: &crate::arcstring::ArcString) -> bool {
        other.crosses(self)
    }

    fn intersects(self, other: &crate::arcstring::ArcString) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::arcstring::ArcString,
    ) -> Option<crate::arcstring::MultiArcString> {
        // TODO: this could probably just be a clip by bounds
        if let Some(convex_hull) = self.convex_hull() {
            convex_hull.intersection(other)
        } else {
            None
        }
    }

    fn touches(self, other: &crate::arcstring::ArcString) -> bool {
        other.touches(self)
    }
}

impl GeometricOperations<&crate::arcstring::MultiArcString> for &AngularBounds {
    fn distance(self, other: &crate::arcstring::MultiArcString) -> f64 {
        self.boundary().unwrap().distance(other)
    }

    fn contains(self, other: &crate::arcstring::MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, _: &crate::arcstring::MultiArcString) -> bool {
        false
    }

    fn crosses(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.boundary().unwrap().crosses(other)
    }

    fn intersects(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::arcstring::MultiArcString,
    ) -> Option<crate::arcstring::MultiArcString> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::arcstring::MultiArcString) -> bool {
        self.boundary().unwrap().touches(other)
    }
}

impl GeometricOperations<&AngularBounds> for &AngularBounds {
    fn distance(self, other: &AngularBounds) -> f64 {
        self.coords().distance(&other.coords())
    }

    fn contains(self, other: &AngularBounds) -> bool {
        other.within(self)
    }

    fn within(self, other: &AngularBounds) -> bool {
        let other = if self.degrees == other.degrees {
            other
        } else if other.degrees {
            &other.to_radians()
        } else {
            &other.to_degrees()
        };

        other.min_x > self.min_x
            && other.min_y > self.min_y
            && other.max_x < self.max_x
            && other.max_y < self.max_y
    }

    fn crosses(self, other: &AngularBounds) -> bool {
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

    fn intersects(self, other: &AngularBounds) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(self, other: &AngularBounds) -> Option<AngularBounds> {
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

            Some(AngularBounds {
                min_x,
                min_y,
                max_x,
                max_y,
                degrees: self.degrees,
            })
        } else {
            None
        }
    }

    fn touches(self, other: &AngularBounds) -> bool {
        self.boundary().unwrap().touches(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::SphericalPolygon> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpolygon::SphericalPolygon) -> f64 {
        self.boundary().unwrap().distance(other)
    }

    fn contains(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.contains(&other.exterior.points)
    }

    fn within(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.coords().within(other)
    }

    fn crosses(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.convex_hull()
            .map_or(false, |convex_hull| convex_hull.crosses(other))
    }

    fn intersects(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::SphericalPolygon,
    ) -> Option<crate::sphericalpolygon::MultiSphericalPolygon> {
        if let Some(convex_hull) = self.convex_hull() {
            convex_hull.intersection(other)
        } else {
            None
        }
    }

    fn touches(self, other: &crate::sphericalpolygon::SphericalPolygon) -> bool {
        self.boundary().unwrap().touches(other)
    }
}

impl GeometricOperations<&crate::sphericalpolygon::MultiSphericalPolygon> for &AngularBounds {
    fn distance(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> f64 {
        self.boundary().unwrap().distance(other)
    }

    fn contains(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.within(self)
    }

    fn within(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn crosses(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.crosses(self)
    }

    fn intersects(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        self.touches(other) || self.crosses(other)
    }

    fn intersection(
        self,
        other: &crate::sphericalpolygon::MultiSphericalPolygon,
    ) -> Option<crate::sphericalpolygon::MultiSphericalPolygon> {
        other.intersection(self)
    }

    fn touches(self, other: &crate::sphericalpolygon::MultiSphericalPolygon) -> bool {
        other.touches(self)
    }
}
