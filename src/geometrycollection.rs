use std::fmt::Display;

use crate::geometry::Geometry;

#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg_attr(feature = "py", pyclass(from_py_object, str))]
#[derive(Clone, Debug)]
pub struct GeometryCollection {
    pub points: Option<crate::sphericalpoint::MultiSphericalPoint>,
    pub strings: Option<crate::arcstring::MultiArcString>,
    pub polygons: Option<crate::sphericalpolygon::MultiSphericalPolygon>,
}

impl Geometry for GeometryCollection {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        let mut xyzs = vec![];
        if let Some(points) = &self.points {
            xyzs.extend(points.vertices().xyzs);
        }
        if let Some(strings) = &self.strings {
            xyzs.extend(strings.vertices().xyzs);
        }
        if let Some(polygons) = &self.polygons {
            xyzs.extend(polygons.vertices().xyzs);
        }

        crate::sphericalpoint::MultiSphericalPoint::try_from(xyzs)
            .expect("no vertices in collection")
    }

    fn boundary(&self) -> Option<Self> {
        Some(Self {
            points: if let Some(strings) = &self.strings {
                strings.boundary()
            } else {
                None
            },
            strings: if let Some(polygons) = &self.polygons {
                polygons.boundary()
            } else {
                None
            },
            polygons: None,
        })
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        if let Some(points) = &self.points {
            return points.representative();
        }
        if let Some(strings) = &self.strings {
            return strings.representative();
        }
        if let Some(polygons) = &self.polygons {
            return polygons.representative();
        }
        panic!("empty collection");
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        todo!()
    }

    fn area(&self) -> f64 {
        return if let Some(polygons) = &self.polygons {
            polygons.area()
        } else {
            0.0
        };
    }

    fn length(&self) -> f64 {
        let strings_length = if let Some(strings) = &self.strings {
            strings.length()
        } else {
            0.0
        };

        let polygons_length = if let Some(polygons) = &self.polygons {
            polygons.length()
        } else {
            0.0
        };

        return if strings_length > polygons_length {
            strings_length
        } else {
            polygons_length
        };
    }

    fn to_wkt(&self, angular: bool) -> String {
        let mut wkts = vec![];
        if let Some(points) = &self.points {
            wkts.push(points.to_wkt(angular));
        }
        if let Some(strings) = &self.strings {
            wkts.push(strings.to_wkt(angular));
        }
        if let Some(polygons) = &self.polygons {
            wkts.push(polygons.to_wkt(angular));
        }
        format!("GEOMETRYCOLLECTION ({})", wkts.join(", "))
    }
}

impl Display for GeometryCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GeometryCollection({:?}, {:?}, {:?})",
            self.points, self.strings, self.polygons
        )
    }
}
