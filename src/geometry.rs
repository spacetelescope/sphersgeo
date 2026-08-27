use std::fmt::Display;

#[cfg(feature = "py")]
use pyo3::prelude::*;

pub trait Geometry {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint;

    /// lower dimension geometry that bounds this geometry's interior
    ///
    /// The boundary of a polygon is a closed arcstring,
    /// the boundary of an arcstring is two endpoints (unless closed),
    /// and the boundary of a point (and a closed arcstring) is null.
    fn boundary(&self) -> Option<impl Geometry>;

    /// point guaranteed to be within this geometry
    fn representative(&self) -> crate::sphericalpoint::SphericalPoint;

    /// mean position of all possible points within this geometry
    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint;

    /// smallest convex polygon containing this geometry
    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.vertices().convex_hull()
    }

    /// surface area of this geometry in square degrees
    fn area(&self) -> f64;

    /// angular length of this geometry in degrees
    fn length(&self) -> f64;

    /// well-known text representation of this geometry
    fn to_wkt(&self, angular: bool) -> String;
}

pub trait MultiGeometry<G: Geometry> {
    /// number of geometries in this collection
    fn len(&self) -> usize;

    /// append the geometry to this collection
    fn push(&mut self, other: G);

    /// extend this collection with geometries from the other collection
    fn extend(&mut self, other: Self);
}

pub trait MultiGeometryUnaryOperations<S: Geometry> {
    /// dissolved union of these geometries
    ///
    /// For further explanation of Unary Union see Shapely's `unary_union`.
    fn unary_union(&self) -> impl MultiGeometry<S>;

    /// overlapping regions between these geometries, if any
    ///
    /// For further explanation of Intersection see Shapely's `object.intersection`.
    fn unary_intersection(&self) -> Option<impl MultiGeometry<S>>;

    /// non-overlapping regions between these geometries
    ///
    /// For further explanation of Symmetric Difference see Shapely's `object.symmetric_difference`.
    fn unary_symmetric_difference(&self) -> Option<impl MultiGeometry<S>>;
}

/// Relationships between geometries of arbitrary types.
///
/// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/
pub trait GeometricRelationships<O: Geometry = Self> {
    /// shortest geodesic from this geometry to another
    fn distance(&self, other: &O) -> f64;

    /// Whether this and the other geometry's interiors are identical and the geometry types are the same.
    ///
    /// For further explanation of Equals see `ArcGIS Equals <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#equals>`_
    /// or Shapely's `object.equals`.
    fn equals(&self, other: &O) -> bool {
        false
    }

    /// Whether the other geometry is a subset of this geometry
    /// (every point of the other geometry is a point on the interior OR boundary of this geometry).
    fn covers(&self, other: &O) -> bool {
        false
    }

    /// Whether this geometry covers the other geometry AND the interiors share at least one point.
    ///
    /// Contains is the inverse of Within.
    ///
    /// For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
    /// or Shapely's `object.contains`.
    fn contains(&self, other: &O) -> bool {
        false
    }

    /// Whether the other geometry covers this geometry AND the interiors share at least one point.
    ///
    /// Within is the inverse of Contains.
    ///
    /// For further explanation of Contains see `ArcGIS Contains <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains>`_
    /// or Shapely's `object.contains`.
    fn within(&self, other: &O) -> bool {
        false
    }

    /// Whether this arcstring / polygon and the other arcstring / polygon share only SOME (not all) interior points, but do NOT overlap.
    ///
    /// Two arcstrings cross if they meet at point(s) only, and at least one of the shared points is internal to both arcstrings.
    /// An arcstring and polygon cross if they share an arcstring on the interior of the polygon, which is NOT equal to the entire arcstring.
    ///
    /// For further explanation of Crosses see `ArcGIS Crosses <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#crosses>`_
    /// or Shapely's `object.crosses`.
    fn crosses(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share any vertices but do not overlap.
    ///
    /// For further explanation of Touches see `ArcGIS Touches <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#touches>`_
    /// or Shapely's `object.touches`.
    fn touches(&self, other: &O) -> bool;

    /// whether any region of this geometry overlaps the other geometry
    ///
    /// This and the other geometry must be of the same geometry type,
    /// AND their intersection also of the same geometry type
    /// BUT not equal to either (in which case this would be `within` or `contains`).
    ///
    /// For further explanation of Overlaps see `ArcGIS Overlaps <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#overlaps>`_
    /// or Shapely's `object.overlaps`.
    fn overlaps(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share ANY point(s).
    /// If this geometries contains, is within, crosses, touches, or overlaps the other geometry, they intersect.
    ///
    /// For further explanation of Intersects see `ArcGIS Intersects <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#intersects>`_
    /// or Shapely's `object.intersects`.
    fn intersects(&self, other: &O) -> bool;

    /// Whether this and the other geometry do NOT share ANY point(s).
    ///
    /// Disjoint is the inverse of Intersects.
    ///
    /// For further explanation of Disjoint see `ArcGIS Disjoint <https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#disjoint>`_
    /// or Shapely's `object.disjoint`.
    fn disjoint(&self, other: &O) -> bool {
        !self.intersects(other)
    }
}

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

/// operations between geometries of the same order (point, arcstring, polygon)
pub trait GeometricOperations<O: Geometry = Self, S: Geometry = Self> {
    /// regions of this geometry that overlap the other geometry
    ///
    /// Intersection is the inverse of Difference.
    ///
    /// For further explanation of Intersection see Shapely's `object.intersection`.
    fn intersection(&self, other: &O) -> GeometryCollection;

    /// regions of this geometry that do not intersect or overlap with the other geometry
    ///
    /// Difference is the inverse of Intersection.
    ///
    /// For further explanation of Difference see Shapely's `object.difference`.
    fn difference(&self, other: &O) -> Option<impl MultiGeometry<S>>;

    /// dissolved union of this geometry and the other geometry
    ///
    /// For further explanation of Union see Shapely's `object.union`.
    fn union(&self, other: &O) -> GeometryCollection;
}

/// define angular separation between 3D vectors
pub struct AngularSeparation {}

impl kiddo::traits::DistanceMetric<f64, 3> for AngularSeparation {
    #[inline]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        // radians subtended
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).acos()
    }

    #[inline]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}

#[cfg_attr(feature = "py", derive(FromPyObject, IntoPyObject))]
#[derive(Debug, Clone, PartialEq)]
pub enum AnyGeometry {
    #[cfg_attr(feature = "py", pyo3(transparent))]
    SphericalPoint(crate::sphericalpoint::SphericalPoint),
    #[cfg_attr(feature = "py", pyo3(transparent))]
    MultiSphericalPoint(crate::sphericalpoint::MultiSphericalPoint),
    #[cfg_attr(feature = "py", pyo3(transparent))]
    ArcString(crate::arcstring::ArcString),
    #[cfg_attr(feature = "py", pyo3(transparent))]
    MultiArcString(crate::arcstring::MultiArcString),
    #[cfg_attr(feature = "py", pyo3(transparent))]
    SphericalPolygon(crate::sphericalpolygon::SphericalPolygon),
    #[cfg_attr(feature = "py", pyo3(transparent))]
    MultiSphericalPolygon(crate::sphericalpolygon::MultiSphericalPolygon),
}

fn try_point_from_wkt_fragment(wkt_fragment: &str) -> Result<Vec<f64>, String> {
    let mut point = vec![];
    for coordinate in wkt_fragment.split_whitespace() {
        point.push(
            coordinate
                .parse::<f64>()
                .map_err(|err| format!("{err} `{coordinate}`"))?,
        );
    }
    Ok(point)
}

fn try_points_from_wkt_fragment(wkt_fragment: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut points = vec![];
    for point_fragment in wkt_fragment.split(", ") {
        points.push(try_point_from_wkt_fragment(point_fragment)?);
    }
    Ok(points)
}

fn try_multipoints_from_wkt_fragment(wkt_fragment: &str) -> Result<Vec<Vec<Vec<f64>>>, String> {
    let mut multipoints = vec![];
    for multipoint_fragment in wkt_fragment.split("), (") {
        multipoints.push(try_points_from_wkt_fragment(
            &multipoint_fragment.trim_matches(|c| c == '(' || c == ')'),
        )?);
    }
    Ok(multipoints)
}

/// construct geometry from well-known text representation
pub fn try_from_wkt(wkt: &str) -> Result<AnyGeometry, String> {
    let wkt = wkt.trim(); // trim whitespace
    if wkt.starts_with("POINT (") {
        crate::sphericalpoint::SphericalPoint::try_from(&try_point_from_wkt_fragment(
            &wkt[7..wkt.len() - 1],
        )?)
        .map(|point| AnyGeometry::SphericalPoint(point))
    } else if wkt.starts_with("MULTIPOINT (") || wkt.starts_with("LINESTRING (") {
        let points = crate::sphericalpoint::MultiSphericalPoint::try_from(
            &try_points_from_wkt_fragment(&wkt[12..wkt.len() - 1])?,
        )?;

        if wkt.starts_with("MULTIPOINT (") {
            Ok(AnyGeometry::MultiSphericalPoint(points))
        } else {
            crate::arcstring::ArcString::try_from(points)
                .map(|arcstring| AnyGeometry::ArcString(arcstring))
        }
    } else if wkt.starts_with("MULTILINESTRING ((") || wkt.starts_with("POLYGON ((") {
        let mut linestrings = vec![];
        for multipoint in try_multipoints_from_wkt_fragment(
            &wkt[if wkt.starts_with("MULTILINESTRING ((") {
                18
            } else {
                10
            }..wkt.len() - 2],
        )? {
            linestrings.push(crate::arcstring::ArcString::try_from(
                crate::sphericalpoint::MultiSphericalPoint::try_from(&multipoint)?,
            )?);
        }

        if wkt.starts_with("MULTILINESTRING ((") {
            crate::arcstring::MultiArcString::try_from(linestrings)
                .map(|multiarcstring| AnyGeometry::MultiArcString(multiarcstring))
        } else {
            if linestrings.len() == 1 {
                crate::sphericalpolygon::SphericalPolygon::try_from(linestrings[0].to_owned())
                    .map(|polygon| AnyGeometry::SphericalPolygon(polygon))
            } else {
                Err(String::from(
                    "multiple linestrings provided in WKT for a single polygon; `sphersgeo` does not currently support holes",
                ))
            }
        }
    } else if wkt.starts_with("MULTIPOLYGON (((") {
        let mut polygons = vec![];
        for polygon_fragment in wkt[16..wkt.len() - 3].split(")), ((") {
            polygons.push(
                match try_from_wkt(format!("POLYGON (({polygon_fragment}))").as_str())? {
                    AnyGeometry::SphericalPolygon(polygon) => polygon,
                    _ => {
                        return Err(String::from("invalid WKT"));
                    }
                },
            );
        }

        crate::sphericalpolygon::MultiSphericalPolygon::try_from(polygons)
            .map(|multipolygon| AnyGeometry::MultiSphericalPolygon(multipolygon))
    } else if wkt.starts_with("GEOMETRYCOLLECTION (") {
        Err(String::from("GEOMETRYCOLLECTION not implemented"))
    } else {
        Err(format!("unknown well-known text: {wkt}"))
    }
}
