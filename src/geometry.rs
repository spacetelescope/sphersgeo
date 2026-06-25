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

/// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/
pub trait GeometricRelationships<O: Geometry = Self> {
    /// Whether this and the other geometry's interiors are identical and the geometry types are the same.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#equals
    fn equals(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share ANY point(s).
    /// If this geometries contains, is within, crosses, touches, or overlaps the other geometry, they intersect.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#intersects
    fn intersects(&self, other: &O) -> bool;

    /// Whether the other geometry is a subset of this geometry
    /// (every point of the other geometry is a point on the interior OR boundary of this geometry).
    fn covers(&self, other: &O) -> bool {
        false
    }

    /// Whether this geometry covers the other geometry AND the interiors share at least one point.
    ///
    /// Contains is the inverse of Within.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains
    fn contains(&self, other: &O) -> bool {
        false
    }

    /// Whether the other geometry covers this geometry AND the interiors share at least one point.
    ///
    /// Within is the inverse of Contains.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#contains
    fn within(&self, other: &O) -> bool {
        false
    }

    /// Whether this arcstring / polygon and the other arcstring / polygon share only SOME (not all) interior points, but do NOT overlap.
    ///
    /// Two arcstrings cross if they meet at point(s) only, and at least one of the shared points is internal to both arcstrings.
    /// An arcstring and polygon cross if they share an arcstring on the interior of the polygon, which is NOT equal to the entire arcstring.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#crosses
    fn crosses(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry share any vertices but do not overlap.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#touches
    fn touches(&self, other: &O) -> bool;

    /// Whether this and the other geometry are of the same geometry type,
    /// AND their intersection is also of the same geometry type BUT is not equal to either.
    ///
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#overlaps
    fn overlaps(&self, other: &O) -> bool {
        false
    }

    /// Whether this and the other geometry do NOT share ANY point(s).
    ///
    /// Disjoint is the inverse of Intersects.
    /// https://developers.arcgis.com/geoanalytics/core-concepts/spatial-relationships/#disjoint
    fn disjoint(&self, other: &O) -> bool {
        !self.intersects(other)
    }
}

pub trait GeometricOperations<O: Geometry = Self, S: Geometry = Self> {
    fn union(&self, other: &O) -> Option<impl MultiGeometry<S>>;

    /// shortest great-circle distance over the sphere from any part of this geometry to another
    fn distance(&self, other: &O) -> f64;

    /// any part of this geometry that is within another
    ///
    /// NOTE: this function is NOT rigorous;
    /// it will ONLY return the lower order of geometry being compared
    /// and will NOT handle touching, colinear overlap, or degenerate cases
    fn intersection(&self, other: &O) -> Option<impl Geometry>;

    /// split this geometry into a multi-geometry, at the crossing with the given geometry
    fn symmetric_difference(&self, other: &O) -> impl MultiGeometry<S>;
}

pub trait GeometryCollection<G: Geometry, M: MultiGeometry<G> = Self> {
    /// join geometries into one
    fn join_self(&self) -> M;

    /// find overlapping regions between geometries, if any
    fn overlap_self(&self) -> Option<M>;

    /// only return non-overlapping regions between geometries
    fn symmetric_difference_self(&self) -> Option<M>;
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
        point.push(coordinate.parse::<f64>().map_err(|err| format!("{err}"))?);
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
        multipoints.push(try_points_from_wkt_fragment(multipoint_fragment)?);
    }
    Ok(multipoints)
}

/// construct geometry from well-known text representation
pub fn try_from_wkt(wkt: &str) -> Result<AnyGeometry, String> {
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
    } else {
        Err(String::from("unknown well-known text"))
    }
}
