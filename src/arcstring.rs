use crate::{
    angularbounds::AngularBounds,
    geometry::{
        AnyGeometry, ExtendMultiGeometry, GeometricOperations, Geometry, MultiGeometry,
        MultiGeometryIntoIterator, MultiGeometryIterator,
    },
    geometrycollection::GeometryCollection,
    sphericalpolygon::{spherical_triangle_area, MultiSphericalPolygon, SphericalPolygon},
    vectorpoint::{cross_vectors, MultiVectorPoint, VectorPoint},
};
use kiddo::ImmutableKdTree;
use numpy::ndarray::{concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use pyo3::prelude::*;
use std::collections::VecDeque;

pub fn arc_interpolate_points(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    n: usize,
) -> Result<Array2<f64>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::linspace(0.0, 1.0, n);
    let t = t.to_shape((n, 1)).unwrap();
    let omega = arc_length_from_vectors(a, b);

    if a.len() == b.len() {
        if a.len() == 3 && b.len() == 3 {
            let offsets = if omega == 0.0 {
                t.to_owned()
            } else {
                (t * omega).sin() / omega.sin()
            };
            let mut inverted_offsets = offsets.to_owned();
            inverted_offsets.invert_axis(Axis(0));

            Ok(concatenate(
                Axis(0),
                &[
                    (inverted_offsets * a + offsets * b).view(),
                    b.to_shape((1, 3)).unwrap().view(),
                ],
            )
            .unwrap())
        } else if a.len() == 2 && b.len() == 2 {
            Ok(concatenate(
                Axis(0),
                &[
                    (a * ((Zip::from(&t).par_map_collect(|t| 1.0 - t) * omega).sin()
                        / omega.sin())
                        + b * &((t * omega).sin() / omega.sin()).view())
                        .view(),
                    b.to_shape((1, 2)).unwrap().view(),
                ],
            )
            .unwrap())
        } else {
            Err(String::from(""))
        }
    } else {
        Err(String::from("shape must match"))
    }
}

/// given points A, B, and C on the unit sphere, retrieve the angle at B between arc AB and arc BC
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn arc_angle(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
    degrees: bool,
) -> f64 {
    let tolerance = 3e-11;

    let ab = arc_length_from_vectors(a, b);
    let bc = arc_length_from_vectors(b, c);
    let ca = arc_length_from_vectors(c, a);

    let angle = if ab > tolerance && bc > tolerance {
        (ca.cos() - bc.cos() * ab.cos()) / (bc.sin() * ab.sin()).acos()
    } else {
        (1.0 - ca.powi(2) / 2.0).acos()
    };

    if degrees {
        angle.to_degrees()
    } else {
        angle
    }
}

/// given points A, B, and C on the unit sphere, retrieve the angle at B between arc AB and arc BC
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn arc_angles(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    c: &ArrayView2<f64>,
    degrees: bool,
) -> Array1<f64> {
    let abx = cross_vectors(a, b);
    let bcx = cross_vectors(b, c);
    let x = cross_vectors(&abx.view(), &bcx.view());

    let diff = (b * x).sum_axis(Axis(1));
    let mut inner = (abx * bcx).sum_axis(Axis(1));
    inner.par_mapv_inplace(|v| v.acos());

    let angles = stack(Axis(0), &[inner.view(), diff.view()])
        .unwrap()
        .map_axis(Axis(0), |v| {
            if v[1] < 0.0 {
                (2.0 * std::f64::consts::PI) - v[0]
            } else {
                v[0]
            }
        });

    if degrees {
        angles.to_degrees()
    } else {
        angles
    }
}

/// radians subtended by this arc on the sphere
///    Notes
///    -----
///    The length is computed using the following:
///
///       l = arccos(A ⋅ B) / r^2
pub fn arc_length_from_vectors(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.dot(b).acos()
}

/// whether the three points exist on the same line
pub fn collinear(a: &ArrayView1<f64>, b: &ArrayView1<f64>, c: &ArrayView1<f64>) -> bool {
    let tolerance = 3e-11;
    spherical_triangle_area(a, b, c) < tolerance

    // let left = arc_length(&a, &p);
    // let right = arc_length(&p, &b);
    // let total = arc_length(&a, &b);

    // let tolerance = 3e-11;
    // if left + right - total < tolerance {
    //     // ensure angle is flat
    //     if angle(&a, &point.xyz.view(), &b, false) - std::f64::consts::PI < tolerance {
    //         return true;
    //     }
    // }
}

/// series of great circle arcs along the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct ArcString {
    pub points: MultiVectorPoint,
}

impl TryFrom<MultiVectorPoint> for ArcString {
    type Error = String;

    fn try_from(points: MultiVectorPoint) -> Result<Self, Self::Error> {
        let arcstring = Self { points };
        if arcstring.intersects(&arcstring) {
            Err(String::from("arcstring intersects itself"))
        } else {
            Ok(arcstring)
        }
    }
}

impl Into<MultiVectorPoint> for ArcString {
    fn into(self) -> MultiVectorPoint {
        self.points
    }
}

impl Into<Vec<ArcString>> for &ArcString {
    fn into(self) -> Vec<ArcString> {
        let vectors = &self.points.xyz;
        let mut arcs = vec![];
        for index in 0..vectors.nrows() - 1 {
            arcs.push(ArcString {
                points: MultiVectorPoint::try_from(
                    vectors.slice(s![index..index + 1, ..]).to_owned(),
                )
                .unwrap(),
            })
        }

        arcs
    }
}

impl ArcString {
    pub fn midpoints(&self) -> MultiVectorPoint {
        MultiVectorPoint::try_from(
            (&self.points.xyz.slice(s![..-1, ..]) + &self.points.xyz.slice(s![1.., ..]) / 2.0)
                .to_owned(),
        )
        .unwrap()
    }

    pub fn lengths(&self) -> Array1<f64> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| arc_length_from_vectors(&a, &b))
    }
}

impl ToString for ArcString {
    fn to_string(&self) -> String {
        format!("ArcString({0})", self.points.to_string())
    }
}

impl PartialEq for ArcString {
    fn eq(&self, other: &ArcString) -> bool {
        &self == &other
    }
}

impl PartialEq<&ArcString> for ArcString {
    fn eq(&self, other: &&ArcString) -> bool {
        &self.points == &other.points
    }
}

impl Geometry for &ArcString {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.lengths().sum()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self.points).convex_hull()
    }

    fn points(&self) -> MultiVectorPoint {
        self.points.to_owned()
    }
}

impl Geometry for ArcString {
    fn area(&self) -> f64 {
        (&self).area()
    }

    fn length(&self) -> f64 {
        (&self).length()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        (&self).convex_hull()
    }

    fn points(&self) -> MultiVectorPoint {
        (&self).points()
    }
}

impl GeometricOperations<&VectorPoint> for &ArcString {
    fn distance(self, other: &VectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, point: &VectorPoint) -> bool {
        // check if point is one of the vertices of this linestring
        if (&self.points).contains(point) {
            return true;
        }

        // check if point is within the bounding box
        if self.bounds(false).contains(point) {
            // compare lengths to endpoints with the arc length
            for index in 0..self.points.xyz.nrows() - 1 {
                let a = self.points.xyz.slice(s![index, ..]);
                let b = self.points.xyz.slice(s![index + 1, ..]);
                let p = point.xyz.view();

                if collinear(&a.view(), &p.view(), &b.view()) {
                    return true;
                }
            }
        }

        return false;
    }

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.intersection(other).len() > 0
    }

    fn intersection(self, other: &VectorPoint) -> GeometryCollection {
        if self.contains(other) {
            GeometryCollection {
                geometries: vec![AnyGeometry::VectorPoint(other.to_owned())],
            }
        } else {
            GeometryCollection::empty()
        }
    }
}

impl GeometricOperations<&MultiVectorPoint> for &ArcString {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        todo!()
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn intersection(self, other: &MultiVectorPoint) -> GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&ArcString> for &ArcString {
    fn distance(self, other: &ArcString) -> f64 {
        todo!();
    }

    fn contains(self, other: &ArcString) -> bool {
        todo!();
    }

    fn within(self, other: &ArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &ArcString) -> bool {
        // TODO: write an intersects algorithm
        self.intersection(other).len() > 0
    }

    /// Returns the point of intersection between two great circle arcs.
    ///
    /// Notes
    /// -----
    /// The basic intersection is computed using linear algebra as follows
    /// [1]_:
    ///
    /// .. math::
    ///
    ///     T = \lVert(A × B) × (C × D)\rVert
    ///
    /// To determine the correct sign (i.e. hemisphere) of the
    /// intersection, the following four values are computed:
    ///
    /// .. math::
    ///
    ///     s_1 = ((A × B) × A) \cdot T
    ///
    ///     s_2 = (B × (A × B)) \cdot T
    ///
    ///     s_3 = ((C × D) × C) \cdot T
    ///
    ///     s_4 = (D × (C × D)) \cdot T
    ///
    /// For :math:`s_n`, if all positive :math:`T` is returned as-is.  If
    /// all negative, :math:`T` is multiplied by :math:`-1`.  Otherwise
    /// the intersection does not exist and is undefined.
    ///
    /// References
    /// ----------
    ///
    /// .. [1] Method explained in an `e-mail
    ///     <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_
    ///     by Roger Stafford.
    ///
    /// https://spherical-geometry.readthedocs.io/en/latest/api/spherical_geometry.great_circle_arc.intersection.html#rb82e4e1c8654-1
    fn intersection(self, other: &ArcString) -> GeometryCollection {
        todo!();
    }
}

impl GeometricOperations<&MultiArcString> for &ArcString {
    fn distance(self, other: &MultiArcString) -> f64 {
        other.distance(self)
    }

    fn contains(self, other: &MultiArcString) -> bool {
        other.within(self)
    }

    fn within(self, other: &MultiArcString) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiArcString) -> crate::geometrycollection::GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&AngularBounds> for &ArcString {
    fn distance(self, other: &AngularBounds) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &AngularBounds) -> crate::geometrycollection::GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&SphericalPolygon> for &ArcString {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &SphericalPolygon) -> GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &ArcString {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        other.distance(self)
    }

    fn contains(self, _: &MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        other.contains(self)
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        other.intersects(self)
    }

    fn intersection(self, other: &MultiSphericalPolygon) -> GeometryCollection {
        other.intersection(self)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct MultiArcString {
    pub arcstrings: VecDeque<ArcString>,
    pub kdtree: ImmutableKdTree<f64, 3>,
}

impl TryFrom<Vec<MultiVectorPoint>> for MultiArcString {
    type Error = String;

    fn try_from(points: Vec<MultiVectorPoint>) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl Into<Vec<MultiVectorPoint>> for MultiArcString {
    fn into(self) -> Vec<MultiVectorPoint> {
        todo!()
    }
}

impl Into<Vec<ArcString>> for MultiArcString {
    fn into(self) -> Vec<ArcString> {
        self.arcstrings.into()
    }
}

impl MultiArcString {
    pub fn midpoints(&self) -> MultiVectorPoint {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.midpoints())
            .sum()
    }

    pub fn lengths(&self) -> Array1<f64> {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.length())
            .collect()
    }
}

impl ToString for MultiArcString {
    fn to_string(&self) -> String {
        format!("MultiArcString({})", self.arcstrings.len())
    }
}

impl PartialEq for MultiArcString {
    fn eq(&self, other: &MultiArcString) -> bool {
        &self == &other
    }
}

impl PartialEq<&MultiArcString> for MultiArcString {
    fn eq(&self, other: &&MultiArcString) -> bool {
        &self.arcstrings == &other.arcstrings
    }
}

impl Geometry for &MultiArcString {
    fn area(&self) -> f64 {
        0.
    }

    fn length(&self) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.length())
            .sum()
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        todo!()
    }

    fn points(&self) -> crate::vectorpoint::MultiVectorPoint {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.to_owned().points)
            .sum()
    }
}

impl Geometry for MultiArcString {
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

impl MultiGeometry for &MultiArcString {
    fn len(&self) -> usize {
        self.arcstrings.len()
    }
}

impl MultiGeometry for MultiArcString {
    fn len(&self) -> usize {
        (&self).len()
    }
}

impl ExtendMultiGeometry<ArcString> for MultiArcString {
    fn extend(&mut self, other: Self) {
        self.arcstrings.extend(other.arcstrings);
    }

    fn push(&mut self, other: ArcString) {
        self.arcstrings.push_back(other);
    }
}

impl GeometricOperations<&VectorPoint> for &MultiArcString {
    fn distance(self, other: &VectorPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &VectorPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, _: &VectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &VectorPoint) -> bool {
        self.contains(other)
    }

    fn intersection(self, other: &VectorPoint) -> crate::geometrycollection::GeometryCollection {
        other.intersection(self)
    }
}

impl GeometricOperations<&MultiVectorPoint> for &MultiArcString {
    fn distance(self, other: &MultiVectorPoint) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &MultiVectorPoint) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, _: &MultiVectorPoint) -> bool {
        false
    }

    fn intersects(self, other: &MultiVectorPoint) -> bool {
        todo!()
    }

    fn intersection(
        self,
        other: &MultiVectorPoint,
    ) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&ArcString> for &MultiArcString {
    fn distance(self, other: &ArcString) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, other: &ArcString) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &ArcString) -> bool {
        todo!()
    }

    fn intersection(self, other: &ArcString) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&MultiArcString> for &MultiArcString {
    fn distance(self, other: &MultiArcString) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.contains(other))
    }

    fn within(self, other: &MultiArcString) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &MultiArcString) -> bool {
        todo!()
    }

    fn intersection(self, other: &MultiArcString) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&AngularBounds> for &MultiArcString {
    fn distance(self, other: &AngularBounds) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &AngularBounds) -> bool {
        false
    }

    fn within(self, other: &AngularBounds) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &AngularBounds) -> bool {
        todo!()
    }

    fn intersection(self, other: &AngularBounds) -> crate::geometrycollection::GeometryCollection {
        todo!()
    }
}

impl GeometricOperations<&SphericalPolygon> for &MultiArcString {
    fn distance(self, other: &SphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &SphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &SphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        self,
        other: &SphericalPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }
}

impl GeometricOperations<&MultiSphericalPolygon> for &MultiArcString {
    fn distance(self, other: &MultiSphericalPolygon) -> f64 {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.distance(other))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn contains(self, _: &MultiSphericalPolygon) -> bool {
        false
    }

    fn within(self, other: &MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .all(|arcstring| arcstring.within(other))
    }

    fn intersects(self, other: &MultiSphericalPolygon) -> bool {
        self.arcstrings
            .iter()
            .any(|arcstring| arcstring.intersects(other))
    }

    fn intersection(
        self,
        other: &MultiSphericalPolygon,
    ) -> crate::geometrycollection::GeometryCollection {
        self.arcstrings
            .iter()
            .map(|arcstring| arcstring.intersection(other))
            .sum()
    }
}

impl<'a> Iterator for MultiGeometryIterator<'a, MultiArcString> {
    type Item = ArcString;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.multi.len() {
            Some(self.multi.arcstrings[self.index].to_owned())
        } else {
            None
        }
    }
}

impl MultiArcString {
    fn iter(&self) -> MultiGeometryIterator<MultiArcString> {
        MultiGeometryIterator::<MultiArcString> {
            multi: self,
            index: 0,
        }
    }
}

impl Iterator for MultiGeometryIntoIterator<MultiArcString> {
    type Item = ArcString;

    fn next(&mut self) -> Option<Self::Item> {
        self.multi.arcstrings.pop_front()
    }
}

impl IntoIterator for MultiArcString {
    type Item = ArcString;

    type IntoIter = MultiGeometryIntoIterator<MultiArcString>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            multi: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::MultiGeometry;
    use numpy::ndarray::{array, linspace, s};

    #[test]
    fn test_midpoint() {
        let tolerance = 1e-10;

        let mut avec = Array2::<f64>::zeros((0, 2));
        let mut bvec = Array2::<f64>::zeros((0, 2));
        for i in linspace(0., 11., 5) {
            for j in linspace(0., 11., 5) {
                let row = array![i, j];
                avec.push_row(row.view()).unwrap();
                bvec.push_row(row.view()).unwrap();
            }
        }
        avec += 7.0;
        bvec += 10.0;

        for a in avec.rows() {
            let a = VectorPoint::try_from_lonlat(&a, true).unwrap();
            for b in bvec.rows() {
                let b = VectorPoint::try_from_lonlat(&b, true).unwrap();
                let c = ArcString { points: &a + &b }.midpoints();
                let aclen = ArcString { points: &a + &c }.length();
                let bclen = ArcString { points: &b + &c }.length();
                assert!((aclen - bclen) < tolerance)
            }
        }
    }

    #[test]
    fn test_contains() {
        let arc = ArcString {
            points: MultiVectorPoint::try_from_lonlats(
                &array![[-30.0, -30.0], [30.0, 30.0]].view(),
                true,
            )
            .unwrap(),
        };
        assert!((&arc).contains(
            &VectorPoint::try_from_lonlat(&array![349.10660535, -12.30998866].view(), true)
                .unwrap()
        ));

        let vertical_arc = ArcString {
            points: MultiVectorPoint::try_from_lonlats(
                &array![[60.0, 0.0], [60.0, 30.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!((&vertical_arc)
                .contains(&VectorPoint::try_from_lonlat(&array![60.0, i].view(), true).unwrap()))
        }

        let horizontal_arc = ArcString {
            points: MultiVectorPoint::try_from_lonlats(
                &array![[0.0, 60.0], [30.0, 60.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!((&horizontal_arc)
                .contains(&VectorPoint::try_from_lonlat(&array![i, 60.0].view(), true).unwrap()));
        }
    }

    #[test]
    fn test_interpolate() {
        let tolerance = 1e-10;

        let a_lonlat = array![60.0, 0.0];
        let b_lonlat = array![60.0, 30.0];
        let lonlats = arc_interpolate_points(&a_lonlat.view(), &b_lonlat.view(), 10).unwrap();

        let a = VectorPoint::try_from_lonlat(&a_lonlat.view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&b_lonlat.view(), true).unwrap();

        assert!(Zip::from(&lonlats.slice(s![0, ..]))
            .and(&a_lonlat.view())
            .all(|test, reference| (test - reference).abs() < tolerance));
        assert!(Zip::from(&lonlats.slice(s![-1, ..]))
            .and(&b_lonlat.view())
            .all(|test, reference| (test - reference).abs() < tolerance));

        let xyzs = arc_interpolate_points(&a.xyz.view(), &b.xyz.view(), 10).unwrap();

        assert!(Zip::from(&xyzs.slice(s![0, ..]))
            .and(&a.xyz.view())
            .all(|test, reference| (test - reference).abs() < tolerance));
        assert!(Zip::from(&xyzs.slice(s![-1, ..]))
            .and(&b.xyz.view())
            .all(|test, reference| (test - reference).abs() < tolerance));

        let arc_from_lonlats = ArcString {
            points: MultiVectorPoint::try_from_lonlats(&lonlats.view(), true).unwrap(),
        };
        let arc_from_xyzs = ArcString {
            points: MultiVectorPoint::try_from(xyzs.to_owned()).unwrap(),
        };

        for xyz in xyzs.rows() {
            let point = VectorPoint::try_from(xyz.to_owned()).unwrap();
            assert!((&arc_from_lonlats).contains(&point));
            assert!((&arc_from_xyzs).contains(&point));
        }

        let distances_from_lonlats = arc_from_lonlats.lengths();
        let distances_from_xyz = arc_from_xyzs.lengths();

        assert!(Zip::from(&distances_from_lonlats)
            .and(&distances_from_xyz)
            .all(|from_lonlats, from_xyz| (from_lonlats - from_xyz).abs() < tolerance));
    }

    #[test]
    fn test_intersection() {
        let tolerance = 1e-10;

        let a = VectorPoint::try_from_lonlat(&array![-10.0, -10.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![10.0, 10.0].view(), true).unwrap();

        let c = VectorPoint::try_from_lonlat(&array![-25.0, 10.0].view(), true).unwrap();
        let d = VectorPoint::try_from_lonlat(&array![15.0, -10.0].view(), true).unwrap();

        // let e = VectorPoint::try_from_lonlat(&array![-20.0, 40.0].view(), true).unwrap();
        // let f = VectorPoint::try_from_lonlat(&array![20.0, 40.0].view(), true).unwrap();

        let reference_intersection = array![0.99912414, -0.02936109, -0.02981403];

        let ab = ArcString { points: &a + &b };
        let cd = ArcString { points: &c + &d };
        assert!((&ab).intersects(&cd));
        let r = (&ab).intersection(&cd);
        assert!(r.len() == 3);
        assert!(Zip::from(r.points().xyz.rows()).all(|point| (&point
            - &reference_intersection.view())
            .abs()
            .sum()
            < tolerance));

        // assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
        // r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
        // assert r.shape == (2, 3)
        // assert_allclose(r[0], reference_intersection)
        // assert np.all(np.isnan(r[1]))

        // Test parallel arcs
        let r = (&ab).intersection(&ab);
        assert!(r.points().xyz.is_all_nan());
    }

    #[test]
    fn test_length() {
        let tolerance = 1e-10;

        let a = VectorPoint::try_from_lonlat(&array![90.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        let ab = ArcString { points: &a + &b };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - std::f64::consts::PI).abs() < tolerance);

        let a = VectorPoint::try_from_lonlat(&array![135.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        let ab = ArcString { points: &a + &b };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - (3.0 / 4.0) * std::f64::consts::PI).abs() < tolerance);

        let a = VectorPoint::try_from_lonlat(&array![0.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![0.0, 90.0].view(), true).unwrap();
        let ab = ArcString { points: &a + &b };
        assert_eq!(ab.length(), (&a).distance(&b));
        assert!((ab.length() - std::f64::consts::PI / 2.0).abs() < tolerance);
    }

    #[test]
    fn test_angle() {
        let a = VectorPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        let b = VectorPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        let c = VectorPoint::try_from(array![0.0, 0.0, 1.0]).unwrap();
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);

        // TODO: More angle tests
    }

    #[test]
    fn test_angle_domain() {
        let a = VectorPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        let b = VectorPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        let c = VectorPoint::try_from(array![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);
        assert!(!(b.angle(&a, &c, false)).is_infinite());
    }

    #[test]
    fn test_length_domain() {
        let a = VectorPoint::try_from(array![std::f64::NAN, 0.0, 0.0]).unwrap();
        let b = VectorPoint::try_from(array![0.0, 0.0, std::f64::INFINITY]).unwrap();
        assert!((&a).distance(&b).is_nan());
    }

    #[test]
    fn test_angle_nearly_coplanar_vec() {
        // test from issue #222 + extra values
        let a =
            MultiVectorPoint::try_from(array![1.0, 1.0, 1.0].broadcast((5, 3)).unwrap().to_owned())
                .unwrap();
        let b = MultiVectorPoint::try_from(
            array![1.0, 0.9999999, 1.0]
                .broadcast((5, 3))
                .unwrap()
                .to_owned(),
        )
        .unwrap();
        let c = MultiVectorPoint::try_from(array![
            [1.0, 0.5, 1.0],
            [1.0, 0.15, 1.0],
            [1.0, 0.001, 1.0],
            [1.0, 0.15, 1.0],
            [-1.0, 0.1, -1.0],
        ])
        .unwrap();
        // vectors = np.stack([A, B, C], axis=0)
        let angles = b.angles(&a, &c, false);

        assert!(
            Zip::from(&angles.slice(s![..-1]).abs_sub(std::f64::consts::PI))
                .all(|value| value < &1e-16)
        );
        assert!(Zip::from(&angles.slice(s![-1]).abs()).all(|value| value < &1e-32));
    }
}
