use crate::vectorpoint::{cross_vectors, max_1darray, min_1darray, normalize_vector};
use crate::{
    geometry::BoundingBox,
    vectorpoint::{MultiVectorPoint, VectorPoint},
};
use numpy::ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, array, concatenate, linspace, s, stack,
};
use pyo3::prelude::*;
use std::ops::Add;

pub fn interpolate(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    n: usize,
) -> Result<Array2<f64>, String> {
    let n = if n < 2 { 2 } else { n };
    let t = Array1::<f64>::from_iter(linspace(0.0, 1.0, n));
    let t = t.to_shape((n, 1)).unwrap();
    let omega = arc_length(a, b);

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
pub fn angle(a: &ArrayView1<f64>, b: &ArrayView1<f64>, c: &ArrayView1<f64>, degrees: bool) -> f64 {
    let tolerance = 3e-11;

    let ab = arc_length(a, b);
    let bc = arc_length(b, c);
    let ca = arc_length(c, a);

    let angle = if ab > tolerance && bc > tolerance {
        (ca.cos() - bc.cos() * ab.cos()) / (bc.sin() * ab.sin()).acos()
    } else {
        (1.0 - ca.powi(2) / 2.0).acos()
    };

    if degrees { angle.to_degrees() } else { angle }
}

/// given points A, B, and C on the unit sphere, retrieve the angle at B between arc AB and arc BC
///
/// References:
/// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
pub fn angles(
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

    if degrees { angles.to_degrees() } else { angles }
}

/// radians subtended by this arc on the sphere
pub fn arc_length(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    normalize_vector(a).dot(&normalize_vector(b)).acos()
}

/// surface area of a spherical triangle via Girard's theorum
pub fn spherical_triangle_area(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> f64 {
    angle(c, a, b, false) + angle(a, b, c, false) + angle(b, c, a, false) - std::f64::consts::PI
}

/// whether the three points exist on the same line
pub fn collinear(a: &ArrayView1<f64>, b: &ArrayView1<f64>, c: &ArrayView1<f64>) -> bool {
    let tolerance = 3e-11;
    spherical_triangle_area(a, b, c) < tolerance
}

/// series of great circle arcs along the sphere
#[pyclass]
#[derive(Clone)]
pub struct ArcString {
    pub points: MultiVectorPoint,
}

impl From<MultiVectorPoint> for ArcString {
    fn from(points: MultiVectorPoint) -> Self {
        Self { points }
    }
}

impl Into<MultiVectorPoint> for ArcString {
    fn into(self) -> MultiVectorPoint {
        self.points
    }
}

impl Into<Vec<ArcString>> for ArcString {
    fn into(self) -> Vec<ArcString> {
        let vectors = self.points.xyz;
        let mut arcs = vec![];
        for index in 0..vectors.nrows() - 1 {
            arcs.push(ArcString {
                points: MultiVectorPoint {
                    xyz: vectors.slice(s![index..index + 1, ..]).to_owned(),
                },
            })
        }

        arcs
    }
}

impl ArcString {
    pub fn midpoints(&self) -> MultiVectorPoint {
        MultiVectorPoint {
            xyz: (&self.points.xyz.slice(s![..-1, ..]) + &self.points.xyz.slice(s![1.., ..]) / 2.0)
                .to_owned(),
        }
    }

    pub fn lengths(&self) -> Array1<f64> {
        Zip::from(self.points.xyz.slice(s![..-1, ..]).rows())
            .and(self.points.xyz.slice(s![1.., ..]).rows())
            .par_map_collect(|a, b| arc_length(&a, &b))
    }

    pub fn length(&self) -> f64 {
        self.lengths().sum()
    }

    pub fn contains(&self, point: &VectorPoint) -> bool {
        // check if point is one of the vertices of this linestring
        if self.points.contains(point) {
            return true;
        }

        // check if point is within the bounding box
        let bounds = self.bounds(false);
        let pc = point.to_lonlat(false);
        if pc[0] >= bounds[0] && pc[1] <= bounds[2] && pc[1] >= bounds[1] && pc[1] <= bounds[3] {
            // compare lengths to endpoints with the arc length
            for index in 0..self.points.xyz.nrows() - 1 {
                let a = self.points.xyz.slice(s![index, ..]);
                let b = self.points.xyz.slice(s![index + 1, ..]);
                let p = point.xyz.view();

                if collinear(&a.view(), &p.view(), &b.view()) {
                    return true;
                }

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
        }

        return false;
    }

    pub fn intersects(&self, other: &Self) -> bool {
        // TODO: write an intersects algorithm
        self.intersection(other).is_some()
    }

    pub fn intersection(&self, other: &Self) -> Option<MultiVectorPoint> {
        // TODO: implement
        None
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

impl Add for ArcString {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&ArcString> for &ArcString {
    type Output = ArcString;

    fn add(self, rhs: &ArcString) -> Self::Output {
        Self::Output {
            points: &self.points + &rhs.points,
        }
    }
}

impl BoundingBox for ArcString {
    fn bounds(&self, degrees: bool) -> [f64; 4] {
        let coordinates = self.points.to_lonlats(degrees);

        let x = coordinates.slice(s![.., 0]);
        let y = coordinates.slice(s![.., 1]);

        [
            min_1darray(&x).unwrap_or(std::f64::NAN),
            min_1darray(&y).unwrap_or(std::f64::NAN),
            max_1darray(&x).unwrap_or(std::f64::NAN),
            max_1darray(&y).unwrap_or(std::f64::NAN),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Distance;
    use crate::vectorpoint::{MultiVectorPoint, VectorPoint};

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
        assert!(
            arc.contains(
                &VectorPoint::try_from_lonlat(&array![349.10660535, -12.30998866].view(), true)
                    .unwrap()
            )
        );

        let vertical_arc = ArcString {
            points: MultiVectorPoint::try_from_lonlats(
                &array![[60.0, 0.0], [60.0, 30.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!(
                vertical_arc.contains(
                    &VectorPoint::try_from_lonlat(&array![60.0, i].view(), true).unwrap()
                )
            )
        }

        let horizontal_arc = ArcString {
            points: MultiVectorPoint::try_from_lonlats(
                &array![[0.0, 60.0], [30.0, 60.0]].view(),
                true,
            )
            .unwrap(),
        };
        for i in linspace(1., 29., 1) {
            assert!(
                horizontal_arc.contains(
                    &VectorPoint::try_from_lonlat(&array![i, 60.0].view(), true).unwrap()
                )
            );
        }
    }

    #[test]
    fn test_interpolate() {
        let tolerance = 1e-10;

        let a_lonlat = array![60.0, 0.0];
        let b_lonlat = array![60.0, 30.0];
        let lonlats = interpolate(&a_lonlat.view(), &b_lonlat.view(), 10).unwrap();

        let a = VectorPoint::try_from_lonlat(&a_lonlat.view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&b_lonlat.view(), true).unwrap();

        assert!(
            Zip::from(&lonlats.slice(s![0, ..]))
                .and(&a_lonlat.view())
                .all(|test, reference| (test - reference).abs() < tolerance)
        );
        assert!(
            Zip::from(&lonlats.slice(s![-1, ..]))
                .and(&b_lonlat.view())
                .all(|test, reference| (test - reference).abs() < tolerance)
        );

        let xyzs = interpolate(&a.xyz.view(), &b.xyz.view(), 10).unwrap();

        assert!(
            Zip::from(&xyzs.slice(s![0, ..]))
                .and(&a.xyz.view())
                .all(|test, reference| (test - reference).abs() < tolerance)
        );
        assert!(
            Zip::from(&xyzs.slice(s![-1, ..]))
                .and(&b.xyz.view())
                .all(|test, reference| (test - reference).abs() < tolerance)
        );

        let arc_from_lonlats = ArcString {
            points: MultiVectorPoint::try_from_lonlats(&lonlats.view(), true).unwrap(),
        };
        let arc_from_xyzs = ArcString {
            points: MultiVectorPoint {
                xyz: xyzs.to_owned(),
            },
        };

        for xyz in xyzs.rows() {
            let point = VectorPoint {
                xyz: xyz.to_owned(),
            };
            assert!(arc_from_lonlats.contains(&point));
            assert!(arc_from_xyzs.contains(&point));
        }

        let distances_from_lonlats = arc_from_lonlats.lengths();
        let distances_from_xyz = arc_from_xyzs.lengths();

        assert!(
            Zip::from(&distances_from_lonlats)
                .and(&distances_from_xyz)
                .all(|from_lonlats, from_xyz| (from_lonlats - from_xyz).abs() < tolerance)
        );
    }

    #[test]
    fn test_intersection() {
        let tolerance = 1e-10;

        let a = VectorPoint::try_from_lonlat(&array![-10.0, -10.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![10.0, 10.0].view(), true).unwrap();

        let c = VectorPoint::try_from_lonlat(&array![-25.0, 10.0].view(), true).unwrap();
        let d = VectorPoint::try_from_lonlat(&array![15.0, -10.0].view(), true).unwrap();

        let e = VectorPoint::try_from_lonlat(&array![-20.0, 40.0].view(), true).unwrap();
        let f = VectorPoint::try_from_lonlat(&array![20.0, 40.0].view(), true).unwrap();

        let reference_intersection = array![0.99912414, -0.02936109, -0.02981403];

        let ab = ArcString { points: a + b };
        let cd = ArcString { points: c + d };
        assert!(ab.intersects(&cd));
        let r = ab.intersection(&cd).unwrap();
        assert!(r.len() == 3);
        assert!(
            Zip::from(r.xyz.rows())
                .all(|point| (&point - &reference_intersection.view()).abs().sum() < tolerance)
        );

        // assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
        // r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
        // assert r.shape == (2, 3)
        // assert_allclose(r[0], reference_intersection)
        // assert np.all(np.isnan(r[1]))

        // Test parallel arcs
        let r = ab.intersection(&ab).unwrap();
        assert!(r.xyz.is_all_nan());
    }

    #[test]
    fn test_distance() {
        let tolerance = 1e-10;

        let a = VectorPoint::try_from_lonlat(&array![90.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        assert!(((&a).distance(&b) - std::f64::consts::PI).abs() < tolerance);

        let a = VectorPoint::try_from_lonlat(&array![135.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![-90.0, 0.0].view(), true).unwrap();
        assert!(((&a).distance(&b) - (3.0 / 4.0) * std::f64::consts::PI).abs() < tolerance);

        let a = VectorPoint::try_from_lonlat(&array![0.0, 0.0].view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&array![0.0, 90.0].view(), true).unwrap();
        assert!(((&a).distance(&b) - std::f64::consts::PI / 2.0).abs() < tolerance);
    }

    #[test]
    fn test_angle() {
        let a = VectorPoint {
            xyz: array![1.0, 0.0, 0.0],
        };
        let b = VectorPoint {
            xyz: array![0.0, 1.0, 0.0],
        };
        let c = VectorPoint {
            xyz: array![0.0, 0.0, 1.0],
        };
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);

        // TODO: More angle tests
    }

    #[test]
    fn test_angle_domain() {
        let a = VectorPoint {
            xyz: array![0.0, 0.0, 0.0],
        };
        let b = VectorPoint {
            xyz: array![0.0, 0.0, 0.0],
        };
        let c = VectorPoint {
            xyz: array![0.0, 0.0, 0.0],
        };
        assert_eq!(b.angle(&a, &c, false), (3.0 / 2.0) * std::f64::consts::PI);
        assert!(!(b.angle(&a, &c, false)).is_infinite());
    }

    #[test]
    fn test_length_domain() {
        let a = VectorPoint {
            xyz: array![std::f64::NAN, 0.0, 0.0],
        };
        let b = VectorPoint {
            xyz: array![0.0, 0.0, std::f64::INFINITY],
        };
        assert!((&a).distance(&b).is_nan());
    }

    #[test]
    fn test_angle_nearly_coplanar_vec() {
        // test from issue #222 + extra values
        let a = MultiVectorPoint {
            xyz: array![1.0, 1.0, 1.0].broadcast((5, 3)).unwrap().to_owned(),
        };
        let b = MultiVectorPoint {
            xyz: array![1.0, 0.9999999, 1.0]
                .broadcast((5, 3))
                .unwrap()
                .to_owned(),
        };
        let c = MultiVectorPoint {
            xyz: array![
                [1.0, 0.5, 1.0],
                [1.0, 0.15, 1.0],
                [1.0, 0.001, 1.0],
                [1.0, 0.15, 1.0],
                [-1.0, 0.1, -1.0],
            ],
        };
        // vectors = np.stack([A, B, C], axis=0)
        let angles = b.angles(&a, &c, false);

        assert!(
            Zip::from(&angles.slice(s![..-1]).abs_sub(std::f64::consts::PI))
                .all(|value| value < &1e-16)
        );
        assert!(Zip::from(&angles.slice(s![-1]).abs()).all(|value| value < &1e-32));
    }
}
