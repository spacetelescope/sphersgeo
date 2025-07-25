use crate::vectorpoint::{cross_vectors, max_1darray, min_1darray, normalize_vector};
use crate::{
    geometry::BoundingBox,
    vectorpoint::{MultiVectorPoint, VectorPoint},
};
use impl_ops::impl_op_ex;
use numpy::ndarray::{
    concatenate, linspace, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use std::ops;

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
                    b.to_shape((1, 3)).unwrap().view(),
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
    let ab = arc_length(a, b);
    let bc = arc_length(b, c);
    let ca = arc_length(c, a);

    let angle = if ab > 3e-11 && bc > 3e-11 {
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

    if degrees {
        angles.to_degrees()
    } else {
        angles
    }
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
    spherical_triangle_area(a, b, c) < 3e-11
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

                // if left + right - total < 3e-11 {
                //     // ensure angle is flat
                //     if angle(&a, &point.xyz.view(), &b, false) - std::f64::consts::PI < 3e-11 {
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

impl_op_ex!(+ |a: &ArcString, b: &ArcString| -> ArcString { ArcString {
                points: &a.points + &b.points,
            } });

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
