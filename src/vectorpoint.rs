use crate::arcstring::{angle, collinear};
use crate::geometry::BoundingBox;
use impl_ops::impl_op_ex;
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use std::ops;

#[inline(always)]
pub fn min_1darray(arr: &ArrayView1<f64>) -> f64 {
    unsafe {
        *(arr
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
            .unwrap_unchecked())
    }
}

#[inline(always)]
pub fn max_1darray(arr: &ArrayView1<f64>) -> f64 {
    unsafe {
        *(arr
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
            .unwrap_unchecked())
    }
}

/// normalize the given vector to length 1 (the unit sphere) while preserving direction
pub fn normalize_vector(xyz: &ArrayView1<f64>) -> Array1<f64> {
    xyz / xyz.pow2().sum().sqrt()
}

pub fn cross_vector(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let bx = b[0];
    let by = b[1];
    let bz = b[2];

    array![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
}

pub fn cross_vectors(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let ax = a.slice(s![.., 0]);
    let ay = a.slice(s![.., 1]);
    let az = a.slice(s![.., 2]);
    let bx = b.slice(s![.., 0]);
    let by = b.slice(s![.., 1]);
    let bz = b.slice(s![.., 2]);

    let result = stack(
        Axis(0),
        &[
            (&ay * &bz - &az - &by).view(),
            (&az * &bx - &ax * &bz).view(),
            (&ax * &by - &ay * &ax).view(),
        ],
    );
    unsafe { result.unwrap_unchecked() }
}

/// normalize the given vectors to length 1 (the unit sphere) while preserving direction
pub fn normalize_vectors(xyz: &ArrayView2<f64>) -> Array2<f64> {
    xyz / xyz
        .pow2()
        .sum_axis(Axis(1))
        .sqrt()
        .to_shape((xyz.shape()[0], 1))
        .unwrap()
        .to_owned()
}

#[pyclass]
#[derive(Clone)]
pub struct VectorPoint {
    pub xyz: Array1<f64>,
}
impl TryFrom<Array1<f64>> for VectorPoint {
    type Error = ();

    #[inline]
    fn try_from(xyz: Array1<f64>) -> Result<Self, Self::Error> {
        if xyz.len() == 3 {
            Ok(Self { xyz })
        } else {
            Err(())
        }
    }
}

impl Into<Array1<f64>> for VectorPoint {
    #[inline]
    fn into(self) -> Array1<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView1<'p, f64>> for &'p VectorPoint {
    #[inline]
    fn into(self) -> ArrayView1<'p, f64> {
        self.xyz.view()
    }
}

impl TryFrom<Vec<f64>> for VectorPoint {
    type Error = ();
    #[inline]
    fn try_from(xyz: Vec<f64>) -> Result<Self, Self::Error> {
        Self::try_from(Array1::<f64>::from_vec(xyz))
    }
}

impl Into<Vec<f64>> for VectorPoint {
    #[inline]
    fn into(self) -> Vec<f64> {
        self.xyz.to_vec()
    }
}

impl From<[f64; 3]> for VectorPoint {
    #[inline]
    fn from(xyz: [f64; 3]) -> Self {
        unsafe { Self::try_from(xyz.to_vec()).unwrap_unchecked() }
    }
}

impl Into<[f64; 3]> for VectorPoint {
    #[inline]
    fn into(self) -> [f64; 3] {
        unsafe { self.xyz.to_vec().try_into().unwrap_unchecked() }
    }
}

impl Into<MultiVectorPoint> for VectorPoint {
    #[inline]
    fn into(self) -> MultiVectorPoint {
        MultiVectorPoint {
            xyz: unsafe { self.xyz.to_shape((1, 3)).unwrap_unchecked() }.to_owned(),
        }
    }
}

impl VectorPoint {
    pub fn new(xyz: &ArrayView1<f64>) -> Self {
        Self {
            xyz: xyz.to_owned(),
        }
    }

    pub fn from_lonlat(coordinates: &ArrayView1<f64>, degrees: bool) -> Self {
        let coordinates = if degrees {
            coordinates.to_radians()
        } else {
            coordinates.to_owned()
        };

        return Self {
            xyz: array![
                coordinates[0].cos() * coordinates[1].cos(),
                coordinates[0].sin() * coordinates[1].cos(),
                coordinates[1].sin(),
            ],
        };
    }

    pub fn to_lonlat(&self, degrees: bool) -> Array1<f64> {
        let mut lon = self.xyz[1].atan2(self.xyz[0]);
        let full_rotation = 2.0 * std::f64::consts::PI;
        if lon < 0.0 {
            lon += full_rotation;
        }
        if lon > full_rotation {
            lon -= full_rotation;
        }
        let radians = array![
            lon,
            self.xyz[2].atan2((self.xyz[0].powi(2) + self.xyz[1].powi(2)).sqrt())
        ];
        return if degrees {
            radians.to_degrees()
        } else {
            radians
        };
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vector(&self.into(), &other.into());
        unsafe { crossed.try_into().unwrap_unchecked() }
    }

    pub fn vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
        let theta = if degrees { theta.to_radians() } else { theta };

        let a = &self.normalized().xyz;
        let ax = a[0];
        let ay = a[1];
        let az = a[2];

        let b = &other.normalized().xyz;
        let bx = b[0];
        let by = b[1];
        let bz = b[2];

        Self {
            xyz: -b * -a * b * (1.0 - theta.cos())
                + a * theta.cos()
                + array![-bz * ay + by * az, bz * ax - bx * az, -by * ax - bx * ay,] * theta.sin(),
        }
    }
}

impl ToString for VectorPoint {
    fn to_string(&self) -> String {
        format!("VectorPoint({0})", self.xyz)
    }
}

impl PartialEq for VectorPoint {
    fn eq(&self, other: &VectorPoint) -> bool {
        &self == &other
    }
}

impl PartialEq<&VectorPoint> for VectorPoint {
    fn eq(&self, other: &&VectorPoint) -> bool {
        (&self.xyz - &other.xyz).sum() < 3e-11
    }
}

impl_op_ex!(+ |a: &VectorPoint, b: &VectorPoint| -> MultiVectorPoint{ MultiVectorPoint {
                xyz: stack(Axis(0), &[a.xyz.view(), b.xyz.view()]).unwrap(),
            } });
/// xyz vectors representing points on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiVectorPoint {
    pub xyz: Array2<f64>,
}

impl TryFrom<Array2<f64>> for MultiVectorPoint {
    type Error = ();

    #[inline]
    fn try_from(xyz: Array2<f64>) -> Result<Self, Self::Error> {
        if xyz.shape()[1] == 3 {
            Ok(Self { xyz })
        } else {
            Err(())
        }
    }
}

impl Into<Array2<f64>> for MultiVectorPoint {
    #[inline]
    fn into(self) -> Array2<f64> {
        self.xyz
    }
}

impl<'p> Into<ArrayView2<'p, f64>> for &'p MultiVectorPoint {
    #[inline]
    fn into(self) -> ArrayView2<'p, f64> {
        self.xyz.view()
    }
}

impl From<Vec<[f64; 3]>> for MultiVectorPoint {
    fn from(xyz: Vec<[f64; 3]>) -> Self {
        unsafe { Self::try_from(Array2::<f64>::from(xyz)).unwrap_unchecked() }
    }
}

impl Into<Vec<[f64; 3]>> for MultiVectorPoint {
    fn into(self) -> Vec<[f64; 3]> {
        self.xyz
            .rows()
            .into_iter()
            .map(|row| unsafe { row.to_vec().try_into().unwrap_unchecked() })
            .collect()
    }
}

impl MultiVectorPoint {
    pub fn new(xyz: &ArrayView2<f64>) -> Self {
        Self {
            xyz: xyz.to_owned(),
        }
    }

    pub fn from_lonlats(coordinates: &ArrayView2<f64>, degrees: bool) -> Self {
        let coordinates = if degrees {
            coordinates.to_radians()
        } else {
            coordinates.to_owned()
        };
        let lon = coordinates.slice(s![.., 0]);
        let lat = coordinates.slice(s![.., 1]);

        return Self {
            xyz: stack(
                Axis(1),
                &[
                    (lon.cos() * lat.cos()).view(),
                    (lon.sin() * lat.cos()).view(),
                    lat.sin().view(),
                ],
            )
            .unwrap(),
        };
    }

    pub fn to_lonlats(&self, degrees: bool) -> Array2<f64> {
        let mut lons =
            Zip::from(self.xyz.rows()).par_map_collect(|vector| vector[1].atan2(vector[0]));
        let full_rotation = 2.0 * std::f64::consts::PI;
        lons.par_mapv_inplace(|lon| {
            if lon < 0.0 {
                lon + full_rotation
            } else if lon > full_rotation {
                lon - full_rotation
            } else {
                lon
            }
        });
        let lats = Zip::from(self.xyz.rows()).par_map_collect(|vector| {
            vector[2].atan2((vector[0].powi(2) + vector[1].powi(2)).sqrt())
        });
        let radians = stack(Axis(1), &[lons.view(), lats.view()]).unwrap();
        if degrees {
            radians.to_degrees()
        } else {
            radians
        }
    }

    pub fn vector_lengths(&self) -> Array1<f64> {
        self.xyz.pow2().sum_axis(Axis(1)).sqrt()
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vectors(&self.into(), &other.into());
        unsafe { crossed.try_into().unwrap_unchecked() }
    }

    pub fn vector_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
        let theta = if degrees { theta.to_radians() } else { theta };

        let a = &self.normalized().xyz;
        let ax = a.slice(s![.., 0]);
        let ay = a.slice(s![.., 1]);
        let az = a.slice(s![.., 2]);

        let b = &other.normalized().xyz;
        let bx = b.slice(s![.., 0]);
        let by = b.slice(s![.., 1]);
        let bz = b.slice(s![.., 2]);

        Self {
            xyz: -b * -a * b * (1.0 - theta.cos())
                + a * theta.cos()
                + stack(
                    Axis(0),
                    &[
                        (-&bz * &ay + &by * &az).view(),
                        (&bz * &ax - &bx * &az).view(),
                        (-&by * &ax - &bx * &ay).view(),
                    ],
                )
                .unwrap()
                    * theta.sin(),
        }
    }

    pub fn angles(&self, a: &MultiVectorPoint, b: &MultiVectorPoint, degrees: bool) -> Array1<f64> {
        Zip::from(self.xyz.rows())
            .and(a.xyz.rows())
            .and(b.xyz.rows())
            .par_map_collect(|point, a, b| angle(&point, &a, &b, degrees))
    }

    pub fn collinear(&self, a: &VectorPoint, b: &VectorPoint) -> Array1<bool> {
        Array1::from_vec(
            self.vectorpoints()
                .iter()
                .map(|point| collinear(&a.xyz.view(), &point.xyz.view(), &b.xyz.view()))
                .collect(),
        )
    }
}

impl ToString for MultiVectorPoint {
    fn to_string(&self) -> String {
        format!("MultiVectorPoint({0})", self.xyz)
    }
}

impl PartialEq for MultiVectorPoint {
    fn eq(&self, other: &MultiVectorPoint) -> bool {
        &self == &other
    }
}

impl PartialEq<&MultiVectorPoint> for MultiVectorPoint {
    fn eq(&self, other: &&MultiVectorPoint) -> bool {
        (&self.xyz - &other.xyz).sum() < 3e-11
    }
}

impl_op_ex!(+ |a: &MultiVectorPoint, b: &MultiVectorPoint| -> MultiVectorPoint{ MultiVectorPoint {
                xyz: concatenate(Axis(0), &[a.xyz.view(), b.xyz.view()]).unwrap(),
            } });

impl BoundingBox for MultiVectorPoint {
    fn bounds(&self, degrees: bool) -> [f64; 4] {
        let coordinates = self.to_lonlats(degrees);
        let x = coordinates.slice(s![.., 0]);
        let y = coordinates.slice(s![.., 1]);

        [
            min_1darray(&x),
            min_1darray(&y),
            max_1darray(&x),
            max_1darray(&y),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorpoint::{MultiVectorPoint, VectorPoint};
    use ndarray::linspace;

    #[test]
    fn test_normalize() {
        let xyz = Array1::<f64>::from_iter(linspace(-100.0, 100.0, 18))
            .broadcast((18, 3))
            .unwrap()
            .to_owned();
        let points = MultiVectorPoint { xyz };

        assert_ne!(
            points.vector_lengths(),
            array![1.0].broadcast(points.xyz.nrows()).unwrap()
        );

        let normalized = points.normalized();

        assert_eq!(
            normalized.vector_lengths(),
            array![1.0].broadcast(normalized.xyz.nrows()).unwrap()
        );

        assert_eq!(
            normalized.xyz.powi(2).sum_axis(Axis(1)).sqrt(),
            array![1.0].broadcast(points.xyz.nrows()).unwrap()
        );
    }

    #[test]
    fn test_already_normalized() {
        for i in 0..3 {
            let mut xyz = array![0.0, 0.0, 0.0];
            xyz[i] = 1.0;
            let normalized = VectorPoint { xyz }.normalized().xyz;
            let length = normalized.powi(2).sum().sqrt();
            assert_eq!(length, 1.0);
        }
    }

    #[test]
    fn test_from_lonlat() {
        let lat = Array1::<f64>::from_iter(linspace(-360.0, 360.0, 360));

        let north_lon = array![90.0];
        let north_pole = MultiVectorPoint::from_lonlats(
            &stack(
                Axis(1),
                &[lat.view(), north_lon.broadcast(lat.len()).unwrap()],
            )
            .unwrap()
            .view(),
            false,
        )
        .xyz;
        assert_eq!(
            north_pole.slice(s![.., 0]),
            array![0.0].broadcast(north_pole.nrows()).unwrap()
        );
        assert_eq!(
            north_pole.slice(s![.., 1]),
            array![0.0].broadcast(north_pole.nrows()).unwrap()
        );
        assert_eq!(
            north_pole.slice(s![.., 2]),
            array![1.0].broadcast(north_pole.nrows()).unwrap()
        );

        let south_lon = array![-90.0];
        let south_pole = MultiVectorPoint::from_lonlats(
            &stack(
                Axis(1),
                &[lat.view(), south_lon.broadcast(lat.len()).unwrap()],
            )
            .unwrap()
            .view(),
            false,
        )
        .xyz;
        assert_eq!(
            south_pole.slice(s![.., 0]),
            array![0.0].broadcast(south_pole.len()).unwrap()
        );
        assert_eq!(
            south_pole.slice(s![.., 1]),
            array![0.0].broadcast(south_pole.len()).unwrap()
        );
        assert_eq!(
            south_pole.slice(s![.., 2]),
            array![-1.0].broadcast(south_pole.len()).unwrap()
        );

        let equator_lon = array![0.0];
        let equator = MultiVectorPoint::from_lonlats(
            &stack(
                Axis(1),
                &[lat.view(), equator_lon.broadcast(lat.len()).unwrap()],
            )
            .unwrap()
            .view(),
            false,
        )
        .xyz;
        assert_eq!(
            equator.slice(s![.., 2]),
            array![0.0].broadcast(equator.len()).unwrap()
        );
    }

    #[test]
    fn test_to_lonlats() {
        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let lonlats = array![[0., 90.], [0., -90.], [45., 0.], [315., 0.]];

        let a = VectorPoint {
            xyz: xyz.slice(s![0, ..]).to_owned(),
        };
        assert_eq!(a.to_lonlat(true), lonlats.slice(s![0, ..]).to_owned());

        let b = VectorPoint {
            xyz: xyz.slice(s![1, ..]).to_owned(),
        };
        assert_eq!(b.to_lonlat(true), lonlats.slice(s![1, ..]).to_owned());

        let c = VectorPoint {
            xyz: xyz.slice(s![2, ..]).to_owned(),
        };
        assert_eq!(c.to_lonlat(true), lonlats.slice(s![2, ..]).to_owned());

        let d = VectorPoint {
            xyz: xyz.slice(s![3, ..]).to_owned(),
        };
        assert_eq!(d.to_lonlat(true), lonlats.slice(s![3, ..]).to_owned());

        let abcd = MultiVectorPoint { xyz };
        assert_eq!(abcd.to_lonlats(true), lonlats);
    }

    #[test]
    fn test_str() {
        assert_eq!(
            VectorPoint {
                xyz: array![0.0, 1.0, 2.0]
            }
            .to_string(),
            "VectorPoint([0, 1, 2])"
        );
        assert_eq!(
            MultiVectorPoint {
                xyz: array![[0.0, 1.0, 2.0]]
            }
            .to_string(),
            "MultiVectorPoint([[0, 1, 2]])"
        );
    }

    #[test]
    fn test_add() {
        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let a = VectorPoint {
            xyz: xyz.slice(s![0, ..]).to_owned(),
        };
        let b = VectorPoint {
            xyz: xyz.slice(s![1, ..]).to_owned(),
        };
        let c = VectorPoint {
            xyz: xyz.slice(s![2, ..]).to_owned(),
        };
        let d = VectorPoint {
            xyz: xyz.slice(s![3, ..]).to_owned(),
        };

        let ab = MultiVectorPoint {
            xyz: xyz.slice(s![0..2, ..]).to_owned(),
        };
        let bc = MultiVectorPoint {
            xyz: xyz.slice(s![1..3, ..]).to_owned(),
        };
        let cd = MultiVectorPoint {
            xyz: xyz.slice(s![2..4, ..]).to_owned(),
        };
        let da = MultiVectorPoint {
            xyz: stack(Axis(0), &[xyz.slice(s![3, ..]), xyz.slice(s![0, ..])]).unwrap(),
        };

        assert_eq!(&a + &b, ab);
        assert_eq!(&b + &c, bc);
        assert_eq!(&c + &d, cd);
        assert_eq!(&d + &a, da);
    }
}
