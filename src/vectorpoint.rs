use crate::arcstring::{angle, arc_length, collinear};
use crate::geometry::{BoundingBox, Distance};
use impl_ops::impl_op_ex;
use numpy::ndarray::{
    array, concatenate, s, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use pyo3::prelude::*;
use std::ops;

#[inline(always)]
pub fn min_1darray(arr: &ArrayView1<f64>) -> Option<f64> {
    if arr.is_any_nan() || arr.is_any_infinite() {
        None
    } else {
        Some(unsafe {
            *(arr
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
                .unwrap_unchecked())
        })
    }
}

#[inline(always)]
pub fn max_1darray(arr: &ArrayView1<f64>) -> Option<f64> {
    if arr.is_any_nan() || arr.is_any_infinite() {
        None
    } else {
        Some(unsafe {
            *(arr
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
                .unwrap_unchecked())
        })
    }
}

/// normalize the given vector to length 1 (the unit sphere) while preserving direction
pub fn normalize_vector(xyz: &ArrayView1<f64>) -> Array1<f64> {
    xyz / xyz.pow2().sum().sqrt()
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

/// xyz vector representing a point on the sphere
#[pyclass]
#[derive(Clone)]
pub struct VectorPoint {
    pub xyz: Array1<f64>,
}

impl TryFrom<Array1<f64>> for VectorPoint {
    type Error = String;

    #[inline]
    fn try_from(xyz: Array1<f64>) -> Result<Self, Self::Error> {
        if xyz.len() != 3 {
            Err(format!("array should have length 3, not {:?}", xyz.len()))
        } else {
            Ok(Self { xyz })
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
    type Error = String;

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
    /// from the given coordinates, build an xyz vector representing a point on the sphere
    pub fn try_from_lonlat(coordinates: &ArrayView1<f64>, degrees: bool) -> Result<Self, String> {
        if coordinates.len() == 2 {
            let coordinates = if degrees {
                coordinates.to_radians()
            } else {
                coordinates.to_owned()
            };

            Ok(Self {
                xyz: array![
                    coordinates[0].cos() * coordinates[1].cos(),
                    coordinates[0].sin() * coordinates[1].cos(),
                    coordinates[1].sin(),
                ],
            })
        } else {
            Err(String::from("invalid shape"))
        }
    }

    /// convert this point on the sphere to angular coordinates
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

    /// normalize the given xyz vector
    pub fn normalize(xyz: &ArrayView1<f64>) -> Self {
        Self {
            xyz: normalize_vector(xyz),
        }
    }

    /// normalize this vector to length 1 (the unit sphere) while preserving direction
    pub fn normalized(&self) -> Self {
        Self {
            xyz: normalize_vector(&self.xyz.view()),
        }
    }

    /// angle on the sphere between this point and two other points
    pub fn angle(&self, a: &VectorPoint, b: &VectorPoint, degrees: bool) -> f64 {
        angle(&a.into(), &self.into(), &b.into(), degrees)
    }

    /// whether this point lies exactly between the given points
    pub fn collinear(&self, a: &VectorPoint, b: &VectorPoint) -> bool {
        collinear(&a.xyz.view(), &self.xyz.view(), &b.xyz.view())
    }

    /// length of the underlying xyz vector
    pub fn vector_length(&self) -> f64 {
        self.xyz.pow2().sum().sqrt()
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vector(&self.into(), &other.into());
        unsafe { crossed.try_into().unwrap_unchecked() }
    }

    /// rotate this xyz vector by theta angle around another xyz vector
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

impl Distance for &VectorPoint {
    fn distance(&self, other: Self) -> f64 {
        if self.xyz == other.xyz {
            0.
        } else {
            arc_length(&self.xyz.view(), &other.xyz.view())
        }
    }
}

/// xyz vectors representing points on the sphere
#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiVectorPoint {
    pub xyz: Array2<f64>,
}

impl From<Vec<VectorPoint>> for MultiVectorPoint {
    fn from(points: Vec<VectorPoint>) -> Self {
        let mut xyz = Array2::zeros((points.len(), 3));
        for (index, mut row) in xyz.axis_iter_mut(Axis(0)).enumerate() {
            row[0] = points[index].xyz[0];
            row[1] = points[index].xyz[1];
            row[2] = points[index].xyz[2];
        }
        unsafe { Self::try_from(xyz).unwrap_unchecked() }
    }
}

impl Into<Vec<VectorPoint>> for MultiVectorPoint {
    fn into(self) -> Vec<VectorPoint> {
        self.xyz
            .rows()
            .into_iter()
            .map(|row| VectorPoint {
                xyz: row.to_owned(),
            })
            .collect()
    }
}

impl TryFrom<Array2<f64>> for MultiVectorPoint {
    type Error = String;

    #[inline]
    fn try_from(xyz: Array2<f64>) -> Result<Self, Self::Error> {
        if xyz.shape()[1] != 3 {
            Err(format!("array should be Nx3, not Nx{:?}", xyz.shape()[1]))
        } else {
            Ok(Self { xyz })
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
    /// from the given coordinates, build xyz vectors representing points on the sphere
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

    /// normalize the given xyz vectors
    pub fn normalize(xyz: &ArrayView2<f64>) -> Self {
        Self {
            xyz: normalize_vectors(xyz),
        }
    }

    /// normalize the underlying vectors to length 1 (the unit sphere) while preserving direction
    pub fn normalized(&self) -> Self {
        Self {
            xyz: normalize_vectors(&self.xyz.view()),
        }
    }

    /// convert to angle coordinates along the sphere
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

    /// number of points in this collection
    pub fn length(&self) -> usize {
        self.xyz.nrows()
    }

    /// whether the given point is one of these points
    pub fn contains(&self, point: &VectorPoint) -> bool {
        Zip::from(&(&point.xyz - &self.xyz).abs().sum_axis(Axis(1))).any(|diff| diff < &3e-11)
    }

    /// length of the underlying xyz vectors
    pub fn vector_lengths(&self) -> Array1<f64> {
        self.xyz.pow2().sum_axis(Axis(1)).sqrt()
    }

    pub fn vector_cross(&self, other: &Self) -> Self {
        let crossed = cross_vectors(&self.into(), &other.into());
        unsafe { crossed.try_into().unwrap_unchecked() }
    }

    /// rotate the underlying vector by theta angle around other vectors
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
        let points: Vec<VectorPoint> = self.to_owned().into();
        Array1::from_vec(
            points
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

impl BoundingBox for &MultiVectorPoint {
    fn bounds(&self, degrees: bool) -> [f64; 4] {
        let coordinates = self.to_lonlats(degrees);
        let x = coordinates.slice(s![.., 0]);
        let y = coordinates.slice(s![.., 1]);

        [
            min_1darray(&x).unwrap(),
            min_1darray(&y).unwrap(),
            max_1darray(&x).unwrap(),
            max_1darray(&y).unwrap(),
        ]
    }
}

impl Distance for &MultiVectorPoint {
    fn distance(&self, other: Self) -> f64 {
        // TODO: write a more efficient algorithm than brute-force
        min_1darray(
            &Zip::from(self.xyz.rows())
                .par_map_collect(|point| {
                    min_1darray(
                        &Zip::from(other.xyz.rows())
                            .par_map_collect(|other_point| arc_length(&point, &other_point))
                            .view(),
                    )
                    .unwrap_or(0.)
                })
                .view(),
        )
        .unwrap_or(0.)
    }
}

#[cfg(test)]
mod tests {
    use std::array::from_fn;

    use super::*;
    use crate::geometry::Distance;
    use crate::vectorpoint::{MultiVectorPoint, VectorPoint};
    use ndarray::linspace;

    #[test]
    fn test_normalize() {
        let xyz = Array1::<f64>::from_iter(linspace(-100.0, 100.0, 18));
        let points = MultiVectorPoint {
            xyz: stack(Axis(1), &[xyz.view(), xyz.view(), xyz.view()]).unwrap(),
        };

        assert_ne!(
            points.vector_lengths(),
            array![1.0].broadcast(points.xyz.nrows()).unwrap()
        );

        let normalized = points.normalized();

        assert!(Zip::from(&normalized.vector_lengths()).all(|length| length == &1.0));

        assert!(Zip::from(&normalized.xyz.powi(2).sum_axis(Axis(1)).sqrt())
            .all(|length| length == &1.0),);
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
        let a_lonlat = array![60.0, 0.0];
        let b_lonlat = array![60.0, 30.0];

        let a = VectorPoint::try_from_lonlat(&a_lonlat.view(), true).unwrap();
        let b = VectorPoint::try_from_lonlat(&b_lonlat.view(), true).unwrap();

        assert!(Zip::from(&(a.to_lonlat(true) - a_lonlat).abs()).all(|point| point < &3e-11));
        assert!(Zip::from(&(b.to_lonlat(true) - b_lonlat).abs()).all(|point| point < &3e-11));

        let lons = Array1::<f64>::from_iter(linspace(-360.0, 360.0, 360));

        let equator_lat = array![0.0];
        let equator_lats = equator_lat.broadcast(lons.len()).unwrap();
        let equators = Zip::from(&lons)
            .and(equator_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lon.to_owned(), lat.to_owned()].view(), false)
                    .unwrap()
            });
        let multi_equator = MultiVectorPoint::from_lonlats(
            &stack(Axis(1), &[equator_lats, lons.view()]).unwrap().view(),
            false,
        );

        assert!(Zip::from(multi_equator.xyz.rows())
            .and(&equators)
            .all(|multi, single| multi == single.xyz));

        assert_eq!(
            multi_equator.xyz.slice(s![.., 2]),
            Array1::<f64>::zeros(multi_equator.xyz.nrows())
        );

        let north_pole_lat = array![90.0];
        let north_pole_lats = north_pole_lat.broadcast(lons.len()).unwrap();
        let north_poles = Zip::from(&lons)
            .and(north_pole_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lon.to_owned(), lat.to_owned()].view(), false)
                    .unwrap()
            });
        let multi_north_pole = MultiVectorPoint::from_lonlats(
            &stack(Axis(1), &[north_pole_lats, lons.view()])
                .unwrap()
                .view(),
            false,
        );

        assert!(Zip::from(multi_north_pole.xyz.rows())
            .and(&north_poles)
            .all(|multi, single| multi == single.xyz));

        assert_eq!(
            multi_north_pole.xyz,
            stack(
                Axis(1),
                &[
                    Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                    Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                    Array1::<f64>::ones(multi_north_pole.xyz.nrows()).view()
                ]
            )
            .unwrap()
        );

        let south_pole_lat = array![-90.0];
        let south_pole_lats = south_pole_lat.broadcast(lons.len()).unwrap();
        let south_poles = Zip::from(&lons)
            .and(south_pole_lats)
            .par_map_collect(|lon, lat| {
                VectorPoint::try_from_lonlat(&array![lat.to_owned(), lon.to_owned()].view(), false)
                    .unwrap()
            });
        let multi_south_pole = MultiVectorPoint::from_lonlats(
            &stack(Axis(1), &[south_pole_lats, lons.view()])
                .unwrap()
                .view(),
            false,
        );

        assert!(Zip::from(multi_south_pole.xyz.rows())
            .and(&south_poles)
            .all(|multi, single| multi == single.xyz));

        assert_eq!(
            multi_south_pole.xyz.view(),
            stack(
                Axis(1),
                &[
                    Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                    Array1::<f64>::zeros(multi_north_pole.xyz.nrows()).view(),
                    (-1.0 * Array1::<f64>::ones(multi_north_pole.xyz.nrows())).view()
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn test_to_lonlat() {
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
        let ac = a.to_lonlat(true);
        assert_eq!(ac, lonlats.slice(s![0, ..]).to_owned());

        let b = VectorPoint {
            xyz: xyz.slice(s![1, ..]).to_owned(),
        };
        let bc = b.to_lonlat(true);
        assert_eq!(bc, lonlats.slice(s![1, ..]).to_owned());

        let c = VectorPoint {
            xyz: xyz.slice(s![2, ..]).to_owned(),
        };
        let cc = c.to_lonlat(true);
        assert_eq!(cc, lonlats.slice(s![2, ..]).to_owned());

        let d = VectorPoint {
            xyz: xyz.slice(s![3, ..]).to_owned(),
        };
        let dc = d.to_lonlat(true);
        assert_eq!(dc, lonlats.slice(s![3, ..]).to_owned());

        let abcd = MultiVectorPoint { xyz };
        let abcdc = abcd.to_lonlats(true);
        assert_eq!(abcdc, lonlats);
        assert_eq!(
            abcdc,
            stack(Axis(0), &[ac.view(), bc.view(), cc.view(), dc.view()]).unwrap()
        )
    }

    #[test]
    fn test_distance() {
        let xyz = array![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];

        let a = VectorPoint::try_from(xyz.slice(s![0, ..]).to_owned()).unwrap();
        let b = VectorPoint::try_from(xyz.slice(s![1, ..]).to_owned()).unwrap();
        let c = VectorPoint::try_from(xyz.slice(s![2, ..]).to_owned()).unwrap();
        let d = VectorPoint::try_from(xyz.slice(s![3, ..]).to_owned()).unwrap();

        let ab = MultiVectorPoint::try_from(xyz.slice(s![..2, ..]).to_owned()).unwrap();
        let bc = MultiVectorPoint::try_from(xyz.slice(s![1..3, ..]).to_owned()).unwrap();
        let cd = MultiVectorPoint::try_from(xyz.slice(s![2.., ..]).to_owned()).unwrap();

        assert_eq!((&a).distance(&b), std::f64::consts::PI);
        assert_eq!((&b).distance(&c), std::f64::consts::PI / 2.);
        assert_eq!((&c).distance(&d), std::f64::consts::PI / 2.);

        assert_eq!((&a).distance(&a), 0.);

        assert_eq!((&ab).distance(&bc), 0.);
        assert_eq!((&bc).distance(&cd), 0.);
        assert_eq!((&ab).distance(&cd), std::f64::consts::PI / 2.);
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

        // assert_eq!(&a + &b, ab);
        // assert_eq!(&b + &c, bc);
        // assert_eq!(&c + &d, cd);
        // assert_eq!(&d + &a, da);
    }
}
