extern crate numpy;

use numpy::ndarray::{array, linspace, stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyType};

#[pymodule(name = "sphersgeo")]
pub mod sphersgeo {
    use super::*;

    #[pyclass]
    #[derive(Clone)]
    pub struct SphericalPoint {
        pub xyz: Array1<f64>,
    }

    impl SphericalPoint {
        pub fn from_lonlat(lonlat: &ArrayView1<f64>, degrees: bool) -> Self {
            let lonlat = if degrees {
                lonlat.to_radians()
            } else {
                lonlat.to_owned()
            };

            return Self {
                xyz: array![
                    lonlat[0].cos() * lonlat[1].cos(),
                    lonlat[0].sin() * lonlat[1].cos(),
                    lonlat[1].sin(),
                ],
            };
        }

        pub fn to_lonlat(&self, degrees: bool) -> Array1<f64> {
            let lonlat = array![
                self.xyz[0].atan2(self.xyz[1]),
                self.xyz[2].atan2((self.xyz[0].powf(2.0) + self.xyz[1].powf(2.0)).sqrt())
            ];
            return if degrees { lonlat.to_degrees() } else { lonlat };
        }

        pub fn normalized(&self) -> Self {
            Self {
                xyz: &self.xyz / self.xyz.pow2().sum().sqrt(),
            }
        }

        pub fn cross_product(&self, other: &Self) -> Self {
            Self {
                xyz: array![
                    self.xyz[1] * other.xyz[2] - self.xyz[2] * other.xyz[1],
                    self.xyz[2] * other.xyz[0] - self.xyz[0] * other.xyz[2],
                    self.xyz[0] * other.xyz[1] - self.xyz[1] * other.xyz[0]
                ],
            }
        }

        pub fn rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            let theta = if degrees { theta.to_radians() } else { theta };

            let a = self.normalized().xyz;
            let ax = a[0];
            let ay = a[1];
            let az = a[2];

            let b = other.normalized().xyz;
            let bx = b[0];
            let by = b[1];
            let bz = b[2];

            Self {
                xyz: -&b * -&a * &b * (1.0 - theta.cos())
                    + &a * theta.cos()
                    + array![
                        -&bz * &ay + &by * &az,
                        &bz * &ax - &bx * &az,
                        -&by * &ax - &bx * &ay,
                    ] * theta.sin(),
            }
        }
    }

    #[pymethods]
    impl SphericalPoint {
        #[new]
        fn new(xyz: PyReadonlyArray1<f64>) -> Self {
            Self {
                xyz: xyz.as_array().to_owned(),
            }
        }

        #[getter]
        fn xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        /// spherical point on the unit sphere corresponding to the given coordinates
        #[classmethod]
        #[pyo3(name="from_lonlat", signature=(lonlat,degrees=true))]
        fn py_from_lonlat<'py>(
            cls: &Bound<'_, PyType>,
            lonlat: PyReadonlyArray1<'py, f64>,
            degrees: bool,
        ) -> Self {
            Self::from_lonlat(&lonlat.as_array(), degrees)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name="to_lonlat", signature=(degrees=true))]
        fn py_to_lonlat<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray1<f64>> {
            self.to_lonlat(degrees).into_pyarray(py)
        }

        /// normalize to length 1 (the unit sphere) while maintaining direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// rotate by theta angle around another vector
        #[pyo3(name = "rotate_around", signature=(other,theta,degrees=true))]
        fn py_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.rotate_around(other, theta, degrees)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct SphericalPoints {
        pub xyz: Array2<f64>,
    }

    impl SphericalPoints {
        pub fn from_lonlats(lonlats: &ArrayView2<f64>, degrees: bool) -> Self {
            let lonlat = if degrees {
                lonlats.to_radians()
            } else {
                lonlats.to_owned()
            };
            let lon = lonlat.index_axis(Axis(1), 0);
            let lat = lonlat.index_axis(Axis(1), 1);

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

        pub fn points(&self) -> Vec<SphericalPoint> {
            self.xyz
                .rows()
                .into_iter()
                .map(|row| SphericalPoint {
                    xyz: row.to_owned(),
                })
                .collect::<Vec<SphericalPoint>>()
        }

        pub fn to_lonlats(&self, degrees: bool) -> Array2<f64> {
            let lonlats = stack(
                Axis(1),
                &[
                    Zip::from(self.xyz.rows())
                        .par_map_collect(|vector| vector[0].atan2(vector[1]))
                        .view(),
                    Zip::from(self.xyz.rows())
                        .par_map_collect(|vector| {
                            vector[2].atan2((vector[0].powf(2.0) + vector[1].powf(2.0)).sqrt())
                        })
                        .view(),
                ],
            )
            .unwrap();
            return if degrees {
                lonlats.to_degrees()
            } else {
                lonlats
            };
        }

        pub fn normalized(&self) -> Self {
            Self {
                xyz: &self.xyz / self.xyz.pow2().sum_axis(Axis(1)).sqrt(),
            }
        }

        pub fn cross_product(&self, other: &Self) -> Self {
            Self {
                xyz: stack(
                    Axis(0),
                    &[
                        (&self.xyz.index_axis(Axis(1), 1) * &other.xyz.index_axis(Axis(1), 2)
                            - &self.xyz.index_axis(Axis(1), 2)
                            - &other.xyz.index_axis(Axis(1), 1))
                            .view(),
                        (&self.xyz.index_axis(Axis(1), 2) * &other.xyz.index_axis(Axis(1), 0)
                            - &self.xyz.index_axis(Axis(1), 0) * &other.xyz.index_axis(Axis(1), 2))
                            .view(),
                        (&self.xyz.index_axis(Axis(1), 0) * &other.xyz.index_axis(Axis(1), 1)
                            - &self.xyz.index_axis(Axis(1), 1) * &other.xyz.index_axis(Axis(1), 0))
                            .view(),
                    ],
                )
                .unwrap(),
            }
        }

        pub fn rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            let theta = if degrees { theta.to_radians() } else { theta };

            let a = self.normalized().xyz;
            let ax = a.index_axis(Axis(1), 0);
            let ay = a.index_axis(Axis(1), 1);
            let az = a.index_axis(Axis(1), 2);

            let b = other.normalized().xyz;
            let bx = b.index_axis(Axis(1), 0);
            let by = b.index_axis(Axis(1), 1);
            let bz = b.index_axis(Axis(1), 2);

            Self {
                xyz: -&b * -&a * &b * (1.0 - theta.cos())
                    + &a * theta.cos()
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
    }

    #[pymethods]
    impl SphericalPoints {
        #[new]
        fn new(xyz: PyReadonlyArray2<f64>) -> Self {
            Self {
                xyz: xyz.as_array().to_owned(),
            }
        }

        #[getter]
        fn xyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            self.xyz.to_owned().into_pyarray(py)
        }

        #[getter]
        #[pyo3(name = "points")]
        fn py_points(&self) -> Vec<SphericalPoint> {
            self.points()
        }

        /// spherical points on the unit sphere corresponding to the given coordinates
        #[classmethod]
        #[pyo3(name="from_lonlats", signature=(lonlats,degrees=true))]
        fn py_from_lonlats<'py>(
            cls: &Bound<'_, PyType>,
            lonlats: PyReadonlyArray2<'py, f64>,
            degrees: bool,
        ) -> Self {
            Self::from_lonlats(&lonlats.as_array(), degrees)
        }

        /// convert to angle coordinates along the sphere
        #[pyo3(name="to_lonlats", signature=(degrees=true))]
        fn py_to_lonlats<'py>(&self, py: Python<'py>, degrees: bool) -> Bound<'py, PyArray2<f64>> {
            self.to_lonlats(degrees).into_pyarray(py)
        }

        /// normalize to length 1 (the unit sphere) while maintaining direction
        #[getter]
        #[pyo3(name = "normalized")]
        fn py_normalized(&self) -> Self {
            self.normalized()
        }

        /// rotate by theta angle around other vectors
        #[pyo3(name="rotate_around", signature=(other,theta,degrees=true))]
        fn py_rotate_around(&self, other: &Self, theta: f64, degrees: bool) -> Self {
            self.rotate_around(other, theta, degrees)
        }
    }

    #[pymodule(name = "great_circle")]
    pub mod great_circle {
        use super::{SphericalPoint, SphericalPoints, *};

        #[pyclass]
        #[derive(Clone)]
        struct GreatCircleArc {
            a: SphericalPoint,
            b: SphericalPoint,
        }

        impl GreatCircleArc {
            pub fn subtends(&self) -> f64 {
                self.a.xyz.dot(&self.b.xyz).acos()
            }

            pub fn intersects(&self, other: &Self) -> bool {
                // TODO: write an intersects algorithm
                self.intersection(other).is_some()
            }

            pub fn contains(&self, point: &SphericalPoint) -> bool {
                // TODO: write a better algorithm; this one is not rigorous
                let left_subtend = GreatCircleArc {
                    a: self.a.to_owned(),
                    b: point.to_owned(),
                }
                .subtends();
                let right_subtend = GreatCircleArc {
                    a: point.to_owned(),
                    b: self.b.to_owned(),
                }
                .subtends();

                let angle = angles(
                    &SphericalPoints {
                        xyz: self.a.xyz.to_shape((1, 3)).unwrap().to_owned(),
                    },
                    &SphericalPoints {
                        xyz: point.xyz.to_shape((1, 3)).unwrap().to_owned(),
                    },
                    &SphericalPoints {
                        xyz: self.b.xyz.to_shape((1, 3)).unwrap().to_owned(),
                    },
                    false,
                )[0];

                left_subtend + right_subtend - self.subtends() < 3e-11
                    && angle == std::f64::consts::PI
            }

            pub fn intersection(&self, other: &Self) -> Option<SphericalPoint> {
                // TODO: implement
                None
            }

            pub fn midpoint(&self) -> SphericalPoint {
                SphericalPoint {
                    xyz: (&self.a.xyz + &self.b.xyz) / 2.0,
                }
                .normalized()
            }

            pub fn interpolate_points(&self, n: usize) -> SphericalPoints {
                let n = if n < 2 { 2 } else { n };
                let t = Array1::<f64>::from_iter(linspace(0.0, 1.0, n));
                let t = t.to_shape((n, 1)).unwrap();

                let omega = self.subtends();

                SphericalPoints {
                    xyz: &self.a.xyz
                        * ((Zip::from(&t).par_map_collect(|t| 1.0 - t) * omega).sin()
                            / omega.sin())
                        + &self.b.xyz * ((t * omega).sin() / omega.sin()),
                }
            }
        }

        #[pymethods]
        impl GreatCircleArc {
            #[new]
            fn new(a: &SphericalPoint, b: &SphericalPoint) -> Self {
                Self {
                    a: a.to_owned(),
                    b: b.to_owned(),
                }
            }

            #[getter]
            fn a(&self) -> SphericalPoint {
                self.a.to_owned()
            }

            #[getter]
            fn b(&self) -> SphericalPoint {
                self.b.to_owned()
            }

            /// radians subtended by the arc on the unit sphere
            #[getter]
            #[pyo3(name = "subtends")]
            fn py_subtends(&self) -> f64 {
                self.subtends()
            }

            /// whether this arc and another given arc intersect
            #[pyo3(name = "intersects")]
            fn py_intersects(&self, other: &Self) -> bool {
                self.intersects(other)
            }

            /// whether this arc and a given point intersect
            #[pyo3(name = "contains")]
            fn py_contains(&self, point: &SphericalPoint) -> bool {
                self.contains(point)
            }

            /// point at which this arc and another given arc intersect
            #[pyo3(name = "intersection", signature=(other))]
            fn py_intersection(&self, other: &Self) -> Option<SphericalPoint> {
                self.intersection(other)
            }

            /// midpoint of this arc
            #[getter]
            #[pyo3(name = "midpoint")]
            fn py_midpoint(&self) -> SphericalPoint {
                self.midpoint()
            }

            /// interpolate the a given number of points along this arc
            #[pyo3(name="interpolate_points", signature=(n=50))]
            fn py_interpolate_points(&self, n: usize) -> SphericalPoints {
                self.interpolate_points(n)
            }
        }

        #[pyclass]
        #[derive(Clone)]
        struct GreatCircleArcs {
            a: SphericalPoints,
            b: SphericalPoints,
        }

        impl GreatCircleArcs {
            pub fn arcs(&self) -> Vec<GreatCircleArc> {
                let mut a = self.a.xyz.rows().into_iter();
                let mut b = self.b.xyz.rows().into_iter();

                let mut arcs = vec![];
                for _ in 0..self.a.xyz.nrows() {
                    arcs.push(GreatCircleArc {
                        a: SphericalPoint {
                            xyz: a.next().unwrap().to_owned(),
                        },
                        b: SphericalPoint {
                            xyz: b.next().unwrap().to_owned(),
                        },
                    })
                }

                arcs
            }

            pub fn subtends(&self) -> Array1<f64> {
                Zip::from(self.a.xyz.rows())
                    .and(self.b.xyz.rows())
                    .par_map_collect(|a, b| a.dot(&b).acos())
            }

            pub fn intersects(&self, other: &Self) -> bool {
                self.intersections(other).xyz.nrows() > 0
            }

            pub fn contains(&self, points: &SphericalPoints) -> bool {
                // TODO: implement
                true
            }

            pub fn intersections(&self, other: &Self) -> SphericalPoints {
                // TODO: implement
                SphericalPoints {
                    xyz: Array2::<f64>::zeros((1, 3)),
                }
            }

            pub fn midpoints(&self) -> SphericalPoints {
                SphericalPoints {
                    xyz: (&self.a.xyz + &self.b.xyz) / 2.0,
                }
                .normalized()
            }

            pub fn interpolate_points(&self, n: usize) -> Vec<SphericalPoints> {
                let n = if n < 2 { 2 } else { n };

                self.arcs()
                    .iter()
                    .map(|arc| arc.interpolate_points(n))
                    .collect()
            }
        }

        #[pymethods]
        impl GreatCircleArcs {
            #[new]
            fn new(a: &SphericalPoints, b: &SphericalPoints) -> Self {
                Self {
                    a: a.to_owned(),
                    b: b.to_owned(),
                }
            }

            #[getter]
            fn a(&self) -> SphericalPoints {
                self.a.to_owned()
            }

            #[getter]
            fn b(&self) -> SphericalPoints {
                self.b.to_owned()
            }

            #[getter]
            #[pyo3(name = "arcs")]
            fn py_arcs(&self) -> Vec<GreatCircleArc> {
                self.arcs()
            }

            /// angular distances subtended by the arcs on the unit sphere
            #[getter]
            #[pyo3(name = "subtends")]
            fn py_subtends<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                self.subtends().to_pyarray(py)
            }

            /// whether these arcs and other given arcs intersect
            #[pyo3(name = "intersects", signature=(other))]
            fn py_intersects(&self, other: &Self) -> bool {
                self.intersects(other)
            }

            /// whether these arcs and the given points intersect
            #[pyo3(name = "contains")]
            fn py_contains(&self, points: &SphericalPoints) -> bool {
                self.contains(points)
            }

            /// points at which these arcs and the other given arcs intersect
            #[pyo3(name = "intersection")]
            fn py_intersections(&self, other: &Self) -> SphericalPoints {
                self.intersections(other)
            }

            /// midpoints of these arcs
            #[getter]
            #[pyo3(name = "midpoints")]
            fn py_midpoints(&self) -> SphericalPoints {
                self.midpoints()
            }

            /// interpolate the given number of points along these arcs
            #[pyo3(name="interpolate_points", signature=(n=50))]
            fn py_interpolate_points(&self, n: usize) -> Vec<SphericalPoints> {
                self.interpolate_points(n)
            }
        }

        pub fn angles(
            a: &SphericalPoints,
            b: &SphericalPoints,
            c: &SphericalPoints,
            degrees: bool,
        ) -> Array1<f64> {
            let abx = a.cross_product(&b).normalized();
            let bcx = c.cross_product(&b).normalized();
            let x = abx.cross_product(&bcx).normalized();

            let diff = (&b.xyz * &x.xyz).sum_axis(Axis(1));
            let mut inner = (&abx.xyz * &bcx.xyz).sum_axis(Axis(1));
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

        /// given spherical points A, B, and C, retrieve the angle at B between AB and BC
        ///
        /// References:
        /// - Miller, Robert D. Computing the area of a spherical polygon. Graphics Gems IV. 1994. Academic Press. doi:10.5555/180895.180907
        #[pyfunction]
        #[pyo3(name = "angles", signature=(a,b,c,degrees=true))]
        fn py_angles<'py>(
            py: Python<'py>,
            a: SphericalPoints,
            b: SphericalPoints,
            c: SphericalPoints,
            degrees: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            angles(&a, &b, &c, degrees).to_pyarray(py)
        }
    }

    #[pymodule(name = "polygon")]
    pub mod polygon {
        use super::*;

        #[pyclass]
        pub struct SphericalPolygon {}

        #[pyclass]
        pub struct SphericalMultiPolygon {}
    }

    #[pymodule(name = "graph")]
    pub mod graph {
        use super::*;

        #[pyclass]
        pub struct Graph {
            polygons: polygon::SphericalPolygon,
        }
    }
}
