use crate::{
    geometry::{Geometry, MultiGeometry},
    vectorpoint::{max_1darray, min_1darray, MultiVectorPoint},
};
use ndarray::{array, s, Array2};
use pyo3::pyclass;

/// collection of assorted geometries
#[pyclass(unsendable)]
pub struct GeometryCollection {
    pub geometries: Vec<Box<dyn Geometry>>,
}

impl GeometryCollection {
    pub fn empty() -> Self {
        Self { geometries: vec![] }
    }
}

impl Geometry for GeometryCollection {
    fn area(&self) -> f64 {
        self.geometries.iter().map(|geometry| geometry.area()).sum()
    }

    fn length(&self) -> f64 {
        self.geometries
            .iter()
            .map(|geometry| geometry.length())
            .sum()
    }

    fn bounds(&self, degrees: bool) -> [f64; 4] {
        let mut flat_bounds = vec![];

        self.geometries
            .iter()
            .for_each(|geometry| flat_bounds.extend(geometry.bounds(degrees)));

        let bounds = unsafe {
            Array2::<f64>::from_shape_vec((self.len(), 4), flat_bounds).unwrap_unchecked()
        };

        [
            min_1darray(&bounds.slice(s![.., 0])).unwrap(),
            min_1darray(&bounds.slice(s![.., 1])).unwrap(),
            max_1darray(&bounds.slice(s![.., 2])).unwrap(),
            max_1darray(&bounds.slice(s![.., 3])).unwrap(),
        ]
    }

    fn convex_hull(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        self.points().convex_hull()
    }

    fn points(&self) -> MultiVectorPoint {
        if self.len() > 0 {
            self.geometries
                .iter()
                .map(|geometry| geometry.points())
                .sum()
        } else {
            unsafe {
                MultiVectorPoint::try_from(array![[std::f64::NAN, std::f64::NAN, std::f64::NAN]])
                    .unwrap_unchecked()
            }
        }
    }
}

impl MultiGeometry for GeometryCollection {
    fn len(&self) -> usize {
        self.geometries.len()
    }
}
