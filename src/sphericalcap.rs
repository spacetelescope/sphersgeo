use crate::{
    arcstring::{xyz_two_arc_crossing, ArcString, MultiArcString},
    edgegraph::EdgeGraph,
    geometry::{
        GeometricOperations, GeometricRelationships, Geometry, GeometryCollection, MultiGeometry,
    },
    sphericalpoint::{
        xyz_add_xyz, xyz_cross, xyz_div_f64, xyz_mul_xyz, xyz_sub_xyz, xyz_sum,
        xyz_two_arc_angle_radians, xyzs_mean, xyzs_sum, MultiSphericalPoint, SphericalPoint,
    },
};
use std::{
    cmp::Ordering,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign},
};

#[cfg(feature = "py")]
use pyo3::prelude::*;

/// cap over the sphere, comprised of a center point and a radius angle
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalCap {
    pub center: SphericalPoint,
    pub radius: f64,
}

impl Display for SphericalCap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalCap({}, {})", self.center, self.radius)
    }
}

impl SphericalCap {}

impl Geometry for SphericalCap {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        crate::sphericalpoint::MultiSphericalPoint::try_from(vec![self.center.to_owned()]).unwrap()
    }

    fn boundary(&self) -> Option<crate::sphericalpolygon::SphericalPolygon> {
        todo!()
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.center.to_owned()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        self.center.to_owned()
    }

    fn area(&self) -> f64 {
        todo!()
    }

    fn length(&self) -> f64 {
        self.radius * 2.0
    }
}

impl GeometricRelationships<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricRelationships<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricRelationships<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricRelationships<crate::sphericalpoint::SphericalPoint> for SphericalCap {}
impl GeometricOperations<crate::sphericalpoint::SphericalPoint> for SphericalCap {}

/// cap over the sphere, comprised of a center point and a radius angle
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone, Debug)]
pub struct MultiSphericalCap {
    pub caps: Vec<SphericalCap>,
}

impl TryFrom<Vec<SphericalCap>> for MultiSphericalCap {
    type Error = String;

    fn try_from(caps: Vec<SphericalCap>) -> Result<Self, Self::Error> {
        if !caps.is_empty() {
            Ok(Self { caps: caps })
        } else {
            Err(String::from("no caps provided"))
        }
    }
}

impl Display for MultiSphericalCap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiSphericalCap({:?})", self.caps)
    }
}

impl PartialEq for MultiSphericalCap {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for cap in &self.caps {
            if !other.caps.contains(cap) {
                return false;
            }
        }

        true
    }
}

impl PartialEq<Vec<SphericalCap>> for MultiSphericalCap {
    fn eq(&self, other: &Vec<SphericalCap>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for cap in &self.caps {
            if !other.contains(cap) {
                return false;
            }
        }

        true
    }
}

impl Add<Self> for &MultiSphericalCap {
    type Output = MultiSphericalCap;

    fn add(self, rhs: Self) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&Self> for MultiSphericalCap {
    fn add_assign(&mut self, other: &Self) {
        self.extend(other.to_owned());
    }
}

impl Add<&SphericalCap> for &MultiSphericalCap {
    type Output = MultiSphericalCap;

    fn add(self, rhs: &SphericalCap) -> Self::Output {
        let mut owned = self.to_owned();
        owned += rhs;
        owned
    }
}

impl AddAssign<&SphericalCap> for MultiSphericalCap {
    fn add_assign(&mut self, other: &SphericalCap) {
        self.push(other.to_owned());
    }
}

impl Geometry for MultiSphericalCap {
    fn vertices(&self) -> crate::sphericalpoint::MultiSphericalPoint {
        self.caps.iter().map(|geometry| geometry.vertices()).sum()
    }

    fn boundary(&self) -> Option<MultiArcString> {
        let arcstrings: Vec<ArcString> = self
            .caps
            .iter()
            .filter_map(|polygon| polygon.boundary())
            .collect();
        MultiArcString::try_from(arcstrings).ok()
    }

    fn representative(&self) -> crate::sphericalpoint::SphericalPoint {
        self.caps[0].representative()
    }

    fn centroid(&self) -> crate::sphericalpoint::SphericalPoint {
        MultiSphericalPoint::try_from(
            self.caps
                .iter()
                .map(|polygon| polygon.centroid().xyz)
                .collect::<Vec<[f64; 3]>>(),
        )
        .unwrap()
        .centroid()
    }

    fn convex_hull(&self) -> Option<SphericalPolygon> {
        self.vertices().convex_hull()
    }

    fn area(&self) -> f64 {
        self.caps.iter().map(|polygon| polygon.area()).sum()
    }

    fn length(&self) -> f64 {
        self.boundary().map_or(0.0, |boundary| boundary.length())
    }
}

impl MultiGeometry<SphericalCap> for MultiSphericalCap {
    fn len(&self) -> usize {
        self.caps.len()
    }

    fn extend(&mut self, other: Self) {
        self.caps.extend(other.caps);
    }

    fn push(&mut self, value: SphericalCap) {
        self.caps.push(value);
    }
}

impl Sum for MultiSphericalCap {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut caps = vec![];
        for multicap in iter {
            caps.extend(multicap.caps);
        }
        Self { caps }
    }
}
