use crate::arcstring::ArcString;
use crate::geometry::{GeometricOperations, Geometry};
use crate::sphericalpoint::MultiSphericalPoint;
use crate::sphericalpolygon::SphericalPolygon;
use ndarray::{s, stack, Array1, Axis};

pub trait Graphed<S: Geometry> {
    fn graph(&self) -> EdgeGraph<S>;
}

#[derive(Clone, Debug)]
pub struct Edge<'a, G: Geometry> {
    pub arc: ArcString,
    /// parent geometrie(s) that this edge belongs to
    pub parent_geometries: Vec<&'a G>,
}

impl<'a, G> PartialEq for Edge<'a, G>
where
    G: Geometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.arc == other.arc
    }
}

#[derive(Clone, Debug)]
/// graph of edges for performing operations (union, overlap, split)
/// on collections of higher-order geometries (arcstrings, polygons)
pub struct EdgeGraph<'a, G: Geometry> {
    pub edges: Vec<Edge<'a, G>>,
    pub geometries: Vec<&'a G>,
}

impl<'a> Default for EdgeGraph<'a, SphericalPolygon> {
    fn default() -> Self {
        Self {
            edges: vec![],
            geometries: vec![],
        }
    }
}

impl<'a> Default for EdgeGraph<'a, ArcString> {
    fn default() -> Self {
        Self {
            edges: vec![],
            geometries: vec![],
        }
    }
}

impl<'a> From<Vec<&'a SphericalPolygon>> for EdgeGraph<'a, SphericalPolygon> {
    fn from(geometries: Vec<&'a SphericalPolygon>) -> Self {
        let mut instance = Self::default();
        for geometry in geometries {
            instance.push(geometry);
        }
        instance
    }
}

impl<'a, G> EdgeGraph<'a, G>
where
    G: Geometry + PartialEq,
{
    /// remove edges belonging to more than one polygon
    /// the result may be disjunct
    pub fn delete_overlapping_edges(&mut self) -> bool {
        let mut changed = false;
        for edge_index in (0..self.edges.len()).rev() {
            if self.edges[edge_index].parent_geometries.len() > 1 {
                self.edges.remove(edge_index);
                changed = true;
            }
        }
        changed
    }

    /// remove edges NOT belonging to multiple parent geometries
    /// the result may be disjunct
    pub fn delete_nonoverlapping_edges(&mut self) -> bool {
        let mut changed = false;
        for edge_index in (0..self.edges.len()).rev() {
            if self.edges[edge_index].parent_geometries.len() <= 1 {
                self.edges.remove(edge_index);
                changed = true;
            }
        }
        changed
    }

    /// remove edges of 0 length
    pub fn delete_degenerate_edges(&mut self) -> bool {
        let mut changed = false;
        for edge_index in (0..self.edges.len()).rev() {
            if self.edges[edge_index].arc.length() == 0.0 {
                self.edges.remove(edge_index);
                changed = true;
            }
        }
        changed
    }

    pub fn add_edge(&mut self, arc: ArcString, geometries: Vec<&'a G>) {
        // check if edge already exists...
        for geometry in &geometries {
            if !self.geometries.contains(geometry) {
                self.geometries.push(geometry);
            }
        }
        if let Some(existing_edge_index) = self.edges.iter().position(|edge| arc == edge.arc) {
            if let Some(existing_edge) = self.edges.get_mut(existing_edge_index) {
                for geometry in &geometries {
                    if !existing_edge.parent_geometries.contains(geometry) {
                        existing_edge.parent_geometries.push(geometry);
                    }
                }
            }
        } else {
            self.edges.push(Edge {
                arc,
                parent_geometries: geometries,
            });
        }
    }
}

pub trait GeometryGraph<'a, G: Geometry> {
    /// add a geometry's edges to the edge graph
    fn push(&mut self, geometry: &'a G);

    /// trace edges to build disjoint geometries
    /// NOTE: this will erase overlapping polygons!
    fn extract_disjoint_geometries(&self) -> Vec<G>;

    // these are duplicated because I can't figure out how to coerce the GeometricOperations trait in the impl
    fn join_edges(&mut self) -> bool;
    fn overlap_edges(&mut self) -> bool;
    fn split_edges(&mut self) -> bool;
}

impl<'a> GeometryGraph<'a, SphericalPolygon> for EdgeGraph<'a, SphericalPolygon> {
    fn push(&mut self, geometry: &'a SphericalPolygon) {
        for index in 0..geometry.boundary.points.xyz.nrows() {
            let arc = ArcString::try_from(
                MultiSphericalPoint::try_from(stack![
                    Axis(0),
                    geometry.boundary.points.xyz.slice(s![index, ..]),
                    geometry.boundary.points.xyz.slice(s![
                        if index < geometry.boundary.points.xyz.nrows() - 1 {
                            index + 1
                        } else {
                            // close polygon to initial point
                            0
                        },
                        ..
                    ])
                ])
                .unwrap(),
            )
            .unwrap();

            self.add_edge(arc, vec![geometry]);
        }
    }

    fn extract_disjoint_geometries(&self) -> Vec<SphericalPolygon> {
        let mut polygons = vec![];
        let mut edges = self.edges.to_owned();

        // depth-first search over edges
        while let Some(starting_edge) = edges.pop() {
            // start a list of points that may potentially form a closed polygon boundary
            let mut points = starting_edge.arc.points.xyz.to_owned();

            // the preferred parent should be the last of the list of parent geometries of the starting edge
            let preferred_parent =
                starting_edge.parent_geometries[starting_edge.parent_geometries.len() - 1];

            // track the working end of the points
            let mut working_end = points.slice(s![points.nrows() - 1, ..]).to_owned();

            // go backwards to ease arbitrary removal of edges
            for other_edge_index in (0..edges.len()).rev() {
                let other_edge = &edges[other_edge_index];
                let a = other_edge.arc.points.xyz.slice(s![0, ..]);
                let b = other_edge.arc.points.xyz.slice(s![1, ..]);

                let tolerance = 2e-8;
                if (&working_end - &a).abs().sum() < tolerance
                    || (&working_end - &b).abs().sum() < tolerance
                {
                    // in order to prevent crosses, only add edges that satisfy the following criteria:
                    // 1. belongs to the preferred geometry of the current edge, and
                    // 2. either
                    //   a. only has ONE parent geometry, or
                    //   b. the preferred geometry of the working edge is NOT the ORIGINAL geometry (first parent) of the other edge
                    if other_edge.parent_geometries.contains(&preferred_parent)
                        && (other_edge.parent_geometries.len() == 1
                            || other_edge.parent_geometries[0] != preferred_parent)
                    {
                        // retrieve the correct end of the edge
                        working_end = if (&working_end - &b).abs().sum() < 2e-8 {
                            a
                        } else {
                            b
                        }
                        .to_owned();

                        // remove the edge from consideration in future polygons
                        edges.remove(other_edge_index);

                        // check if the boundary loops back to the starting point
                        if (&working_end - &points.slice(s![0, ..])).abs().sum() < tolerance {
                            polygons.push(
                                SphericalPolygon::new(
                                    ArcString {
                                        points: MultiSphericalPoint::try_from(points).unwrap(),
                                        closed: true,
                                    },
                                    None,
                                )
                                .unwrap(),
                            );
                            break;
                        } else {
                            points.push_row(working_end.view()).unwrap();
                        }
                    }
                }
            }
        }

        polygons
    }

    fn join_edges(&mut self) -> bool {
        let mut changed = self.split_edges();
        changed = self.delete_overlapping_edges() || changed;
        self.delete_degenerate_edges() || changed
    }

    fn overlap_edges(&mut self) -> bool {
        let mut changed = self.split_edges();
        changed = self.delete_nonoverlapping_edges() || changed;
        self.delete_degenerate_edges() || changed
    }

    fn split_edges(&mut self) -> bool {
        let mut changed = false;
        let mut edges: Vec<Edge<'a, SphericalPolygon>> = vec![];
        while let Some(edge_a) = self.edges.pop() {
            let mut crossed = false;
            for edge_b in &edges {
                // check if edge intersects any points...
                let vertices = edge_b.arc.vertices();
                if let Some(vertices) = edge_a.arc.intersection(&vertices) {
                    let arcstrings = crate::arcstring::split_arc_at_points(
                        &edge_a.arc.points.xyz.view(),
                        &vertices.xyz.view(),
                    );

                    // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                    self.edges.extend(arcstrings.iter().map(|arc| {
                        Edge {
                            arc: ArcString::try_from(
                                MultiSphericalPoint::try_from(arc.to_owned()).unwrap(),
                            )
                            .unwrap(),
                            parent_geometries: edge_a.parent_geometries.to_owned(),
                        }
                    }));
                    crossed = true;
                    changed = true;
                    break;
                // check if edge intersects any other edges...
                } else if let Some(intersection) = edge_a.arc.intersection(&edge_b.arc) {
                    let crossing: Array1<f64> = intersection.xyz.slice(s![0, ..]).to_owned();
                    // create four new edges
                    let mut edge_a1 = Edge {
                        arc: ArcString::try_from(
                            MultiSphericalPoint::try_from(stack![
                                Axis(0),
                                edge_a.arc.points.xyz.slice(s![0, ..]),
                                crossing
                            ])
                            .unwrap(),
                        )
                        .unwrap(),
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_a2 = Edge {
                        arc: ArcString::try_from(
                            MultiSphericalPoint::try_from(stack![
                                Axis(0),
                                crossing,
                                edge_a.arc.points.xyz.slice(s![1, ..]),
                            ])
                            .unwrap(),
                        )
                        .unwrap(),
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_b1 = Edge {
                        arc: ArcString::try_from(
                            MultiSphericalPoint::try_from(stack![
                                Axis(0),
                                edge_b.arc.points.xyz.slice(s![0, ..]),
                                crossing
                            ])
                            .unwrap(),
                        )
                        .unwrap(),
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };
                    let mut edge_b2 = Edge {
                        arc: ArcString::try_from(
                            MultiSphericalPoint::try_from(stack![
                                Axis(0),
                                crossing,
                                edge_b.arc.points.xyz.slice(s![1, ..]),
                            ])
                            .unwrap(),
                        )
                        .unwrap(),
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };

                    // assign parent polygons that each edge intersects with
                    for source in &edge_a.parent_geometries {
                        if edge_a1.arc.crosses(*source) {
                            edge_a1.parent_geometries.push(source);
                        }
                        if edge_a2.arc.crosses(*source) {
                            edge_a2.parent_geometries.push(source);
                        }
                    }
                    for source in &edge_b.parent_geometries {
                        if edge_b1.arc.crosses(*source) {
                            edge_b1.parent_geometries.push(source);
                        }
                        if edge_b2.arc.crosses(*source) {
                            edge_b2.parent_geometries.push(source);
                        }
                    }

                    // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                    self.edges
                        .extend_from_slice(&[edge_a1, edge_a2, edge_b1, edge_b2]);
                    crossed = true;
                    changed = true;
                    break;
                }
            }

            // if the edge did not intersect any other egdes or vertices, add it unchanched to the edge list
            if !crossed {
                edges.push(edge_a);
            }
        }

        self.edges = edges;

        changed
    }
}

impl<'a> GeometryGraph<'a, ArcString> for EdgeGraph<'a, ArcString> {
    fn push(&mut self, geometry: &'a ArcString) {
        todo!()
    }

    fn extract_disjoint_geometries(&self) -> Vec<ArcString> {
        todo!()
    }

    fn join_edges(&mut self) -> bool {
        todo!()
    }

    fn overlap_edges(&mut self) -> bool {
        todo!()
    }

    fn split_edges(&mut self) -> bool {
        todo!()
    }
}
