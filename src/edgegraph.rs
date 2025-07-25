use crate::arcstring::ArcString;
use crate::geometry::{GeometricOperations, Geometry};
use crate::sphericalpoint::{xyz_eq, MultiSphericalPoint};
use crate::sphericalpolygon::SphericalPolygon;
use ndarray::{s, Array1};

pub trait ToGraph<S: Geometry> {
    fn to_graph(&self) -> EdgeGraph<S>;
}

#[derive(Clone, Debug)]
pub struct Edge<'a, G: Geometry> {
    pub a: &'a [f64; 3],
    pub b: &'a [f64; 3],
    /// parent geometrie(s) that this edge belongs to
    pub parent_geometries: Vec<&'a G>,
}

impl<'a, G> Edge<'a, G>
where
    G: Geometry,
{
    pub fn points(&self) -> Vec<&'a [f64; 3]> {
        vec![self.a, self.b]
    }

    pub fn arc(&self) -> ArcString {
        ArcString {
            points: MultiSphericalPoint::try_from(vec![self.a.to_owned(), self.b.to_owned()])
                .unwrap(),
            closed: false,
        }
    }
}

impl<'a, G> PartialEq for Edge<'a, G>
where
    G: Geometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.arc() == other.arc()
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
    fn from(polygons: Vec<&'a SphericalPolygon>) -> Self {
        let mut instance = Self::default();
        for polygon in polygons {
            instance.push(polygon);
        }
        instance
    }
}

impl<'a> From<Vec<&'a ArcString>> for EdgeGraph<'a, ArcString> {
    fn from(arcstrings: Vec<&'a ArcString>) -> Self {
        let mut instance = Self::default();
        for arcstring in arcstrings {
            instance.push(arcstring);
        }
        instance
    }
}

impl<'a, G> EdgeGraph<'a, G>
where
    G: Geometry + PartialEq,
{
    pub fn add_edge(&mut self, a: &'a [f64; 3], b: &'a [f64; 3], geometries: Vec<&'a G>) {
        for geometry in &geometries {
            if !self.geometries.contains(geometry) {
                self.geometries.push(geometry);
            }
        }

        // check if edge already exists...
        if let Some(existing_edge_index) = self.edges.iter().position(|edge| {
            (xyz_eq(a, edge.a) && xyz_eq(b, edge.b)) || (xyz_eq(a, edge.b) && xyz_eq(b, edge.a))
        }) {
            if let Some(existing_edge) = self.edges.get_mut(existing_edge_index) {
                for geometry in &geometries {
                    if !existing_edge.parent_geometries.contains(geometry) {
                        existing_edge.parent_geometries.push(geometry);
                    }
                }
            }
        } else {
            self.edges.push(Edge {
                a,
                b,
                parent_geometries: geometries,
            });
        }
    }

    /// remove edges belonging to more than one geometry
    /// the result may be disjunct
    pub fn prune_overlapping_edges(&mut self) -> bool {
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
    pub fn prune_nonoverlapping_edges(&mut self) -> bool {
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
    pub fn prune_degenerate_edges(&mut self) -> bool {
        let mut changed = false;
        for edge_index in (0..self.edges.len()).rev() {
            if self.edges[edge_index].arc().length() == 0.0 {
                self.edges.remove(edge_index);
                changed = true;
            }
        }
        changed
    }
}

impl<'a> EdgeGraph<'a, SphericalPolygon> {
    pub fn split_edges(&'a mut self) -> bool {
        let mut changed = false;
        let mut edges: Vec<Edge<'a, SphericalPolygon>> = vec![];
        while let Some(edge_a) = self.edges.pop() {
            let mut crossed = false;
            for edge_b in &edges {
                // check if edge intersects any points...
                let vertices = edge_b.arc().points;
                if crate::sphericalpoint::xyzs_collinear(edge_a.a, edge_b.a, edge_a.b)
                    || crate::sphericalpoint::xyzs_collinear(edge_a.a, edge_b.b, edge_a.b)
                {
                    let arcstrings = crate::arcstring::split_arc_at_points(
                        vec![edge_a.a, edge_a.b],
                        vec![edge_b.a, edge_b.b],
                    );

                    // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                    for arc in arcstrings {
                        self.add_edge(arc[0], arc[1], edge_a.parent_geometries.to_owned());
                    }
                    crossed = true;
                    changed = true;
                    break;
                // check if edge intersects any other edges...
                } else if let Some(intersection) = edge_a.arc().intersection(&edge_b.arc()) {
                    // assume intersection between two arcs will always only have 1 row
                    let crossing = intersection.xyzs[0];
                    // create four new edges
                    let mut edge_a1 = Edge {
                        a: edge_a.a,
                        b: &crossing,
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_a2 = Edge {
                        a: &crossing,
                        b: edge_a.b,
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_b1 = Edge {
                        a: edge_b.a,
                        b: &crossing,
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };
                    let mut edge_b2 = Edge {
                        a: &crossing,
                        b: edge_b.b,
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };

                    // assign parent geometries that each new edge is now within
                    for source in &edge_a.parent_geometries {
                        if edge_a1.arc().within(*source) {
                            edge_a1.parent_geometries.push(source);
                        }
                        if edge_a2.arc().within(*source) {
                            edge_a2.parent_geometries.push(source);
                        }
                    }
                    for source in &edge_b.parent_geometries {
                        if edge_b1.arc().within(*source) {
                            edge_b1.parent_geometries.push(source);
                        }
                        if edge_b2.arc().within(*source) {
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

impl<'a> EdgeGraph<'a, ArcString> {
    pub fn split_edges(&mut self) -> bool {
        let tolerance = 2e-8;
        let mut changed = false;
        let mut counter = 0;
        let mut edges: Vec<Edge<'a, ArcString>> = vec![];
        while let Some(edge_a) = self.edges.pop() {
            let mut crossed = false;
            for edge_b in &edges {
                // check if edge intersects any points...
                let mut splitting_points = vec![];

                for endpoint in [edge_b.a, edge_b.b] {
                    // skip if the point is equal to one of the endpoints
                    if xyz_eq(edge_a.a, endpoint)
                        && xyz_eq(endpoint, edge_a.b)
                        && crate::sphericalpoint::xyzs_collinear(edge_a.a, endpoint, edge_a.b)
                    {
                        splitting_points.push(endpoint);
                    }
                }

                if !splitting_points.is_empty() {
                    let arcstrings =
                        crate::arcstring::split_arc_at_points(edge_a.points(), splitting_points);

                    // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                    for arc in arcstrings {
                        self.add_edge(arc[0], arc[1], edge_a.parent_geometries.to_owned());
                    }
                    crossed = true;
                    changed = true;
                    break;
                // check if edge intersects any other edges...
                } else if let Some(intersection) = edge_a.arc().intersection(&edge_b.arc()) {
                    // assume intersection between two arcs will always only have 1 row
                    let crossing: Array1<f64> = intersection.xyz.slice(s![0, ..]).to_owned();

                    // create four new edges
                    let mut edge_a1 = Edge {
                        a: edge_a.a,
                        b: crossing,
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_a2 = Edge {
                        a: crossing,
                        b: edge_a.b,
                        parent_geometries: edge_a.parent_geometries.to_owned(),
                    };
                    let mut edge_b1 = Edge {
                        a: edge_b.a,
                        b: crossing,
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };
                    let mut edge_b2 = Edge {
                        a: crossing,
                        b: edge_b.b,
                        parent_geometries: edge_b.parent_geometries.to_owned(),
                    };

                    // assign parent geometries that each new edge is now within
                    for source in &edge_a.parent_geometries {
                        if edge_a1.arc().within(*source) {
                            edge_a1.parent_geometries.push(source);
                        }
                        if edge_a2.arc().within(*source) {
                            edge_a2.parent_geometries.push(source);
                        }
                    }
                    for source in &edge_b.parent_geometries {
                        if edge_b1.arc().within(*source) {
                            edge_b1.parent_geometries.push(source);
                        }
                        if edge_b2.arc().within(*source) {
                            edge_b2.parent_geometries.push(source);
                        }
                    }

                    // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                    self.add_edge(edge_a1.a, edge_a1.b, edge_a1.parent_geometries);
                    self.add_edge(edge_a2.a, edge_a2.b, edge_a2.parent_geometries);
                    self.add_edge(edge_b1.a, edge_b1.b, edge_b1.parent_geometries);
                    self.add_edge(edge_b2.a, edge_b2.b, edge_b2.parent_geometries);

                    crossed = true;
                    changed = true;
                    break;
                }
            }

            // if the edge did not intersect any other edges or vertices, add it unchanched to the edge list
            if !crossed {
                edges.push(edge_a);
            }

            counter += 1;
            if counter > 10 {
                break;
            }
        }

        self.edges = edges;

        changed
    }
}

pub trait GeometryGraph<'a, G: Geometry> {
    /// add a geometry's edges to the edge graph
    fn push(&mut self, geometry: &'a G);

    /// trace edges to build disjoint geometries
    /// NOTE: this will erase geometry overlap
    fn find_disjoint_geometries(&self) -> Vec<G>;
}

impl<'a> GeometryGraph<'a, SphericalPolygon> for EdgeGraph<'a, SphericalPolygon> {
    fn push(&mut self, polygon: &'a SphericalPolygon) {
        for arc in Vec::<ArcString>::from(&polygon.boundary) {
            let xyz = arc.points.xyz;
            self.add_edge(
                xyz.slice(s![0, ..]).to_owned(),
                xyz.slice(s![1, ..]).to_owned(),
                vec![polygon],
            );
        }
    }

    fn find_disjoint_geometries(&self) -> Vec<SphericalPolygon> {
        let mut polygons = vec![];
        let mut edges = self.edges.to_owned();

        let tolerance = 2e-8;

        // depth-first search over edges
        while let Some(starting_edge) = edges.pop() {
            // start a list of points that may potentially form a closed polygon boundary
            let mut points = starting_edge.points();

            // the preferred parent should be the last of the list of parent geometries of the starting edge
            let preferred_parent =
                starting_edge.parent_geometries[starting_edge.parent_geometries.len() - 1];

            // track the working end of the points
            let mut working_end = points.slice(s![points.nrows() - 1, ..]).to_owned();

            // go backwards to ease arbitrary removal of edges
            for other_edge_index in (0..edges.len()).rev() {
                let other_edge = &edges[other_edge_index];

                if (&working_end - &other_edge.a).abs().sum() < tolerance
                    || (&working_end - &other_edge.b).abs().sum() < tolerance
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
                        working_end = if (&working_end - &other_edge.b).abs().sum() < 2e-8 {
                            &other_edge.a
                        } else {
                            &other_edge.b
                        }
                        .to_owned();

                        // remove the edge from consideration in future polygons
                        edges.remove(other_edge_index);

                        // check if the boundary loops back to the starting point
                        if (&working_end - &points.slice(s![0, ..])).abs().sum() < tolerance {
                            polygons.push(
                                SphericalPolygon::new(
                                    ArcString::new(
                                        MultiSphericalPoint::try_from(points).unwrap(),
                                        true,
                                    )
                                    .unwrap(),
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
}

impl<'a> GeometryGraph<'a, ArcString> for EdgeGraph<'a, ArcString> {
    fn push(&mut self, arcstring: &'a ArcString) {
        for arc in Vec::<ArcString>::from(arcstring) {
            let xyz = arc.points.xyz;
            self.add_edge(
                xyz.slice(s![0, ..]).to_owned(),
                xyz.slice(s![1, ..]).to_owned(),
                vec![arcstring],
            );
        }
    }

    fn find_disjoint_geometries(&self) -> Vec<ArcString> {
        let mut arcstrings = vec![];
        let mut edges = self.edges.to_owned();

        let tolerance = 2e-8;

        // depth-first search over edges
        while let Some(starting_edge) = edges.pop() {
            // start a list of points that may potentially form a closed polygon boundary
            let mut points = starting_edge.points();

            // the preferred parent should be the last of the list of parent geometries of the starting edge
            let preferred_parent =
                starting_edge.parent_geometries[starting_edge.parent_geometries.len() - 1];
            let preferred_parent_a = preferred_parent.points.xyz.slice(s![0, ..]).to_owned();
            let preferred_parent_b = preferred_parent
                .points
                .xyz
                .slice(s![preferred_parent.points.xyz.nrows() - 1, ..])
                .to_owned();

            // track the working end of the points
            let mut working_end = points.slice(s![points.nrows() - 1, ..]).to_owned();

            // go backwards to ease arbitrary removal of edges
            for other_edge_index in (0..edges.len()).rev() {
                let other_edge = &edges[other_edge_index];

                if (&working_end - &other_edge.a).abs().sum() < tolerance
                    || (&working_end - &other_edge.b).abs().sum() < tolerance
                {
                    // retrieve the correct end of the edge
                    let point_to_add = if (&working_end - &other_edge.b).abs().sum() < 2e-8 {
                        &other_edge.a
                    } else {
                        &other_edge.b
                    }
                    .to_owned();

                    // avoid adding the same edge twice
                    if (&points.slice(s![points.nrows() - 2, ..]) - &point_to_add.view())
                        .abs()
                        .sum()
                        > 2e-8
                    {
                        // remove the edge from consideration in future polygons
                        edges.remove(other_edge_index);

                        points.push_row(point_to_add.view()).unwrap();
                        working_end = point_to_add;
                    }
                }
            }

            // check if the arcstring loops back to the starting point
            let closed = (&working_end - &points.slice(s![0, ..])).abs().sum() < tolerance;
            arcstrings.push(ArcString {
                points: MultiSphericalPoint::try_from(points).unwrap(),
                closed,
            });
        }

        arcstrings
    }
}
