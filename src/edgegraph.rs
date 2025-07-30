use crate::geometry::GeometricPredicates;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Node {
    /// point in 3D Cartesian coordinates
    pub xyz: [f64; 3],
    /// indices of connected node indices, and geometry source(s) that that connection (edge) came from
    pub edges: HashMap<usize, Vec<usize>>,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        crate::sphericalpoint::xyz_eq(&self.xyz, &other.xyz)
    }
}

#[derive(Clone, Debug)]
/// graph of edges for performing operations (union, overlap, split)
/// on collections of higher-order geometries (arcstrings, polygons)
pub struct EdgeGraph<'a, G: crate::geometry::Geometry> {
    pub nodes: Vec<Node>,
    pub geometries: Vec<&'a G>,
}

impl<'a> Default for EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon> {
    fn default() -> Self {
        Self {
            nodes: vec![],
            geometries: vec![],
        }
    }
}

impl<'a> Default for EdgeGraph<'a, crate::arcstring::ArcString> {
    fn default() -> Self {
        Self {
            nodes: vec![],
            geometries: vec![],
        }
    }
}

impl<'a> From<&'a crate::sphericalpolygon::MultiSphericalPolygon>
    for EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon>
{
    fn from(polygons: &'a crate::sphericalpolygon::MultiSphericalPolygon) -> Self {
        let mut graph = Self::default();
        for polygon in polygons.polygons.iter() {
            graph.push(polygon);
        }
        graph
    }
}

impl<'a> From<Vec<&'a crate::sphericalpolygon::SphericalPolygon>>
    for EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon>
{
    fn from(polygons: Vec<&'a crate::sphericalpolygon::SphericalPolygon>) -> Self {
        let mut graph = Self::default();
        for polygon in polygons {
            graph.push(polygon);
        }
        graph
    }
}

impl<'a> From<&'a crate::arcstring::MultiArcString> for EdgeGraph<'a, crate::arcstring::ArcString> {
    fn from(arcstrings: &'a crate::arcstring::MultiArcString) -> Self {
        let mut graph = Self::default();
        for arcstring in arcstrings.arcstrings.iter() {
            graph.push(arcstring);
        }
        graph
    }
}

impl<'a> From<Vec<&'a crate::arcstring::ArcString>> for EdgeGraph<'a, crate::arcstring::ArcString> {
    fn from(arcstrings: Vec<&'a crate::arcstring::ArcString>) -> Self {
        let mut graph = Self::default();
        for arcstring in arcstrings {
            graph.push(arcstring);
        }
        graph
    }
}

type Edge = ((usize, usize), Vec<usize>);

impl<'a, G> EdgeGraph<'a, G>
where
    G: crate::geometry::Geometry + PartialEq,
{
    pub fn new_node(&mut self, xyz: &'a [f64; 3]) -> usize {
        // check if node already exists...
        if let Some(existing_index) = self
            .nodes
            .iter()
            .position(|existing_node| crate::sphericalpoint::xyz_eq(&existing_node.xyz, xyz))
        {
            existing_index
        } else {
            self.push_node(Node {
                xyz: xyz.to_owned(),
                edges: HashMap::<usize, Vec<usize>>::new(),
            });
            self.nodes.len() - 1
        }
    }

    pub fn push_edge(&mut self, a: usize, b: usize, sources: Vec<usize>) {
        let node_a = self.nodes.get_mut(a).unwrap();
        if let Some(entry) = node_a.edges.get_mut(&b) {
            entry.extend(&sources);
        } else {
            node_a.edges.insert(b, sources.to_owned());
        }

        let node_b = self.nodes.get_mut(b).unwrap();
        if let Some(entry) = node_b.edges.get_mut(&a) {
            entry.extend(&sources);
        } else {
            node_b.edges.insert(a, sources.to_owned());
        }
    }

    pub fn remove_edge(&mut self, a: usize, b: usize) -> Option<((usize, usize), Vec<usize>)> {
        let mut sources = vec![];

        let node_a = self.nodes.get_mut(a).unwrap();
        if let Some(node_a_sources) = node_a.edges.remove(&b) {
            sources.extend(node_a_sources);
        }

        let node_b = self.nodes.get_mut(b).unwrap();
        node_b.edges.remove(&a);
        if let Some(node_b_sources) = node_b.edges.remove(&a) {
            sources.extend(node_b_sources);
        }

        if sources.is_empty() {
            None
        } else {
            Some(((a, b), sources))
        }
    }

    pub fn push_node(&mut self, node: Node) {
        let num_nodes = self.nodes.len();
        let num_geometries = self.geometries.len();

        for (node_index, sources) in &node.edges {
            for geometry_index in sources {
                if geometry_index >= &num_geometries {
                    panic!(
                        "new node attempts to reference invalid index {geometry_index} (out of {num_geometries} geometries)"
                    );
                }
            }

            if node_index >= &num_nodes {
                panic!("new node attempts to reference invalid index {node_index} (out of {num_nodes} nodes)");
            }

            self.nodes[*node_index]
                .edges
                .insert(num_nodes, sources.to_owned());
        }

        self.nodes.push(node);
    }

    pub fn get_node_from_xyz(&self, xyz: &[f64; 3]) -> Option<usize> {
        for (index, node) in self.nodes.iter().enumerate() {
            if crate::sphericalpoint::xyz_eq(&node.xyz, xyz) {
                return Some(index);
            }
        }

        None
    }

    pub fn swap_remove_node(&mut self, index: usize) -> Node {
        // first, remove all references to the node
        for node in self.nodes.iter_mut() {
            node.edges.remove(&index);
        }

        // then, remove the node, swapping it with the last node to minimize index shifting
        let swapped_node_old_index = self.nodes.len() - 1;
        let removed = self.nodes.swap_remove(index);

        // then, replace all references to the last node with this new node index
        for node in self.nodes.iter_mut() {
            if let Some(sources) = node.edges.remove(&swapped_node_old_index) {
                node.edges.insert(index, sources);
            }
        }

        removed
    }

    /// remove edges sourced from more than one geometry
    /// the result may be disjunct
    pub fn remove_multisourced_edges(&mut self) -> Option<Vec<Edge>> {
        let mut removed = vec![];
        for (node_index, node) in self.nodes.iter_mut().enumerate() {
            for edge_node_index in node.edges.to_owned().into_keys() {
                if node.edges[&edge_node_index].len() > 1 {
                    if let Some(sources) = node.edges.remove(&edge_node_index) {
                        removed.push(((node_index, edge_node_index), sources))
                    }
                }
            }
        }

        if removed.is_empty() {
            None
        } else {
            Some(removed)
        }
    }

    /// remove edges sourced from one or less geometry
    /// the result may be disjunct
    pub fn remove_unisourced_edges(&mut self) -> Vec<((usize, usize), Vec<usize>)> {
        let mut removed = vec![];
        for (node_index, node) in self.nodes.iter_mut().enumerate() {
            for edge_node_index in node.edges.to_owned().into_keys() {
                if node.edges[&edge_node_index].len() <= 1 {
                    if let Some(sources) = node.edges.remove(&edge_node_index) {
                        removed.push(((node_index, edge_node_index), sources))
                    }
                }
            }
        }

        removed
    }

    /// remove edges of 0 length
    pub fn remove_degenerate_edges(&mut self) -> Vec<((usize, usize), Vec<usize>)> {
        let tolerance = 3e-11;
        let mut removed = vec![];
        for (node_index, node) in self.nodes.to_owned().iter().enumerate() {
            for edge_node_index in node.edges.keys() {
                if crate::sphericalpoint::xyzs_distance_over_sphere_radians(
                    &node.xyz,
                    &self.nodes[*edge_node_index].xyz,
                ) < tolerance
                {
                    if let Some(edge) = self.remove_edge(node_index, *edge_node_index) {
                        removed.push(edge);
                    }
                }
            }
        }

        removed
    }

    pub fn remove_orphaned_nodes(&mut self) -> Vec<Node> {
        let mut removed = vec![];
        // iterate over node list backwards to minimize index shuffling
        for node_index in (0..self.nodes.len()).rev() {
            if self.nodes[node_index].edges.is_empty() {
                removed.push(self.swap_remove_node(node_index));
            }
        }

        removed
    }

    // split all overlapping and intersecting edges
    pub fn split_edges(&mut self) -> bool {
        let mut changed = false;

        let nodes = self.nodes.to_owned();

        for _ in 0..self.nodes.len() {
            let mut updated_edges = false;
            let mut added_nodes = false;
            for (start_node_index, start_node) in nodes.iter().enumerate() {
                for (end_node_index, edge_sources) in start_node.edges.iter() {
                    let end_node = &nodes[*end_node_index];
                    // check if any other nodes lie on this edge...
                    for (middle_node_index, middle_node) in self.nodes.iter().enumerate() {
                        if crate::sphericalpoint::xyzs_collinear(
                            &start_node.xyz,
                            &middle_node.xyz,
                            &end_node.xyz,
                        ) {
                            // replace this edge with references to the middle node
                            self.remove_edge(start_node_index, *end_node_index);
                            self.push_edge(
                                start_node_index,
                                middle_node_index,
                                edge_sources.to_owned(),
                            );
                            self.push_edge(
                                middle_node_index,
                                *end_node_index,
                                edge_sources.to_owned(),
                            );
                            updated_edges = true;
                            break;
                        }
                    }

                    if updated_edges {
                        break;
                    }

                    // check if any other edges cross this edge...
                    for (other_start_node_index, other_start_node) in nodes.iter().enumerate() {
                        for (other_end_node_index, other_edge_sources) in
                            other_start_node.edges.iter()
                        {
                            let other_end_node = &self.nodes[*other_end_node_index];
                            if let Some(crossing) = crate::arcstring::xyz_two_arc_crossing(
                                (&start_node.xyz, &end_node.xyz),
                                (&other_start_node.xyz, &other_end_node.xyz),
                            ) {
                                // create a new node at the crossing with four edges
                                let crossing_node = Node {
                                    xyz: crossing,
                                    edges: HashMap::from([
                                        (start_node_index, edge_sources.to_owned()),
                                        (*end_node_index, edge_sources.to_owned()),
                                        (other_start_node_index, other_edge_sources.to_owned()),
                                        (*other_end_node_index, other_edge_sources.to_owned()),
                                    ]),
                                };

                                // send the new edges to the end of the edge list, to be analyzed again for further possible intersections
                                self.push_node(crossing_node);
                                added_nodes = true;
                                break;
                            }
                        }
                    }

                    if added_nodes {
                        break;
                    }
                }

                if updated_edges || added_nodes {
                    changed = true;
                    break;
                }
            }
        }

        changed
    }
}

impl<'a> EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon> {
    /// assign each edge to intersecting polygon(s)
    pub fn assign_polygons_to_edges(&mut self) {
        let mut pending_source_updates: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node_index, node) in self.nodes.iter().enumerate() {
            for (edge_node_index, sources) in &node.edges {
                let arcstring = crate::arcstring::ArcString::try_from(
                    crate::sphericalpoint::MultiSphericalPoint::try_from(vec![
                        node.xyz,
                        self.nodes[*edge_node_index].xyz,
                    ])
                    .unwrap(),
                )
                .unwrap();

                for (polygon_index, polygon) in self.geometries.iter().enumerate() {
                    if !sources.contains(&polygon_index) && polygon.intersects(&arcstring) {
                        pending_source_updates
                            .entry(node_index)
                            .or_insert_with(|| sources.to_owned());

                        pending_source_updates
                            .get_mut(&node_index)
                            .unwrap()
                            .push(polygon_index);
                    }
                }
            }
        }

        for source_update in pending_source_updates {
            todo!()
        }
    }
}

pub trait GeometryGraph<'a, G: crate::geometry::Geometry> {
    /// add a geometry's edges to the edge graph
    fn push(&mut self, geometry: &'a G);
}

impl<'a> GeometryGraph<'a, crate::sphericalpolygon::SphericalPolygon>
    for EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon>
{
    fn push(&mut self, polygon: &'a crate::sphericalpolygon::SphericalPolygon) {
        let node_indices = polygon
            .boundary
            .points
            .xyzs
            .iter()
            .map(|xyz| self.new_node(xyz))
            .collect::<Vec<usize>>();

        let geometry_index = if let Some(index) = self
            .geometries
            .iter()
            .position(|existing_polygon| existing_polygon == &polygon)
        {
            index
        } else {
            self.geometries.push(polygon);
            self.geometries.len() - 1
        };

        for xyz_index in 0..node_indices.len() {
            let node = self.nodes.get_mut(node_indices[xyz_index]).unwrap();

            let prev_node_index = node_indices[if xyz_index > 0 {
                xyz_index - 1
            } else {
                node_indices.len() - 1
            }];
            if let Some(edge) = node.edges.get_mut(&prev_node_index) {
                if !edge.contains(&geometry_index) {
                    edge.push(geometry_index);
                }
            } else {
                node.edges.insert(prev_node_index, vec![geometry_index]);
            }

            let next_node_index = node_indices[if xyz_index < node_indices.len() - 1 {
                xyz_index + 1
            } else {
                0
            }];
            if let Some(edge) = node.edges.get_mut(&next_node_index) {
                if !edge.contains(&geometry_index) {
                    edge.push(geometry_index);
                }
            } else {
                node.edges.insert(next_node_index, vec![geometry_index]);
            }
        }
    }
}

impl<'a> From<EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon>>
    for Vec<crate::sphericalpolygon::SphericalPolygon>
{
    fn from(graph: EdgeGraph<'a, crate::sphericalpolygon::SphericalPolygon>) -> Self {
        let mut polygons = vec![];
        let mut claimed_nodes = vec![];

        // depth-first search over edges
        for (node_index, node) in graph.nodes.iter().enumerate() {
            if !claimed_nodes.contains(&node_index) {
                // start a list of points that may potentially form a closed polygon boundary
                for (edge_node_index, edge_sources) in &node.edges {
                    polygons.extend(trace_polygons(
                        &graph.nodes,
                        &mut claimed_nodes,
                        &vec![node_index, *edge_node_index],
                        edge_sources,
                    ));
                }
            }
        }

        polygons
    }
}

impl<'a> GeometryGraph<'a, crate::arcstring::ArcString>
    for EdgeGraph<'a, crate::arcstring::ArcString>
{
    fn push(&mut self, arcstring: &'a crate::arcstring::ArcString) {
        let node_indices = arcstring
            .points
            .xyzs
            .iter()
            .map(|xyz| self.new_node(xyz))
            .collect::<Vec<usize>>();

        let geometry_index = if let Some(index) = self
            .geometries
            .iter()
            .position(|existing_arcstring| existing_arcstring == &arcstring)
        {
            index
        } else {
            self.geometries.push(arcstring);
            self.geometries.len() - 1
        };

        for xyz_index in 0..node_indices.len() - if arcstring.closed { 0 } else { 1 } {
            let node = self.nodes.get_mut(node_indices[xyz_index]).unwrap();

            if xyz_index > 0 || arcstring.closed {
                let prev_node_index = node_indices[if xyz_index > 0 {
                    xyz_index - 1
                } else {
                    node_indices.len() - 1
                }];
                if let Some(edge) = node.edges.get_mut(&prev_node_index) {
                    if !edge.contains(&geometry_index) {
                        edge.push(geometry_index);
                    }
                } else {
                    node.edges.insert(prev_node_index, vec![geometry_index]);
                }
            }

            let next_node_index = node_indices[if xyz_index < node_indices.len() - 1 {
                xyz_index + 1
            } else {
                0
            }];
            if let Some(edge) = node.edges.get_mut(&next_node_index) {
                if !edge.contains(&geometry_index) {
                    edge.push(geometry_index);
                }
            } else {
                node.edges.insert(next_node_index, vec![geometry_index]);
            }
        }
    }
}

impl<'a> From<EdgeGraph<'a, crate::arcstring::ArcString>> for Vec<crate::arcstring::ArcString> {
    fn from(value: EdgeGraph<'a, crate::arcstring::ArcString>) -> Self {
        todo!()
    }
}

fn trace_polygons(
    nodes: &Vec<Node>,
    claimed_node_indices: &mut Vec<usize>,
    polygon_node_indices: &Vec<usize>,
    edge_sources: &Vec<usize>,
) -> Vec<crate::sphericalpolygon::SphericalPolygon> {
    if nodes.is_empty() {
        vec![]
    } else {
        if polygon_node_indices.len() < 2 {
            panic!("must have at least 2 nodes populated in polygon node indices");
        }

        let mut polygon_node_indices = polygon_node_indices.to_owned();

        let working_node_index = polygon_node_indices[polygon_node_indices.len() - 1];
        let working_node = &nodes[working_node_index];

        let previous_node_index = polygon_node_indices[polygon_node_indices.len() - 2];
        let previous_node = &nodes[previous_node_index];

        polygon_node_indices.push(working_node_index);

        // to prevent crosses, compare the angles between this edge and the next candidate edges,
        // and choose between one of the two extremes (smallest and largest angles)
        let edge_angles = working_node
            .edges
            .iter()
            .filter_map(|(edge_node_index, edge_sources)| {
                if !claimed_node_indices.contains(edge_node_index) {
                    Some((
                        (edge_node_index, edge_sources),
                        crate::sphericalpoint::xyz_two_arc_angle_radians(
                            &previous_node.xyz,
                            &working_node.xyz,
                            &nodes[*edge_node_index].xyz,
                        ),
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<((&usize, &Vec<usize>), f64)>>();

        if edge_angles.len() > 1 {
            let left_node = edge_angles
                .iter()
                .min_by(|(_, edge_angle_a), (_, edge_angle_b)| {
                    edge_angle_a.partial_cmp(edge_angle_b).unwrap()
                })
                .unwrap()
                .0;
            let right_node = edge_angles
                .iter()
                .min_by(|(_, edge_angle_a), (_, edge_angle_b)| {
                    edge_angle_a.partial_cmp(edge_angle_b).unwrap()
                })
                .unwrap()
                .0;

            polygon_node_indices.push(
                *if left_node
                    .1
                    .iter()
                    .any(|source| edge_sources.contains(source))
                {
                    left_node
                } else {
                    right_node
                }
                .0,
            );

            trace_polygons(
                nodes,
                claimed_node_indices,
                &polygon_node_indices,
                edge_sources,
            )
        } else {
            // if the traced polygon is closed...
            if polygon_node_indices[0] == polygon_node_indices[polygon_node_indices.len() - 1] {
                vec![crate::sphericalpolygon::SphericalPolygon::try_new(
                    crate::arcstring::ArcString::try_from(
                        crate::sphericalpoint::MultiSphericalPoint::try_from(
                            polygon_node_indices
                                .iter()
                                .map(|node_index| nodes[*node_index].xyz)
                                .collect::<Vec<[f64; 3]>>(),
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                    None,
                )
                .unwrap()]
            } else {
                vec![]
            }
        }
    }
}
