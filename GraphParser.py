from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os


class EdgeDistanceImportance(Enum):
    LVL_1 = 1
    LVL_2 = 5
    LVL_3 = 7

class NodeType(Enum):
    VIRTUAL = 1
    REAL = 2
    BORDER = 3


class EdgeDistType(Enum):
    LESS_THAN = 1
    MORE_THAN = 2
    FIXED = 3
    MINIMIZE = 4
    MAXIMIZE = 5
    UNIMPORTANT = 6


class EdgeConnectionType(Enum):
    DISTANCE = 1  # not connected, only care about the distance
    WIRE = 2  # physically connected with the wire, minimized intersection, care about the distance
    BORDER = 3  # components border edges, cannot intersect at all (constraint)


class Node:
    def __init__(
            self,
            name: str,
            node_type: NodeType
    ):
        self.name = name
        self.node_type = node_type


class Edge:
    def __init__(
            self,
            vertices: tuple[Node, Node],
            edge_dist_type: EdgeDistType,
            edge_connection_type: EdgeConnectionType,
            reference_dist: float = 0,
            importance: EdgeDistanceImportance = EdgeDistanceImportance.LVL_1
    ):
        self.vertices = vertices
        self.edge_dist_type = edge_dist_type
        self.edge_connection_type = edge_connection_type
        self.reference_dist = int(round(reference_dist))
        self.importance = importance


class Graph:
    def __init__(
            self,
            vertices: list[str],
            edges: list[tuple[str, str]],
            edges_by_connection_type: dict[EdgeConnectionType, list[tuple[str, str]]],
            edges_by_dist_type: dict[EdgeDistType, list[tuple[str, str]]],
            reference_distances: dict[tuple[str, str], int],
            unparsed_edges: list[Edge],
            component_edges: list[list[Edge]],
            component_bounds: list[tuple[str, str, str, str]]
    ):
        self.vertices = vertices
        self.edges = edges
        self.edges_by_connection_type = edges_by_connection_type
        self.edges_by_dist_type = edges_by_dist_type
        self.reference_distances = reference_distances
        self.unparsed_edges = unparsed_edges
        self.component_edges = component_edges
        self.component_bounds = component_bounds

        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(self.vertices)
        self.nx_graph.add_edges_from(
            self.edges_by_connection_type[EdgeConnectionType.WIRE] +
            self.edges_by_connection_type[EdgeConnectionType.BORDER]
        )
        self.positions = nx.spring_layout(self.nx_graph)

    def get_edge_importance(self, edge: tuple[str, str]):
        for e in self.unparsed_edges:
            u = e.vertices[0].name
            v = e.vertices[1].name
            if (u, v) == edge:
                return e.importance.value

    def visualize(self, name="graphieee", display_dist=True):
        fig, ax = plt.subplots()
        nx.draw(self.nx_graph, self.positions, with_labels=True, node_size=150, node_color='lightblue', edge_color='gray', ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_title(f"{name}")
        ax.grid()

        if display_dist:
            pos_array = np.array([self.positions[v] for v in self.nx_graph.nodes()])
            dist_matrix = squareform(pdist(pos_array))
            for i, j in self.nx_graph.edges():
                i_idx = list(self.nx_graph.nodes).index(i)
                j_idx = list(self.nx_graph.nodes).index(j)
                x = (self.positions[i][0] + self.positions[j][0]) / 2
                y = (self.positions[i][1] + self.positions[j][1]) / 2
                label = f"{dist_matrix[i_idx][j_idx]:.2f}"
                ax.text(x, y, label, fontsize=7, ha='center', va='center')

        plt.show()
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fig.savefig(f"plots/{name}.png")


class GraphParser:

    def parse(
            self,
            input_vertices: list[Node],
            component_edges: list[list[Edge]],
            input_edges: list[Edge],
            component_bounds: list[tuple[str, str, str, str]]
    ):

        def append_to_dict(d: dict, key, value):
            if key not in d:
                d[key] = []
            d[key].append(value)
            return d

        # nodes
        # virtual nodes (ignore that for now)
        vertices = [v.name for v in input_vertices]

        # edges with dists < k
        # edges with dists > k
        # edges with dists = k (within component)
        # -----
        # edges that can be crossed (virtual distance edges)
        # edges that can't be crossed (wires)
        all_edges: list[tuple[str, str]] = []
        edges_by_connection_type: dict[EdgeConnectionType, list[tuple[str, str]]] = {}
        edges_by_dist_type: dict[EdgeDistType, list[tuple[str, str]]] = {}
        reference_distances: dict[tuple[str, str], int] = {}

        for e in input_edges:
            all_edges.append((e.vertices[0].name, e.vertices[1].name))
            edges_by_connection_type = append_to_dict(edges_by_connection_type, e.edge_connection_type, (e.vertices[0].name, e.vertices[1].name))
            edges_by_dist_type = append_to_dict(edges_by_dist_type, e.edge_dist_type, (e.vertices[0].name, e.vertices[1].name))
            reference_distances[(e.vertices[0].name, e.vertices[1].name)] = e.reference_dist

        graph = Graph(
            vertices=vertices,
            edges=all_edges,
            edges_by_connection_type=edges_by_connection_type,
            edges_by_dist_type=edges_by_dist_type,
            reference_distances=reference_distances,
            unparsed_edges=input_edges,
            component_edges=component_edges,
            component_bounds=component_bounds
        )

        return graph
