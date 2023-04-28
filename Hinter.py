import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from GraphParser import EdgeConnectionType, EdgeDistType, NodeType, Node, Edge, Graph
from itertools import product, combinations
from sklearn.manifold import MDS

class Hinter:

    def __init__(self, graph):
        self.graph = graph

    def get_hinted_positions(self):
        comp_vertices_pos = self.get_hinted_positions_for_components()
        noncomp_vertices_pos = self.get_hinted_positions_for_noncomp_vertices(comp_vertices_pos)

        # combine the two dictionaries
        pos = {**comp_vertices_pos, **noncomp_vertices_pos}

        min_x = min([pos[v][0] for v in pos])
        min_y = min([pos[v][1] for v in pos])
        # shift positions to the origin
        for v in pos:
            pos[v] = (pos[v][0] - min_x, pos[v][1] - min_y)
        max_x = max([pos[v][0] for v in pos])
        max_y = max([pos[v][1] for v in pos])

        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        self.visualize(G, pos, name="hinted positions", display_dist=False)
        return pos, max_x, max_y

    def get_hinted_positions_for_noncomp_vertices(self, comp_vertices_pos):
        max_x = max([comp_vertices_pos[v][0] for v in comp_vertices_pos])
        max_y = max([comp_vertices_pos[v][1] for v in comp_vertices_pos])

        # get all vertices that do not belong to any component
        noncomp_vertices = [v for v in self.graph.vertices if self.find_component_of_vertex(v) == -1]
        delta = max_y // len(noncomp_vertices)
        noncomp_vertices_pos = {}
        for i in range(len(noncomp_vertices)):
            noncomp_vertices_pos[noncomp_vertices[i]] = (max_x + 1, i * delta)

        return noncomp_vertices_pos

    def get_hinted_positions_for_components(self):

        pos_comp = {}
        size_comp = {}

        for i in range(len(self.graph.component_edges)):
            pos_comp[i], size_comp[i] = self.place_component(self.graph.component_edges[i], self.graph.component_bounds[i])

        min_dist = 1
        max_dist = 3
        default_dist = 2
        inter_comp_dist = {}
        for i in range(len(self.graph.unparsed_edges)):
            edge = self.graph.unparsed_edges[i]
            u, v = edge.vertices
            u_comp = self.find_component_of_vertex(u.name)
            v_comp = self.find_component_of_vertex(v.name)
            if (u_comp, v_comp) in inter_comp_dist:
                continue
            if (u_comp != v_comp) and (u_comp != -1) and (v_comp != -1):

                if edge.edge_dist_type == EdgeDistType.MINIMIZE:
                    inter_comp_dist[(u_comp, v_comp)] = min_dist
                    # inter_comp_dist[(v_comp, u_comp)] = min_dist
                elif edge.edge_dist_type == EdgeDistType.MAXIMIZE:
                    inter_comp_dist[(u_comp, v_comp)] = max_dist
                    # inter_comp_dist[(v_comp, u_comp)] = max_dist

        for i, j in combinations(range(len(self.graph.component_bounds)), 2):
            if (i, j) not in inter_comp_dist:
                all_comps_in_dict = set(np.array(list(inter_comp_dist.keys())).flatten())
                if i not in all_comps_in_dict:
                    inter_comp_dist[(i, j)] = default_dist
                    # inter_comp_dist[(j, i)] = default_dist
                elif j not in all_comps_in_dict:
                    inter_comp_dist[(i, j)] = default_dist
                    # inter_comp_dist[(j, i)] = default_dist

        inter_comp_dist = self.adjust_inter_comp_dists_with_size(inter_comp_dist, size_comp)
        comp_pos = self.position_comp_graph(inter_comp_dist)

        pos = {}
        for comp, com_pos in comp_pos.items():
            pos_of_comp = pos_comp[comp]
            for vertex in pos_of_comp:
                pos[vertex] = pos_of_comp[vertex] + com_pos

        return pos

    def position_comp_graph(self, inter_comp_dist):
        edges = list(inter_comp_dist.keys())
        vertices = list(set(np.array(list(edges)).flatten()))
        G = nx.Graph()
        G.add_nodes_from(vertices)
        # G.add_edges_from(edges)
        # add lengths to edges from inter_comp_dist
        for i, j in edges:
            # G[i][j]['length'] = inter_comp_dist[(i, j)]
            G.add_edge(i, j, weight=inter_comp_dist[(i, j)])

        vertices_count = len(vertices)
        dist_matrix = np.zeros((vertices_count, vertices_count))
        for i, j in combinations(range(vertices_count), 2):
            vertex1, vertex2 = vertices[i], vertices[j]
            distance = nx.shortest_path_length(G, vertex1, vertex2, weight='weight')
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

        # Compute the MDS layout
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(dist_matrix)
        pos = {vertices[i]: embedding[i] for i in range(vertices_count)}
        #
        # new_inter_comp_dist = {}
        # for i, j in edges:
        #     new_inter_comp_dist[(i, j)] = np.linalg.norm(pos[i] - pos[j])
        #
        # # compute the largest error between inter_comp_dist and new_inter_comp_dist
        # max_error = max([inter_comp_dist[(i, j)] - new_inter_comp_dist[(i, j)] for i, j in edges])
        #
        # # Calculate the scaling factor
        # scaling_factor = 1 + max_error / max([np.linalg.norm(coord) for coord in embedding])
        #
        # def scale_positions(pos, scaling_factor):
        #     scaled_positions = {}
        #     for node, coords in pos.items():
        #         scaled_positions[node] = np.array(coords) * scaling_factor
        #     return scaled_positions
        # scaled_pos = scale_positions(pos, scaling_factor)
        #

        int_pos = {v: (round(p[0]), round(p[1])) for v, p in pos.items()}

        return int_pos

    def adjust_inter_comp_dists_with_size(self, inter_comp_dist, size_comp):
        for i, j in inter_comp_dist:
            adj = 0
            if i in size_comp:
                adj += size_comp[i] * 1.2
            if j in size_comp:
                adj += size_comp[j] * 1.2
            inter_comp_dist[(i, j)] = inter_comp_dist[(i, j)] + adj + 1

        return inter_comp_dist

    def find_component_of_vertex(self, vertex):
        for i in range(len(self.graph.component_edges)):
            vertices = set(np.array([
                [edge.vertices[0].name, edge.vertices[1].name] for edge in self.graph.component_edges[i]
            ]).flatten())
            if vertex in vertices:
                return i
        return -1

    def place_component(self, component_edges, component_borders):
        # assumes that a component only has ports on opposite sides
        vertices = set()
        edges = []
        lengths = {}
        for edge in component_edges:
            # vertices.add(edge.nodes[0].name)
            # vertices.add(edge.nodes[1].name)
            if edge.edge_connection_type == EdgeConnectionType.BORDER:
                vertices.add(edge.vertices[0].name)
                vertices.add(edge.vertices[1].name)
                e = (edge.vertices[0].name, edge.vertices[1].name)
                edges.append(e)
                lengths[e] = edge.reference_dist

        top_side = (component_borders[0], component_borders[1])
        bottom_side = (component_borders[2], component_borders[3])
        left_side = (component_borders[0], component_borders[2])
        right_side = (component_borders[1], component_borders[3])

        vertical_connected = ((left_side[0], left_side[1]) in edges) or ((right_side[0], right_side[1]) in edges) or ((left_side[1], left_side[0]) in edges) or ((right_side[1], right_side[0]) in edges)

        unconnected_sides = [top_side, bottom_side] if vertical_connected else [left_side, right_side]
        part_side = unconnected_sides[0]
        another_part_side = unconnected_sides[1]

        side_vertices = set()
        another_side_vertices = set()
        side_dist = 0
        another_side_dist = 0
        # find vertices connected along the part_side
        curr_vertex = part_side[0]
        another_curr_vertix = another_part_side[0]
        for edge in component_edges:
            if edge.edge_connection_type == EdgeConnectionType.BORDER:
                u, v = edge.vertices[0].name, edge.vertices[1].name
                if (u == curr_vertex) and (v not in component_borders):
                    side_vertices.add(v)
                    side_dist = edge.reference_dist
                    curr_vertex = v
                elif (v == curr_vertex) and (u not in component_borders):
                    side_vertices.add(u)
                    side_dist = edge.reference_dist
                    curr_vertex = u
                elif (u == another_curr_vertix) and (v not in component_borders):
                    another_side_vertices.add(v)
                    another_side_dist = edge.reference_dist
                    another_curr_vertix = v
                elif (v == another_curr_vertix) and (u not in component_borders):
                    another_side_vertices.add(u)
                    another_side_dist = edge.reference_dist
                    another_curr_vertix = u

        side_vertices.add(part_side[0])
        side_vertices.add(part_side[1])
        another_side_vertices.add(another_part_side[0])
        another_side_vertices.add(another_part_side[1])

        # arrange side_vertices in a line side_dist apart
        # start with bound vertix
        # find the next vertex that is connected to the bound vertex
        # find the next vertex that is connected to the previous vertex
        # repeat until the other bound vertex is reached

        pos = {
            part_side[0]: np.array([0, 0]),
        }
        another_pos = {
            another_part_side[0]: np.array([0, 0]),
        }
        curr_vertex = part_side[0]
        another_curr_vertex = another_part_side[0]
        for i in range(len(side_vertices) - 1):
            for edge in component_edges:
                if edge.edge_connection_type == EdgeConnectionType.BORDER:
                    u, v = edge.vertices[0].name, edge.vertices[1].name
                    if (u == curr_vertex) and (v in side_vertices) and (v not in pos):
                        pos[v] = pos[u] + np.array([0, side_dist])
                        curr_vertex = v
                    elif (v == curr_vertex) and (u in side_vertices) and (u not in pos):
                        pos[u] = pos[v] + np.array([0, side_dist])
                        curr_vertex = u

        for i in range(len(another_side_vertices) - 1):
            for edge in component_edges:
                if edge.edge_connection_type == EdgeConnectionType.BORDER:
                    u, v = edge.vertices[0].name, edge.vertices[1].name
                    if (u == another_curr_vertex) and (v in another_side_vertices) and (v not in another_pos):
                        another_pos[v] = another_pos[u] + np.array([0, another_side_dist])
                        another_curr_vertex = v
                    elif (v == another_curr_vertex) and (u in another_side_vertices) and (u not in another_pos):
                        another_pos[u] = another_pos[v] + np.array([0, another_side_dist])
                        another_curr_vertex = u

        # get the distance between part_side[0] and another_part_side[0]
        dist = 0
        for e in component_edges:
            if e.edge_connection_type == EdgeConnectionType.BORDER:
                u, v = e.vertices[0].name, e.vertices[1].name
                if (u == part_side[0]) and (v == another_part_side[0]):
                    dist = e.reference_dist
                elif (u == another_part_side[0]) and (v == part_side[0]):
                    dist = e.reference_dist

        another_pos = {k: v + np.array([dist, 0]) for k, v in another_pos.items()}

        dist_ax1 = side_dist * (len(side_vertices) - 1)
        dist_ax2 = dist
        dist_diag = np.sqrt(dist_ax1 ** 2 + dist_ax2 ** 2)

        # join the dicts pos and another pos
        pos.update(another_pos)

        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        # add lengths to the graph
        nx.set_edge_attributes(G, lengths, 'weight')
        # self.visualize(G, pos, name="comp graphieee", display_dist=True)

        return pos, dist_diag

    def visualize(self, graph, pos, name="comp graphieee", display_dist=True):
        fig, ax = plt.subplots()
        nx.draw(graph, pos, with_labels=True, node_size=150, node_color='lightblue', edge_color='gray', ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_title(f"{name}")
        ax.grid()

        if display_dist:
            pos_array = np.array([pos[v] for v in graph.vertices()])
            dist_matrix = squareform(pdist(pos_array))
            for i, j in graph.edges():
                i_idx = list(graph.vertices).index(i)
                j_idx = list(graph.vertices).index(j)
                x = (pos[i][0] + pos[j][0]) / 2
                y = (pos[i][1] + pos[j][1]) / 2
                label = f"{dist_matrix[i_idx][j_idx]:.2f}"
                ax.text(x, y, label, fontsize=7, ha='center', va='center')

        plt.show()
        # if not os.path.exists("plots"):
        #     os.makedirs("plots")
        # fig.savefig(f"plots/{name}.png")

