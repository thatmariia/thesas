from GraphParser import Graph, EdgeConnectionType, EdgeDistType
from ortools.sat.python import cp_model
from itertools import combinations, product
from copy import deepcopy
import json
import numpy as np


class CPOptimizer:

    def __init__(self, graph: Graph, max_width, max_height, scale, hint=None):
        self.graph = graph

        self.model = cp_model.CpModel()

        self.scale = scale
        self.w = max_width * scale
        self.h = max_height * scale

        self.hinted_positions = None
        if hint is not None:
            hinted_positions, max_x, max_y = hint
            self.hinted_positions = hinted_positions
            self.w = max(max_width * scale, max_x)
            self.h = max(max_height * scale, max_y)

        # Variables the values of which we are trying to determine
        self.pos_x = {
            v: self.model.NewIntVar(0, self.w, f'x_{v}')
            for v in self.graph.vertices
        }
        self.pos_y = {
            v: self.model.NewIntVar(0, self.h, f'y_{v}')
            for v in self.graph.vertices
        }

        # Adding other variables that we will need to use
        self.dists2 = {
            e: self.model.NewIntVar(0, self.w**2 + self.h**2, f'dist_{e}')
            for e in self.graph.edges
        }  # the squared values of the distance between the two vertices of the edge
        self.are_wire_edges_crossing = {
            (e1, e2): self.model.NewBoolVar(f'crossing_{e1}_{e2}')
            for e1, e2 in combinations(self.graph.edges_by_connection_type[EdgeConnectionType.WIRE], 2)
        } # bool array indicating whether two wire edges are crossing

        # Adding the objective function variable
        self.min_for_maximize = (self.w ** 2 + self.h ** 2) // 25
        self.max_for_minimize = (self.w ** 2 + self.h ** 2) // 9
        obj_bound1, obj_bound2, obj_bound3, obj_bound = self._get_objective_bound()
        self.objective = self.model.NewIntVar(2, obj_bound, 'objective')

    def optimize(self, multiple_workers=False):
        # adding constraints to fill out definitions of variables
        self._add_edges_constraints_def()
        self._add_breaking_symmetries_constraints()
        self._add_non_overlapping_vertices_constraints()
        self._add_non_intersecting_constraints_def(EdgeConnectionType.WIRE)
        self._add_non_intersecting_constraints_def(EdgeConnectionType.BORDER)
        #### self._add_non_containing_boxes_constraints()
        self._add_no_points_within_components_constraints()
        self._add_orientation_constraints()
        self._add_interborder_dist_constraint()  # - todo:: CHECK

        # constraints for distance edges
        self._add_fixed_edges_dist_constraints()
        self._add_less_than_edges_dist_constraints()
        self._add_more_than_edges_dist_constraints()

        # adding constraints for the objective function
        self._add_objective_function_constraints_def()

        # setting the objective function
        self.model.Minimize(self.objective)

        self._add_hints()

        # chose coordinates for the vertices
        self._add_decision_strategy()

        # solving the model
        solver = cp_model.CpSolver()

        solver.parameters.log_search_progress = True
        if multiple_workers:
            # cannot use the solution printer with multiple workers
            solver.parameters.num_search_workers = 4
            status = solver.Solve(self.model)
        else:
            solution_printer = VarArraySolutionPrinter(
                self.objective,
                self.graph,
                self.pos_x,
                self.pos_y,
            )
            solver.parameters.enumerate_all_solutions = True
            status = solver.Solve(self.model, solution_printer)

        # Interpret the status code
        if status == cp_model.OPTIMAL:
            print('The solver found an optimal solution.')
        elif status == cp_model.FEASIBLE:
            print('The solver found a feasible solution, but it may not be optimal.')
        elif status == cp_model.INFEASIBLE:
            print('The solver determined that there are no feasible solutions.')
        else:
            print('The solver returned an unknown status code.')

        # get resulting positions from solver
        positions = self._extract_positions_from_solutions(solver)
        print("RESULT")
        print(positions)
        return positions

    def _add_decision_strategy(self):
        list_pos = []
        for v in self.graph.vertices:
            list_pos.append(self.pos_x[v])
            list_pos.append(self.pos_y[v])

        order_strategy = cp_model.CHOOSE_FIRST
        domain_strategy = cp_model.SELECT_MIN_VALUE
        self.model.AddDecisionStrategy(list_pos, order_strategy, domain_strategy)

    def _add_hints(self):
        if self.hinted_positions is None:
            return
        for v, (x, y) in self.hinted_positions.items():
            # self.model.Add(self.pos_x[v] == x)
            # self.model.Add(self.pos_y[v] == y)
            self.model.AddHint(self.pos_x[v], x)
            self.model.AddHint(self.pos_y[v], y)

    def _extract_positions_from_solutions(self, solver):
        positions = {}
        for v in self.graph.vertices:
            positions[v] = (solver.Value(self.pos_x[v]), solver.Value(self.pos_y[v]))
        return positions

    def _get_objective_bound(self):
        obj1_bound = len(self.graph.edges_by_connection_type[EdgeConnectionType.WIRE]) ** 2 + 1
        obj2_bound = (self.max_for_minimize) * len(self.graph.edges_by_dist_type[EdgeDistType.MINIMIZE])
        obj3_bound = ((self.w**2 + self.h**2) - self.min_for_maximize) * len(self.graph.edges_by_dist_type[EdgeDistType.MAXIMIZE])
        total_obj = obj1_bound * (obj2_bound + obj3_bound)
        return obj1_bound, obj2_bound, obj3_bound, total_obj

    def _add_objective_function_constraints_def(self):
        obj_bound1, obj_bound2, obj_bound3, _ = self._get_objective_bound()

        # --- objective for crossing minimum nr of wires
        # count the number of True's in self.are_wire_edges_crossing
        objective_crossings = self.model.NewIntVar(0, obj_bound1, 'objective_crossings')
        self.model.Add(objective_crossings == sum(self.are_wire_edges_crossing.values()) + 1)

        # --- objective for minimizing the distance between edges that needs to be minimized
        objective_min_dist = self.model.NewIntVar(0, obj_bound2, 'objective_min_dist')
        min_dist_errors = {
            e: self.model.NewIntVar(0, (self.w**2 + self.h**2), f'min_dist_{e}')
            for e in self.graph.edges_by_dist_type[EdgeDistType.MINIMIZE]
        }
        for e in self.graph.edges_by_dist_type[EdgeDistType.MINIMIZE]:
            self.model.Add(min_dist_errors[e] == self.dists2[e])
        self.model.Add(objective_min_dist == sum(min_dist_errors.values()) + 1)

        # --- objective for maximizing the distance between edges that needs to be maximized
        max_dist = (self.w**2 + self.h**2) - self.min_for_maximize
        objective_max_dist = self.model.NewIntVar(0, obj_bound3, 'objective_max_dist')
        max_dist_errors = {
            e: self.model.NewIntVar(0, (self.w**2 + self.h**2), f'max_dist_{e}')
            for e in self.graph.edges_by_dist_type[EdgeDistType.MAXIMIZE]
        }
        for e in self.graph.edges_by_dist_type[EdgeDistType.MAXIMIZE]:
            self.model.Add(max_dist_errors[e] == max_dist - self.dists2[e])
        self.model.Add(objective_max_dist == sum(max_dist_errors.values()) + 1)

        # --- final objective
        objective23 = self.model.NewIntVar(0, obj_bound2 + obj_bound3, 'objective23')
        self.model.Add(objective23 == objective_min_dist + objective_max_dist)
        self.model.AddMultiplicationEquality(self.objective, [objective_crossings, objective23])

    def _add_interborder_dist_constraint(self):
        border_vertices = list(np.array(self.graph.component_bounds).flatten())
        for u, v in combinations(border_vertices, 2):
            # get distance between u and v
            abs_dist_x = self.model.NewIntVar(0, self.w, f'interborder_abs_dist_x_{u}_{v}')
            abs_dist_y = self.model.NewIntVar(0, self.h, f'interborder_abs_dist_y_{u}_{v}')
            self.model.AddAbsEquality(abs_dist_x, self.pos_x[u] - self.pos_x[v])
            self.model.AddAbsEquality(abs_dist_y, self.pos_y[u] - self.pos_y[v])
            dist_x2 = self.model.NewIntVar(0, self.w**2, f'interborder_dist_x2_{u}_{v}')
            dist_y2 = self.model.NewIntVar(0, self.h**2, f'interborder_dist_y2_{u}_{v}')
            self.model.AddMultiplicationEquality(dist_x2, [abs_dist_x, abs_dist_x])
            self.model.AddMultiplicationEquality(dist_y2, [abs_dist_y, abs_dist_y])
            dist2 = self.model.NewIntVar(0, self.w**2 + self.h**2, f'interborder_dist2_{u}_{v}')
            self.model.Add(dist2 == dist_x2 + dist_y2)
            self.model.Add(dist2 >= 1)

    def _add_breaking_symmetries_constraints(self):
        nr_components = len(self.graph.component_bounds)
        component = self.graph.component_bounds[nr_components // 2]
        a, b, c, d = component  # (a, b) || (c, d) ; (a, c) || (b, d)

        self.model.Add(self.pos_x[a] < self.pos_x[b])
        self.model.Add(self.pos_x[c] < self.pos_x[d])
        self.model.Add(self.pos_y[a] > self.pos_y[c])
        self.model.Add(self.pos_y[b] > self.pos_y[d])

    def _add_orientation_constraints(self):
        for component in self.graph.component_bounds:
            a, b, c, d = component  # (a, b) || (c, d) ; (a, c) || (b, d)
            orientations = [
                (a, b, c, d),
                (c, a, d, b),
                (d, c, b, a),
                (b, d, a, c)
            ]

            # make sure that the component can only rotate 90 degrees
            rotations = {
                orientation: self.model.NewBoolVar(f'rotations_{component}_{orientation}') for orientation in orientations
            }

            def check_rotation(orientation):
                k, l, m, n = orientation

                left_x_aligned = self.model.NewBoolVar(f'left_x_aligned_{component}_{orientation}')
                self.model.Add(self.pos_x[k] == self.pos_x[m]).OnlyEnforceIf(rotations[orientation])

                right_x_aligned = self.model.NewBoolVar(f'right_x_aligned_{component}_{orientation}')
                self.model.Add(self.pos_x[l] == self.pos_x[n]).OnlyEnforceIf(rotations[orientation])

                top_y_aligned = self.model.NewBoolVar(f'top_y_aligned_{component}_{orientation}')
                self.model.Add(self.pos_y[k] == self.pos_y[l]).OnlyEnforceIf(rotations[orientation])

                bottom_y_aligned = self.model.NewBoolVar(f'bottom_y_aligned_{component}_{orientation}')
                self.model.Add(self.pos_y[m] == self.pos_y[n]).OnlyEnforceIf(rotations[orientation])

                num_aligned = self.model.NewIntVar(0, 4, f'num_aligned_{component}_{orientation}')
                self.model.Add(num_aligned == left_x_aligned + right_x_aligned + top_y_aligned + bottom_y_aligned)

                are_all_aligned = self.model.NewBoolVar(f'are_all_aligned_{component}_{orientation}')
                # are_all_aligned = true if num_aligned == 4, else false
                self.model.Add(num_aligned == 4).OnlyEnforceIf(are_all_aligned)
                self.model.Add(num_aligned != 4).OnlyEnforceIf(are_all_aligned.Not())

                self.model.Add(rotations[orientation] == are_all_aligned)

            for orientation in orientations:
                check_rotation(orientation)

            self.model.Add(sum(rotations.values()) == 1)

    def _add_non_overlapping_vertices_constraints(self):
        for u, v in combinations(self.graph.vertices, 2):
            diff_x = self.model.NewIntVar(-self.w, self.w, f'diff_x_{u}_{v}')
            self.model.Add(diff_x == self.pos_x[u] - self.pos_x[v])
            diff_y = self.model.NewIntVar(-self.h, self.h, f'diff_y_{u}_{v}')
            self.model.Add(diff_y == self.pos_y[u] - self.pos_y[v])

            is_diff_x_zero = self.model.NewBoolVar(f'is_diff_x_zero_{u}_{v}')
            self.model.Add(diff_x == 0).OnlyEnforceIf(is_diff_x_zero)
            self.model.Add(diff_x != 0).OnlyEnforceIf(is_diff_x_zero.Not())
            is_diff_y_zero = self.model.NewBoolVar(f'is_diff_y_zero_{u}_{v}')
            self.model.Add(diff_y == 0).OnlyEnforceIf(is_diff_y_zero)
            self.model.Add(diff_y != 0).OnlyEnforceIf(is_diff_y_zero.Not())

            self.model.Add(is_diff_x_zero + is_diff_y_zero <= 1)

    def _add_more_than_edges_dist_constraints(self):
        # for edge in self.graph.edges_by_dist_type[EdgeDistType.MORE_THAN]:
        #     self.model.Add(
        #         self.dists2[edge] >= self.graph.reference_distances[edge] ** 2
        #     )

        for edge in self.graph.edges_by_dist_type[EdgeDistType.MAXIMIZE]:
            self.model.Add(
                self.dists2[edge] >= self.min_for_maximize
            )

    def _add_less_than_edges_dist_constraints(self):
        # for edge in self.graph.edges_by_dist_type[EdgeDistType.LESS_THAN]:
        #     self.model.Add(
        #         self.dists2[edge] <= self.graph.reference_distances[edge] ** 2
        #     )

        for edge in self.graph.edges_by_dist_type[EdgeDistType.MINIMIZE]:
            self.model.Add(
                self.dists2[edge] <= self.max_for_minimize
            )

    def _add_fixed_edges_dist_constraints(self):
        for edge in self.graph.edges_by_dist_type[EdgeDistType.FIXED]:
            print(edge, self.graph.reference_distances[edge] ** 2)
            self.model.Add(
                self.dists2[edge] == self.graph.reference_distances[edge] ** 2
            )

    def _add_non_intersecting_constraints_def(self, edge_connection_type):
        for edge1, edge2 in combinations(self.graph.edges_by_connection_type[edge_connection_type], 2):
            # quit if the same edge
            if edge1 == edge2:
                continue
            u1, v1 = edge1
            u2, v2 = edge2
            # quit if the edges share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue

            # check for intersections

            t = {
                idx: self.model.NewIntVar(-self.w * self.h, self.w * self.h, f't_{u1}_{v1}_{u2}_{v2}_{idx}')
                for idx in range(1, 5)
            }

            def half_plane_test(idx, p, q, r):
                term1_1 = self.model.NewIntVar(-self.w, self.w, f'term1_1_{edge1}_{edge2}')
                term1_2 = self.model.NewIntVar(-self.h, self.h, f'term1_2_{edge1}_{edge2}')
                term2_1 = self.model.NewIntVar(-self.h, self.h, f'term2_1_{edge1}_{edge2}')
                term2_2 = self.model.NewIntVar(-self.w, self.w, f'term2_2_{edge1}_{edge2}')
                term1 = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'term1_{edge1}_{edge2}')
                term2 = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'term2_{edge1}_{edge2}')
                res = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'res_{edge1}_{edge2}')
                self.model.Add(term1_1 == self.pos_x[q] - self.pos_x[p])
                self.model.Add(term1_2 == self.pos_y[r] - self.pos_y[p])
                self.model.AddMultiplicationEquality(term1, [term1_1, term1_2])
                self.model.Add(term2_1 == self.pos_y[q] - self.pos_y[p])
                self.model.Add(term2_2 == self.pos_x[r] - self.pos_x[p])
                self.model.AddMultiplicationEquality(term2, [term2_1, term2_2])
                self.model.Add(res == term1 - term2)
                self.model.Add(t[idx] == res)

            def do_segments_intersect(p1, q1, p2, q2):
                half_plane_test(1, p1, q1, p2)
                half_plane_test(2, p1, q1, q2)
                half_plane_test(3, p2, q2, p1)
                half_plane_test(4, p2, q2, q1)

                t12 = self.model.NewIntVar(-self.w ** 2 * self.h ** 2, self.w ** 2 * self.h ** 2, f't12_{edge1}_{edge2}')
                t34 = self.model.NewIntVar(-self.w ** 2 * self.h ** 2, self.w ** 2 * self.h ** 2, f't34_{edge1}_{edge2}')
                self.model.AddMultiplicationEquality(t12, [t[1], t[2]])
                self.model.AddMultiplicationEquality(t34, [t[3], t[4]])
                is_t12_lest_than_zero = self.model.NewBoolVar(f'is_t12_lest_than_zero_{edge1}_{edge2}')
                is_t34_lest_than_zero = self.model.NewBoolVar(f'is_t34_lest_than_zero_{edge1}_{edge2}')
                self.model.Add(t12 <= 0).OnlyEnforceIf(is_t12_lest_than_zero)
                self.model.Add(t12 > 0).OnlyEnforceIf(is_t12_lest_than_zero.Not())
                self.model.Add(t34 <= 0).OnlyEnforceIf(is_t34_lest_than_zero)
                self.model.Add(t34 > 0).OnlyEnforceIf(is_t34_lest_than_zero.Not())
                are_both_less_than_zero = self.model.NewBoolVar(f'are_both_less_than_zero_{edge1}_{edge2}')
                self.model.AddBoolAnd([is_t12_lest_than_zero, is_t34_lest_than_zero]).OnlyEnforceIf(are_both_less_than_zero)
                self.model.AddBoolOr([is_t12_lest_than_zero.Not(), is_t34_lest_than_zero.Not()]).OnlyEnforceIf(are_both_less_than_zero.Not())

                if edge_connection_type == EdgeConnectionType.WIRE:
                    # are edges crossing?
                    self.model.Add(self.are_wire_edges_crossing[edge1, edge2] == are_both_less_than_zero)
                else:
                    # enforce the edges aren't crossing --> are_both_less_than_zero.Not()
                    self.model.Add(1 == 1).OnlyEnforceIf(are_both_less_than_zero.Not())

            do_segments_intersect(u1, v1, u2, v2)

    def _add_no_points_within_components_constraints(self):
        for i in range(len(self.graph.component_bounds)):
            component = self.graph.component_bounds[i]
            a, b, c, d = component

            min_x = self.model.NewIntVar(0, self.w, f'min_x_{component}')
            max_x = self.model.NewIntVar(0, self.w, f'max_x_{component}')
            min_y = self.model.NewIntVar(0, self.h, f'min_y_{component}')
            max_y = self.model.NewIntVar(0, self.h, f'max_y_{component}')

            self.model.AddMinEquality(min_x, [self.pos_x[a], self.pos_x[b], self.pos_x[c], self.pos_x[d]])
            self.model.AddMaxEquality(max_x, [self.pos_x[a], self.pos_x[b], self.pos_x[c], self.pos_x[d]])
            self.model.AddMinEquality(min_y, [self.pos_y[a], self.pos_y[b], self.pos_y[c], self.pos_y[d]])
            self.model.AddMaxEquality(max_y, [self.pos_y[a], self.pos_y[b], self.pos_y[c], self.pos_y[d]])

            component_vertices = set()
            for edge in self.graph.component_edges[i]:
                u, v = edge.vertices[0], edge.vertices[1]
                component_vertices.add(u)
                component_vertices.add(v)

            # check that no vertex is inside the component
            for vertex in self.graph.vertices:
                if vertex in component_vertices:
                    continue
                conditions = [
                    self.model.NewBoolVar(f'no_points_within_{component}_{vertex}_{i}') for i in range(4)
                ]
                self.model.Add(self.pos_x[vertex] > min_x).OnlyEnforceIf(conditions[0])
                self.model.Add(self.pos_x[vertex] <= min_x).OnlyEnforceIf(conditions[0].Not())
                self.model.Add(self.pos_x[vertex] < max_x).OnlyEnforceIf(conditions[1])
                self.model.Add(self.pos_x[vertex] >= max_x).OnlyEnforceIf(conditions[1].Not())
                self.model.Add(self.pos_y[vertex] > min_y).OnlyEnforceIf(conditions[2])
                self.model.Add(self.pos_y[vertex] <= min_y).OnlyEnforceIf(conditions[2].Not())
                self.model.Add(self.pos_y[vertex] < max_y).OnlyEnforceIf(conditions[3])
                self.model.Add(self.pos_y[vertex] >= max_y).OnlyEnforceIf(conditions[3].Not())
                self.model.AddBoolOr([conditions[0].Not(), conditions[1].Not(), conditions[2].Not(), conditions[3].Not()])
                # self.model.Add(sum(conditions) != 4)

    def _add_non_containing_boxes_constraints(self):
        for box1, box2 in combinations(self.graph.component_bounds, 2):
            if box1 == box2:
                continue
            # order: top left, top right, bottom left, bottom right
            a, b, c, d = box1  # a, b, d, c clockwise
            k, l, m, n = box2  # k, l, n, m clockwise

            def point_inside_polygon(p, box):  # Ray Casting
                e, f, g, h = box
                edges_box = [(e, f), (f, h), (h, g), (g, e)]  # clockwise

                is_inside_box = self.model.NewBoolVar(f'is_inside_{p}_{box}')
                is_inside_edge = {
                    edge: self.model.NewBoolVar(f'is_inside_{p}_{box}') for edge in edges_box
                }

                for i, j in edges_box:
                    edge = (i, j)

                    # Calculate cross product
                    term1 = self.model.NewIntVar(-self.w, self.w, f'term1_{p}_{box}')
                    self.model.Add(term1 == self.pos_x[j] - self.pos_x[i])
                    term2 = self.model.NewIntVar(-self.h, self.h, f'term2_{p}_{box}')
                    self.model.Add(term2 == self.pos_y[p] - self.pos_y[i])

                    term3 = self.model.NewIntVar(-self.w, self.w, f'term3_{p}_{box}')
                    self.model.Add(term3 == self.pos_x[p] - self.pos_x[i])
                    term4 = self.model.NewIntVar(-self.h, self.h, f'term4_{p}_{box}')
                    self.model.Add(term4 == self.pos_y[j] - self.pos_y[i])

                    cross_product = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'cross_product_{p}_{box}')
                    self.model.AddMultiplicationEquality(cross_product, [term1, term2])
                    cross_product_term2 = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'cross_product_term2_{p}_{box}')
                    self.model.AddMultiplicationEquality(cross_product_term2, [term3, term4])

                    self.model.Add(cross_product - cross_product_term2 <= 0).OnlyEnforceIf(is_inside_edge[edge])

                # Vertex K is inside quad1 if and only if it's inside with respect to all edges
                self.model.AddBoolAnd([is_inside_edge[edge] for edge in edges_box]).OnlyEnforceIf(is_inside_box)

                # Make sure that the vertex is not inside the box
                self.model.Add(is_inside_box == False)

            for i in range(2):
                curr_box = box1 if i == 0 else box2
                another_box = box2 if i == 0 else box1
                for vertex in list(another_box):
                    # Do the check for each vertex of another box
                    # Check if vertex is inside the box
                    point_inside_polygon(vertex, curr_box)

    def _add_edges_constraints_def(self):
        # adding variables and constraints for the simple definitions like the edges
        for u, v in self.graph.edges:

            # the abs value of the diff of the x coords of the two vertices of the edge
            edge_x = self.model.NewIntVar(0, self.w, f'edge_x_{u}_{v}')
            # the abs value of the diff of the y coords of the two vertices of the edge
            edge_y = self.model.NewIntVar(0, self.h, f'edge_y_{u}_{v}')
            # the squared value of the diff of the x coords of the two vertices of the edge
            edge_x2 = self.model.NewIntVar(0, self.w ** 2, f'edge_w2_{u}_{v}')
            # the squared value of the diff of the y coords of the two vertices of the edge
            edge_y2 = self.model.NewIntVar(0, self.h ** 2, f'edge_y2_{u}_{v}')

            self.model.AddAbsEquality(edge_x, self.pos_x[u] - self.pos_x[v])
            self.model.AddAbsEquality(edge_y, self.pos_y[u] - self.pos_y[v])
            self.model.AddMultiplicationEquality(edge_x2, [edge_x, edge_x])
            self.model.AddMultiplicationEquality(edge_y2, [edge_y, edge_y])
            self.model.Add(self.dists2[(u, v)] == edge_x2 + edge_y2)

            # nodes cannot be in the same position
            # self.model.Add(self.dists[(u, v)] > 0)


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, objective, graph, pos_x, pos_y):
        cp_model.CpSolverSolutionCallback.__init__(self)
        #self.__variables = variables
        self.__objective = objective
        self.__graph = deepcopy(graph)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.__solution_count = 0

    def _extract_positions(self):
        positions = {}
        for v in self.__graph.vertices:
            positions[v] = (self.Value(self.pos_x[v]), self.Value(self.pos_y[v]))
        return positions

    def _save_positions_to_file(self):
        self.__graph.positions = self._extract_positions()
        with open(f"pos/solution_{self.__solution_count}_obj_{self.Value(self.__objective)}.json", "w") as f:
            # Write the dictionary to the file
            json.dump(self.__graph.positions, f)

    def on_solution_callback(self):
        self.__solution_count += 1
        objective_value = self.Value(self.__objective)
        print(f'Solution {self.__solution_count}: Objective value: {objective_value}\n')
        self._save_positions_to_file()
        self.__graph.visualize(name=f'solution_{self.__solution_count}', display_dist=True)

    def solution_count(self):
        return self.__solution_count


# def point_inside_polygon(p, box):  # Ray Casting
            #     e, f, g, h = box
            #     edges_box = [(e, f), (f, h), (h, g), (g, e)]  # clockwise
            #
            #     is_inside_box = self.model.NewBoolVar(f'is_inside_{p}_{box}')
            #     is_inside_edge = {
            #         edge: self.model.NewBoolVar(f'is_inside_{p}_{box}') for edge in edges_box
            #     }
            #
            #     for i, j in edges_box:
            #         edge = (i, j)
            #         inter = self.model.NewIntVar(-self.w, self.w, f'inter_{p}_{box}')
            #
            #         # Add constraints to calculate intersections and check if vertex K is inside quad1
            #         term1 = self.model.NewIntVar(-self.h, self.h, f'term1_{p}_{box}')
            #         self.model.Add(term1 == self.pos_y[p] - self.pos_y[i])
            #         term2 = self.model.NewIntVar(-self.w, self.w, f'term2_{p}_{box}')
            #         self.model.Add(term2 == self.pos_x[j] - self.pos_x[i])
            #         term12 = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'term12_{p}_{box}')
            #         self.model.AddMultiplicationEquality(term12, [term1, term2])
            #         term3 = self.model.NewIntVar(-self.h, self.h, f'term3_{p}_{box}')
            #         self.model.Add(term3 == self.pos_y[j] - self.pos_y[i])
            #         term123 = self.model.NewIntVar(-self.w * self.h, self.w * self.h, f'term123_{p}_{box}')
            #         self.model.AddDivisionEquality(term123, term12, term3)
            #         self.model.Add(inter == term123 + self.pos_x[i]).OnlyEnforceIf(is_inside_edge[edge])
            #
            #         self.model.Add(self.pos_x[p] <= inter).OnlyEnforceIf(is_inside_edge[edge])
            #
            #     # Vertex K is inside quad1 if and only if it's inside with respect to all edges
            #     self.model.AddBoolAnd([is_inside_edge[edge] for edge in edges_box]).OnlyEnforceIf(is_inside_box)
            #     # make sure that the vertex is not inside the box
            #     self.model.Add(is_inside_box == False)
            #
            # for i in range(2):
            #     curr_box = box1 if i == 0 else box2
            #     another_box = box2 if i == 0 else box1
            #     for vertex in list(another_box):
            #         # do the check for each vertex of another box
            #         # check if vertex is inside the box
            #         point_inside_polygon(vertex, curr_box)

# def _add_non_containing_boxes_constraints_old(self):
#     for box1, box2 in combinations(self.graph.component_bounds, 2):
#         if box1 == box2:
#             continue
#         # order: top left, top right, bottom left, bottom right
#         a, b, c, d = box1  # a, b, d, c clockwise
#         k, l, m, n = box2  # k, l, n, m clockwise
#         all_nodes = [a, b, c, d, k, l, m, n]
#         edges_box1 = [(a, b), (b, d), (d, c), (c, a)]
#         edges_box2 = [(k, l), (l, n), (n, m), (m, k)]
#         all_edges = edges_box1 + edges_box2
#
#         edge_directions_x = {
#             (x, y): self.model.NewIntVar(-self.w, self.w, f'edge_direction_x_{x}_{y}') for x, y in all_edges
#         }
#         edge_directions_y = {
#             (x, y): self.model.NewIntVar(-self.h, self.h, f'edge_direction_y_{x}_{y}') for x, y in all_edges
#         }
#
#         def edge_direction(x, y):
#             self.model.Add(edge_directions_x[x, y] == self.pos_x[y] - self.pos_x[x])
#             self.model.Add(edge_directions_y[x, y] == self.pos_y[y] - self.pos_y[x])
#
#         orthogonal_x = {
#             (x, y): self.model.NewIntVar(-self.w, self.w, f'orthogonal_x_{x}_{y}') for x, y in all_edges
#         }
#         orthogonal_y = {
#             (x, y): self.model.NewIntVar(-self.h, self.h, f'orthogonal_y_{x}_{y}') for x, y in all_edges
#         }
#
#         def orthogonal(x, y):
#             self.model.Add(orthogonal_x[x, y] == -edge_directions_y[x, y])
#             self.model.Add(orthogonal_y[x, y] == edge_directions_x[x, y])
#
#         dots_x = {x: self.model.NewIntVar(-self.w ** 2, self.w ** 2, f'dot_x_{x}') for x in all_nodes}
#         dots_y = {x: self.model.NewIntVar(-self.h ** 2, self.h ** 2, f'dot_y_{x}') for x in all_nodes}
#         dots = {x: self.model.NewIntVar(-self.w ** 2 - self.h ** 2, self.w ** 2 + self.h ** 2, f'dot_{x}') for x
#                 in all_nodes}
#         min_dots = {
#             (box, e): self.model.NewIntVar(-self.w ** 2 - self.h ** 2, self.w ** 2 + self.h ** 2,
#                                            f'min_dot_{box}_{e}')
#             for box, e in product([box1, box2], all_edges)
#         }
#         max_dots = {
#             (box, e): self.model.NewIntVar(-self.w ** 2 - self.h ** 2, self.w ** 2 + self.h ** 2,
#                                            f'max_dot_{box}_{e}')
#             for box, e in product([box1, box2], all_edges)
#         }
#
#         def project(box, orth_edge):
#             for x in list(box):
#                 self.model.AddMultiplicationEquality(dots_x[x], [self.pos_x[x], orthogonal_x[orth_edge]])
#                 self.model.AddMultiplicationEquality(dots_y[x], [self.pos_y[x], orthogonal_y[orth_edge]])
#                 self.model.Add(dots[x] == dots_x[x] + dots_y[x])
#             self.model.AddMinEquality(min_dots[(box, orth_edge)], [dots[x] for x in list(box)])
#             self.model.AddMaxEquality(max_dots[(box, orth_edge)], [dots[x] for x in list(box)])
#
#         overlaping1 = {
#             e: self.model.NewBoolVar(f'overlaping1_{box1}_{box2}_{e}') for e in all_edges
#         }
#         overlaping2 = {
#             e: self.model.NewBoolVar(f'overlaping2_{box1}_{box2}_{e}') for e in all_edges
#         }
#         overlaping = {
#             e: self.model.NewBoolVar(f'overlaping_{box1}_{box2}_{e}') for e in all_edges
#         }
#
#         def overlap(orth_edge):
#             # min_dots[(box1, orth_edge)] <= max_dots[(box2, orth_edge)] --> overlaping1
#             self.model.Add(min_dots[(box1, orth_edge)] <= max_dots[(box2, orth_edge)]).OnlyEnforceIf(
#                 overlaping1[orth_edge])
#             # min_dots[(box2, orth_edge)] <= max_dots[(box1, orth_edge)] --> overlaping2
#             self.model.Add(min_dots[(box2, orth_edge)] <= max_dots[(box1, orth_edge)]).OnlyEnforceIf(
#                 overlaping2[orth_edge])
#             # overlaping1 and overlaping2 --> overlaping
#             self.model.AddBoolAnd([overlaping1[orth_edge], overlaping2[orth_edge]]).OnlyEnforceIf(
#                 overlaping[orth_edge])
#
#         for x, y in all_edges:
#             edge_direction(x, y)
#             orthogonal(x, y)
#             project(box1, (x, y))
#             project(box2, (x, y))
#             overlap((x, y))
#             # enforce not overlaping!!
#             self.model.AddBoolOr([overlaping[(x, y)].Not()])