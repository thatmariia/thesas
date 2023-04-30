from RawData import RawData
from GraphParser import GraphParser
from Hinter import Hinter
from CPOptimizer import CPOptimizer


if __name__ == '__main__':

    rd = RawData()
    graph = GraphParser().parse(rd.vertices, rd.component_edges, rd.edges, rd.component_bounds)
    graph.visualize(name="initial graphie", display_dist=True)

    # hinter = Hinter(graph)
    # hint = hinter.get_hinted_positions()

    pos_suboptimal = {'CO_1': (13, 8), 'CO_2': (13, 12), 'CO_3': (8, 8), 'CO_4': (8, 12), 'COV': (13, 10), 'COG': (8, 10), 'JO_1': (24, 17), 'JO_2': (14, 17), 'JO_3': (24, 0), 'JO_4': (14, 0), 'JOG': (19, 0), 'JOV': (19, 17), 'L1_1': (40, 1), 'L1_2': (40, 15), 'L1_3': (28, 1), 'L1_4': (28, 15), 'L1L': (34, 1), 'L1H': (34, 15), 'JI_1': (33, 23), 'JI_2': (23, 23), 'JI_3': (33, 40), 'JI_4': (23, 40), 'JIG': (28, 40), 'JIV': (28, 23), 'CI_1': (1, 12), 'CI_2': (5, 12), 'CI_3': (1, 7), 'CI_4': (5, 7), 'CIG': (3, 7), 'CIV': (3, 12), 'U_1': (16, 22), 'U_2': (12, 22), 'U_3': (16, 18), 'U_4': (12, 18), 'UVI': (16, 19), 'USW': (16, 20), 'UG': (16, 21), 'UFB': (12, 19), 'UEN': (12, 20), 'UVS': (12, 21), 'C1_1': (25, 16), 'C1_2': (30, 16), 'C1_3': (25, 20), 'C1_4': (30, 20), 'CSW': (25, 18), 'CVS': (30, 18), 'R1_1': (4, 25), 'R1_2': (4, 30), 'R1_3': (0, 25), 'R1_4': (0, 30), 'R1V': (2, 25), 'R1FB': (2, 30), 'R2_1': (2, 37), 'R2_2': (2, 31), 'R2_3': (6, 37), 'R2_4': (6, 31), 'R2FB': (2, 34), 'R2G': (6, 34), 'VOUT': (0, 11), 'VIN': (1, 40), 'SW': (3, 38), 'FB': (11, 19), 'GND': (39, 0), 'VS': (2, 39)}
    hint = (pos_suboptimal, 0, 0)

    cp = CPOptimizer(graph, max_width=rd.max_width, max_height=rd.max_height, scale=1, hint=hint)
    graph.positions = cp.optimize(multiple_workers=True)
    graph.visualize(name="optimized graphie", display_dist=False)


