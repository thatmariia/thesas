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

    cp = CPOptimizer(graph, max_width=rd.max_width, max_height=rd.max_height, scale=1, hint=None)
    graph.positions = cp.optimize(multiple_workers=True)
    graph.visualize(name="optimized graphie", display_dist=False)


