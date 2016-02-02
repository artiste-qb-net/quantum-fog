# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import copy as cp
import heapq as he

from Graph import *
from Clique import *
from Star import *


class TriangulatedGraph(Graph):
    """
    A TriangulatedGraph is an undirected Graph that is constructed from a
    MoralGraph by adding additional edges to it. The new edges are added
    using a non-unique heuristic. Construction of the TriangulatedGraph
    yields a set of cliques of the original BayesNet that was used to
    construct the MoralGraph in the first place. These cliques are used to
    build a JoinTree which is needed by the JoinTreeEngine.

    Attributes
    ----------
    cliques : list[Clique]
    star _heap : list[Star]

    nodes : set[Node]

    """

    def __init__(self, mo_graph, do_print=False):
        """
        Constructor

        Parameters
        ----------
        mo_graph : MoralGraph
        do_print : bool

        Returns
        -------

        """
        # self will be allowed to modify (stomp on) moral graph
        nodes = mo_graph.nodes
        Graph.__init__(self, nodes)
        ord_nodes = list(nodes)
        self.star_heap = []
        # Make yet another copy of moral graph nodes
        # to do bookkeeping. _p stands for prime.
        # Primed nodes will be losing neighbors but
        # unprimed ones will be gaining them.

        node_dict = dict(zip(cp.deepcopy(ord_nodes), ord_nodes))

        # Stuff heap with primed nodes
        for node_p in node_dict.keys():
            he.heappush(self.star_heap, Star(node_p))
        cliques = []
        id_num = 0
        while self.star_heap:
            pop_star = he.heappop(self.star_heap)
            for medge in pop_star.medges:
                # Heap contains primed nodes but
                # want to add new links to both
                # primed and unprimed nodes
                [n1_p, n2_p] = list(medge)
                node_dict[n1_p].add_neighbor(node_dict[n2_p])
                n1_p.add_neighbor(n2_p)
            preclique = {
                node_dict[nd_p] for nd_p in pop_star.node.neighbors} | \
                {node_dict[pop_star.node]}
            # a preclique is what Huang and Darwiche call an
            # induced cluster
            if do_print:
                print(
                    "node(|medges|, w):",
                    pop_star.node.name,
                    (pop_star.num_medges, pop_star.weight), ", ",
                    "preclique:",
                    sorted([nd.name for nd in preclique]), ", ",
                    "edges added:",
                    [{n1.name, n2.name} for (n1, n2) in pop_star.medges]
                )
            if self.preclique_is_maximal(cliques, preclique):
                cliques.append(Clique(id_num, preclique))
                id_num += 1
            self.refresh_star_heap(pop_star)
        self.cliques = cliques

    @staticmethod
    def preclique_is_maximal(clique_list, preclique):
        """
        Answer whether the set 'preclique' is contained in any of the
        cliques in 'clique_list'.

        Parameters
        ----------
        clique_list : list[Clique]
        preclique : set[BayesNode]

        Returns
        -------
        bool

        """
        verdict = True
        for cli in clique_list:
            # compare two sets
            if cli.subnodes >= preclique:
                verdict = False
                break
        return verdict

    def refresh_star_heap(self, pop_star):
        """
        Remove pop_star.node from neighbor sets of all stars in star_heap

        Parameters
        ----------
        pop_star : Star

        Returns
        -------
        None

        """

        star_list = list(self.star_heap)
        changed_star_list = []
        for star in star_list:
            beg_len = len(star.node.neighbors)
            star.node.neighbors = \
                star.node.neighbors - {pop_star.node}
            if beg_len != len(star.node.neighbors):
                changed_star_list.append(star)
        # recompute weights of changed stars
        for star in changed_star_list:
            star.recompute()
        # reorder now that some star weights have changed
        he.heapify(star_list)
        self.star_heap = star_list

    def describe_yourself(self):
        """
        Print a pretty summary of the attributes of self.

        Returns
        -------

        """
        print("Triangle Graph:")
        for node in self.nodes:
            print("name: ", node.name)
            print("neighbors: ",
                  sorted([x.name for x in node.neighbors]))
            print("\n")

from MoralGraph import *
from ExamplesC.HuaDar import *
if __name__ == "__main__":
    bnet = HuaDar.build_bnet()
    mo = MoralGraph(bnet)
    tri_graph = TriangulatedGraph(mo, do_print=True)
    tri_graph.describe_yourself()
