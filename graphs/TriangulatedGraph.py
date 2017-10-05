# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

from graphs.Graph import *
import heapq as he
from graphs.Star import *
from nodes.Clique import *


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

    """

    def __init__(self, mo_graph, verbose=False):
        """
        Constructor

        Parameters
        ----------
        mo_graph : MoralGraph
        verbose : bool

        Returns
        -------

        """
        # self will be allowed to modify (stomp on) moral graph
        nodes = mo_graph.nodes
        Graph.__init__(self, nodes)
        ord_nodes = list(nodes)
        self.star_heap = []

        # Make yet another copy of moral graph nodes to do bookkeeping. _key
        #  nodes in node_dict will be losing neighbors (inside method
        # refresh_star_heap() ) but value nodes will be gaining them
        node_dict = dict(zip(cp.deepcopy(ord_nodes), ord_nodes))

        # Stuff heap with _key nodes
        for node_key in node_dict.keys():
            he.heappush(self.star_heap, Star(node_key))
        cliques = []
        id_num = 0
        while self.star_heap:
            pop_star = he.heappop(self.star_heap)
            for medge in pop_star.medges:
                # Heap contains key nodes but
                # want to add new neighbors to both
                # key and value nodes
                [n1_key, n2_key] = list(medge)
                node_dict[n1_key].add_neighbor(node_dict[n2_key])
                n1_key.add_neighbor(n2_key)
            preclique = {
                node_dict[nd_key] for nd_key in pop_star.node.neighbors} | \
                {node_dict[pop_star.node]}
            # a preclique is what Huang and Darwiche call an
            # induced cluster
            if verbose:
                print(
                    "\nnode=", pop_star.node.name,
                    ", num_medges=", pop_star.num_medges,
                    ", weight=", pop_star.weight)
                print(
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
        Return False iff the set 'preclique' is contained in any of the 
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

if __name__ == "__main__":
    from examples_cbnets.HuaDar import *
    bnet = HuaDar.build_bnet()
    mo = MoralGraph(bnet)
    tri_graph = TriangulatedGraph(mo, verbose=True)
    print("---------------------neighbors")
    tri_graph.print_neighbors()
