# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# from potentials.Potential import *
# from graphs.Graph import *
# from graphs.Sepset import *
from graphs.BayesNet import *
from graphs.TriangulatedGraph import *
# from MyExceptions import *


class JoinTree(Graph):
    """
    JoinTree is the final undirected graph that is constructed for
    JoinTreeEngine. The nodes of a JoinTree are cliques. Cliques are
    themselves sets of subnodes. The cliques are determined in a previous
    step which involves building a TriangulatedGraph. The Sepset's (
    separation sets) are not nodes of the JoinTree graph. A clique stores a
    sepset for each clique adjacent to it.

    Attributes
    ----------
    nodes : set[Node]

    """

    def __init__(self, tri_gra, bnet):
        """
        Constructor


        Parameters
        ----------
        tri_gra : TriangulatedGraph
        bnet : BayesNet

        Returns
        -------

        """
        cliques = self.get_bnet_cliques(tri_gra, bnet)
        # We start by creating a subgraph with each clique,
        # and putting them into a list called subgraphs
        subgraphs = [Graph({cl}) for cl in cliques]
        sepset_heap = Sepset.create_sepset_heap(cliques)
        # Join n - 1 subgraphs together forming (hopefully)
        # a single graph called the join tree.
        for n in range(len(subgraphs) - 1):
            while sepset_heap:
                sepset = he.heappop(sepset_heap)
                # Find out which tree each clique is from
                subg_x = JoinTree.get_clique_owner(
                        subgraphs, sepset.clique_x)
                subg_y = JoinTree.get_clique_owner(
                        subgraphs, sepset.clique_y)
                if not subg_x == subg_y:
                    # If the cliques are on different
                    # subgraphs, then join subgraphs
                    # to make a larger one.
                    JoinTree.merge_subgraphs(sepset, subg_x, subg_y)
                    subgraphs.remove(subg_y)
                    break
        assert len(subgraphs) == 1, \
            "Subgraphs did not merge into single graph"
        Graph.__init__(self, subgraphs[0].nodes)

    @staticmethod
    def get_bnet_cliques(tri_graph, bnet):
        """
        The TriangulatedGraph returns cliques whose subnodes are COPIES of
        the nodes in the original Bnet. This method swaps the subnodes of
        each clique by the original bnet nodes, the real McCoy. This method
        modifies tri_graph (stomps on it) now that it has already served its
        purpose of finding the cliques.

        Parameters
        ----------
        tri_graph : TriangulatedGraph
        bnet : BayesNet

        Returns
        -------
        list[Clique]

        """
        tri_id_to_nd = {node.id_num: node for node in tri_graph.nodes}
        bnet_id_to_nd = {node.id_num: node for node in bnet.nodes}
        id_nums = [node.id_num for node in bnet.nodes]
        node_dict = {tri_id_to_nd[k]: bnet_id_to_nd[k] for k in id_nums}
        for clique in tri_graph.cliques:
            # this won't work because you can't change index
            # you are iterating over?
            # for node in clique.subnodes:
            #      node = node_dict[node]
            clique.subnodes = {node_dict[node]
                               for node in clique.subnodes}
        return tri_graph.cliques

    @staticmethod
    def get_clique_owner(subgraphs, clique):
        """
        From a list of subgraphs called 'subgraphs', find a subgraph that
        contains the nodes of clique.

        Parameters
        ----------
        subgraphs : list[Graph]
        clique : Clique

        Returns
        -------
        Graph

        """
        for sub_g in subgraphs:
            if clique in sub_g.nodes:
                return sub_g

    @staticmethod
    def merge_subgraphs(sepset, strong_g, weak_g):
        """
        This merges two JoinTree subgraphs called strong_g and weak_g and
        labels the merged result strong_g. (That is why it is called
        strong_g; because it is the one of the 2 that survives). To link the
        2 subgraphs, the two cliques in 'sepset' are made neighbors and
        'sepset' is put into their sepset lists.

        Parameters
        ----------
        sepset : Sepset
        strong_g : Graph
        weak_g : Graph

        Returns
        -------
        None

        """
        cx = sepset.clique_x
        cy = sepset.clique_y

        # this adds neighbors to both cx and cy
        cx.add_neighbor(cy)

        cx.add_sepset(sepset)
        cy.add_sepset(sepset)

        strong_g.add_nodes(weak_g.nodes)

    def set_clique_and_sepset_pots_to_one(self, is_quantum):
        """
        This sets clique and sepset pots to 1 over all states of their
        subnodes, not just over the active ones. If 1 over the active states
        and 0 over the inactive ones is desired, next apply mask_self() to
        the potential. is_quantum is needed because pot_arr will be type
        float64 in the CBnet case and type complex128 in the QBnet case.

        Parameters
        ----------
        is_quantum : bool

        Returns
        -------
        None

        """
        # to avoid setting sepset pots to one
        # twice, first set flags of
        # all sepsets to False, then
        # change flag to True for sepsets that are done.
        for clique in self.nodes:
            for sepset in clique.sepsets:
                sepset.flag = False
        for clique in self.nodes:
            clique.set_pot_to_one(is_quantum)

            # print("clique", clique.potential)

            for sepset in clique.sepsets:
                if not sepset.flag:
                    sepset.set_pot_to_one(is_quantum)
                    sepset.flag = True

                    # print("sepset", sepset.potential)

    def init_clique_potentials_with_bnet_info(self):
        """
        Once the Clique potentials have been set to one, multiply them times
        the conditional PDs or conditional PADs of the nodes of the CBnet or
        QBnet. A clique accepts only the PD or PAD of those nodes that
        contain all their family inside that clique. abbreviations in
        MyConstants.py.

        Returns
        -------
        None

        """

        for clique in self.nodes:
            for v in clique.subnodes:
                v.clique = None

        for clique in self.nodes:
            for v in clique.subnodes:
                if v.clique is None:
                    v_and_pars = v.parents | {v}  # called the family of v
                    if clique.subnodes >= v_and_pars:
                        v.clique = clique

                        # print("\nnode being absorbed:", v.name)
                        # print("its clique:", v.clique.name)
                        # print("clique pot before abs", clique.potential)
                        # old_clique_pot = cp.deepcopy(clique.potential)

                        clique.potential *= v.potential

                        # print("clique pot after abs", clique.potential)
                        # print("this should be small",
                        #       old_clique_pot*v.get_masked_pot()
                        #       - clique.potential)
        # for clique in self.nodes:
        #     print("clique pot:", clique.potential, "\n")

    def mask_clique_potentials(self):
        """
        Masks clique potentials assuming they have been constructed already.

        Returns
        -------
        None

        """
        for clique in self.nodes:
            clique.potential.mask_self()

    def describe_yourself(self):
        """
        Prints a pretty summary of the attributes of self.

        Returns
        -------

        """
        print("JoinTree:")
        for cliq in self.nodes:
            print("clique name: ", cliq.name)
            print("subnodes:",
                  sorted([node.name for node in cliq.subnodes]))
            print("sepsets:")
            for sep in cliq.sepsets:
                print(
                    sorted([node.name for node in sep.subnodes]),
                    " between ",
                    sep.clique_x.name,
                    " and ",
                    sep.clique_y.name)
            print("\n")


from examples_cbnets.HuaDar import *
if __name__ == "__main__":
    bnet = HuaDar.build_bnet()
    moral_graph = MoralGraph(bnet)
    tri_graph = TriangulatedGraph(moral_graph)
    jtree = JoinTree(tri_graph, bnet)
    jtree.describe_yourself()

