# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import numpy as np
# import heapq as he
# import copy as cp

# from potentials.DiscreteUniPot import *
#  from graphs.TriangulatedGraph import *
# from nodes.Sepset import *
from graphs.JoinTree import *
# from graphs.BayesNet import *
# from graphs.Graph import *
from graphs.MoralGraph import *
from inference.InferenceEngine import *


class JoinTreeEngine(InferenceEngine):
    """
    Our implementation of the Join Tree (or Junction Tree) inference
    algorithm follows very closely the very detailed and clear reference:

    "Belief Networks: A Procedural Guide" By Cecil Huang an Adnan Darwiche (
    1996).

    As far as I know, the Join Tree algorithm has only been used in the past
    for CBnets, but this computer program applies it to both CBnets and
    QBnets with only a few modifications. Most of the steps of the algorithm
    are topological (or graph theoretic) in nature and those steps apply to
    both the CBnet and QBnet cases. The main difference between CBnets and
    QBnets arises whenever the norm of a potential is required and there one
    simply uses the 1-norm for classical and the 2-norm for quantum.

    This algorithm first creates a MoralGraph, then a TriangulatedGraph,
    then a JoinTree. A list of UniPots is then computed after performing a
    global propagation on the JoinTree.

    Attributes
    ----------
    jtree : JoinTree

    """

    def __init__(self, bnet, verbose=False, is_quantum=False):
        """
        Constructor. Note that the constructor of every inference engine is
        designed so that one of its objects can be created just once at the
        beginning and then reused to calculate probabilities under several
        evidence assumptions.

        Parameters
        ----------
        bnet : BayesNet
        verbose : bool
        is_quantum : bool

        Returns
        -------

        """
        InferenceEngine.__init__(self, bnet, verbose, is_quantum)
        moral_graph = MoralGraph(self.bnet)
        tri_graph = TriangulatedGraph(moral_graph)
        self.jtree = JoinTree(tri_graph, bnet)
        if verbose:
            print("------------------triangulated graph:")
            tri_graph.print_neighbors()
            print("------------------JoinTree:")
            self.jtree.describe_yourself()

    def get_unipot_list(self, node_list):
        """
        For each node in node_list, this method returns a uni-potential that
        gives the probabilities for the states of that node. Obviously,
        such a PD has the active states of the node as support.

        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        list[DiscreteUniPot]

        """
        self.global_propagation()

        pot_list = []
        for node in node_list:
            cl_pot = cp.deepcopy(node.clique.potential)

            # print("\nnode:", node.name)
            # print("clique pot initial", cl_pot)

            pot = cl_pot.get_new_marginal([node])

            # print("its marginal", pot)

            pot1 = DiscreteUniPot(self.is_quantum, node, pot_arr=pot.pot_arr)
            pot1.normalize_self()

            # print("normalized marg", pot1)

            if self.is_quantum:
                pot1 = pot1.get_probs_from_amps()
            pot_list.append(pot1)
        return pot_list

    def global_propagation(self):
        """
        Given the JoinTree, this method does all the calculations necessary
        to give to each clique and sepset a potential suitable for
        marginalization.

        Returns
        -------
        None

        """
        self.jtree.set_clique_and_sepset_pots_to_one(self.is_quantum)
        self.jtree.init_clique_potentials_with_bnet_info()

        # Here and only here is where we introduce the
        # evidence. To handle multiple evidence cases,
        # can use same jointree but must do a global propagation
        # for each evidence case.

        # Once Clique pots are masked, the new sepset pots
        # obtained by marginalizing clique pots become masked too,
        # so masking sepset pots is unnecessary.

        self.jtree.mask_clique_potentials()

        # print("\ndeviation before global prop",
        #     self.get_deviation())

        # pick start_clique to be one owned by a node
        # with the lowest topological index number

        min_nd_topo_index = min([node.topo_index for node in self.bnet.nodes])
        start_clique = self.bnet.get_node_with_topo_index(
                                        min_nd_topo_index).clique
        if self.verbose:
            print("start clique", start_clique.name)

        self.jtree.unmark_all_nodes()

        if self.verbose:
            print("\nNext will pass messages towards start_clique")
        # Below, from_clique=None, to_clique=start_clique, sepset=None
        self.collect_evidence(None, start_clique, None)
        self.jtree.unmark_all_nodes()

        if self.verbose:
            print("\nNext will pass messages away from start_clique")

        self.distribute_evidence(start_clique)

    # def get_deviation(self):
    #     """
    #     For debugging purposes only
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     full_sep_pot = Potential(list(self.bnet.nodes), bias=1)
    #     for clique in self.jtree.nodes:
    #         for sepset in clique.sepsets:
    #             sepset.flag = False
    #     for clique in self.jtree.nodes:
    #         for sepset in clique.sepsets:
    #             if not sepset.flag:
    #                 full_sep_pot *= sepset.potential
    #                 sepset.flag = True
    #
    #     full_bnet_pot = Potential(list(self.bnet.nodes), bias=1)
    #     for node in self.bnet.nodes:
    #         full_bnet_pot *= node.potential
    #     full_bnet_pot.mask_self()
    #
    #     full_jtree_pot = Potential(list(self.bnet.nodes), bias=1)
    #     for clique in self.jtree.nodes:
    #         full_jtree_pot *= clique.potential
    #     full_jtree_pot /= full_sep_pot
    #
    #     return Potential.distance(full_jtree_pot, full_bnet_pot)

    def pass_message(self, from_clique, to_clique, sepset):
        """
        Pass a message from 'from_clique' to 'to_clique' connected by 'sepset'


        Parameters
        ----------
        from_clique : Clique
        to_clique : Clique
        sepset : Sepset

        Returns
        -------
        None

        """

        # deepcopy for pots has been defined so that
        # no deepcopy of nodes, only of pot_arr
        old_sepset_pot = cp.deepcopy(sepset.potential)

        sepset.potential = from_clique.potential.get_new_marginal(
                sepset.potential.ord_nodes)
        # if self.verbose:
        #     old_to_clique_pot = cp.deepcopy(to_clique.potential)

        # Absorb the sepset pot ratio into the to_clique pot

        to_clique.potential *= (sepset.potential/old_sepset_pot)

        if self.verbose:
            print("\npassing message from ",
                  from_clique.name, " to ", to_clique.name)

        # print("deviation after message was passed =", self.get_deviation())
        # print("from_clique pot", from_clique.potential)
        # print("new sepset pot", sepset.potential)
        # print("old sepset pot", old_sepset_pot)
        # print("old to_clique pot:", old_to_clique_pot)
        # print("new to_clique pot:", to_clique.potential)

    def collect_evidence(self, from_clique, to_clique,
                         sepset, clique_counter=1):
        """
        Pass messages from outer cliques towards the start clique.

        Parameters
        ----------
        from_clique : Clique | None
        to_clique : Clique
        sepset : Sepset | None
        clique_counter : int

        Returns
        -------
        None

        """
        if clique_counter > len(self.jtree.nodes):
            return None
        else:
            to_clique.visited = True
            for sep in to_clique.sepsets:
                # Do a DFS search of the tree, only visiting
                # unvisited clique nodes
                neighbor_cliq = sep.get_other_clique(to_clique)
                if not neighbor_cliq.visited:
                    self.collect_evidence(
                        to_clique, neighbor_cliq, sep, clique_counter + 1)
            # After we have iterated
            # over all neighbors, send back a message from each
            # back towards the start clique
            if clique_counter > 1:
                self.pass_message(to_clique, from_clique, sepset)

    def distribute_evidence(self, cur_clique, clique_counter=1):
        """
        Pass messages away from the start clique.

        Parameters
        ----------
        cur_clique : Clique
        clique_counter : int

        Returns
        -------
        None

        """
        if clique_counter > len(self.jtree.nodes):
            return None
        else:
            cur_clique.visited = True
            for sep in cur_clique.sepsets:
                # Do a DFS search of the tree, only visiting
                # unvisited clique nodes
                neighbor_cliq = sep.get_other_clique(cur_clique)
                if not neighbor_cliq.visited:
                    self.pass_message(cur_clique, neighbor_cliq, sep)
                    self.distribute_evidence(neighbor_cliq, clique_counter + 1)

if __name__ == "__main__":
    from examples_cbnets.HuaDar import *
    bnet = HuaDar.build_bnet()
    inf_eng = JoinTreeEngine(bnet, verbose=True)

    # introduce some evidence after creating engine
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]

    pot_list = inf_eng.get_unipot_list(inf_eng.bnet_ord_nodes)
    for pot in pot_list:
        print(pot, "\n")
