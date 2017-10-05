# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


import numpy as np
import copy as cp
import random as ra

# from nodes.BayesNode import *
from inference.InferenceEngine import *
from potentials.DiscreteUniPot import *


class MCMC_Engine(InferenceEngine):
    """
    An MCMC_Engine is an InferenceEngine that uses MCMC = Markov Chain Monte
    Carlo. The algo used here is described in the textbook:

    S. Russell and P. Norvig, "Artificial Intelligence, a Modern Approach,
    3rd. Ed." (Prentice Hall, 2010) Section 14.5 "Approximate Inference in
    Bayesian Networks", page 537

    Attributes
    ----------

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

    def get_unipot_list(self, node_list, num_cycles, warmup):
        """
        For each node in node_list, this method returns a uni-potential that
        gives the probabilities for the states of that node. Obviously,
        such a PD has the active states of the node as support.

        Parameters
        ----------
        node_list : list[BayesNode]
        num_cycles : int
            How many times we should cycle through all the nodes of the graph
        warmup : int
            How many cycles should we wait before we start collecting data
            for the unipot list.

        Returns
        -------
        list[DiscreteUniPot]

        """
        assert set(node_list) <= self.bnet.nodes
        nd_to_pot = {node: DiscreteUniPot(self.is_quantum, node, bias=0)
                    for node in node_list}
        # initialize current story to a random one
        cur_story = [ra.choice(node.active_states)
                        for node in self.bnet_ord_nodes]
        # cur_story = [node.active_states[0]
        #              for node in self.bnet_ord_nodes]
        annotated_story = dict(zip(
                self.bnet_ord_nodes, cur_story))
        for cy in range(num_cycles):
            for node in self.bnet_ord_nodes:
                prev_state = annotated_story[node]
                (sam_state, sam_pot) = \
                    self.sample_node_given_markov_blanket(
                        node, annotated_story)

                if node in nd_to_pot:
                    if cy > warmup:
                        nd_to_pot[node][sam_state] += 1

                # this didn't work well
                # for nd in nd_to_pot:
                #     if cy > warmup:
                #         nd_to_pot[nd][annotated_story[nd]] += 1

                if self.verbose:
                    if num_cycles-4 < cy < num_cycles:
                        print("cycle=", cy, ", node", node.name,
                              "goes from", prev_state,
                              "to", sam_state, ", sampled from pot")
                        print(sam_pot)
                        InferenceEngine.print_annotated_story(
                                annotated_story)
                        print("\n")
        pot_list = []
        for node in node_list:
            pot = nd_to_pot[node]
            pot.tr_normalize_self()
            if self.is_quantum:
                pot = pot.get_probs_from_amps()
            pot_list.append(pot)
        return pot_list

    def sample_node_given_markov_blanket(self, focus_node, annotated_story):
        """
        For fixed values of the states of the Markov Blanket of the focus
        node, sample a possible state of the focus node. The states of the
        Markov Blanket nodes are gleaned from 'annotated_story'.

        annotated_story is a dictionary mapping bnet_ord_nodes to their
        current state.

        A story (aka history or Feynman path or bnet instantiation) is a
        list of the current states of the nodes in the list bnet_ord_nodes.

        Parameters
        ----------
        focus_node : BayesNode
        annotated_story : dict[BayesNode, int]

        Returns
        -------
        (int, DiscreteUniPot)

        """
        if len(focus_node.active_states) == 1:
            return (focus_node.active_states[0], None)
        sam_pot = DiscreteUniPot(self.is_quantum, focus_node, bias=0)
        sam_pot[focus_node.active_states] = 1
        near_nodes = focus_node.get_markov_blanket() | {focus_node}
        for state in focus_node.active_states:
            # forget previous state of focus_node
            annotated_story[focus_node] = state
            for n_node in near_nodes:
                states = [
                    annotated_story[node]
                    for node in n_node.potential.ord_nodes]
                slicex = n_node.potential.slicex_from_nds(
                        states, n_node.potential.ord_nodes)
                sam_pot[(state, )] *= n_node.potential[slicex]
        sam_state = sam_pot.sample()
        annotated_story[focus_node] = sam_state
        return (sam_state, sam_pot)

if __name__ == "__main__":
    from examples_cbnets.HuaDar import *
    bnet = HuaDar.build_bnet()
    inf_eng = MCMC_Engine(bnet, verbose=True)

    # introduce some evidence after creating engine
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]

    num_cycles = 4000
    warmup = 200
    pot_list = inf_eng.get_unipot_list(
            inf_eng.bnet_ord_nodes, num_cycles, warmup)
    for pot in pot_list:
        print(pot, "\n")
