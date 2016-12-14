# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import itertools as it

from inference.InferenceEngine import *
from potentials.DiscreteUniPot import *


class EnumerationEngine(InferenceEngine):
    """
    EnumerationEngine is an InferenceEngine that enumerates all (brute force
    method) possible instantiations (aka stories, histories, Feynman paths)
    consistent with the active states of each node.

    Attributes
    ----------
    bnet_ord_nodes : list[BayesNode]

    bnet : BayesNet
    verbose : bool
    is_quantum : bool

    """

    def __init__(self, bnet, verbose=False, is_quantum=False):
        """
        Constructor

        Parameters
        ----------
        bnet : BayesNet
        verbose : bool
        is_quantum : bool

        Returns
        -------

        """
        InferenceEngine.__init__(self, bnet, verbose, is_quantum)
        self.bnet_ord_nodes = list(self.bnet.nodes)

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
        assert(set(node_list) <= self.bnet.nodes)
        pot_list = [DiscreteUniPot(self.is_quantum, node, bias=0)
                    for node in node_list]
        story_counter = 0
        total_pot_val = 0.
        for cur_story in self.story_generator():
            story_counter += 1
            annotated_story = dict(zip(
                    self.bnet_ord_nodes, cur_story))
            pot_val = self.get_story_potential_val(
                annotated_story)
            total_pot_val += pot_val
            if self.verbose:
                print("[", story_counter, "] pot_val=", pot_val)
                InferenceEngine.print_annotated_story(
                        annotated_story)
                print("\n")
            state_list = [annotated_story[v] for v in node_list]
            for (pot, state) in zip(pot_list, state_list):
                pot[state] += pot_val
        if self.verbose:
            print("tot_pot_val= ", total_pot_val,
                  "# equals 1 if you comment out the evidence")
            print("\n")
        pot_list1 = []
        for pot in pot_list:
            pot.normalize_self()
            if self.is_quantum:
                pot1 = pot.get_probs_from_amps()
            else:
                pot1 = pot
            pot_list1.append(pot1)
        return pot_list1

    def story_generator(self):
        """
        Generate the next possible story constrained by the evidence (i.e.,
        by the active states of each node).

        Returns
        -------
        itertools.product

        """

        x = [node.active_states for node in self.bnet_ord_nodes]
        return it.product(*x)

    def get_story_potential_val(self, annotated_story):
        """
        Given an annotated story (i.e., a dictionary that maps all nodes to
        their current state), it returns a float for CBnets and a complex
        for QBnet. The returned value is the pot value for that particular
        annotated story.

        Parameters
        ----------
        annotated_story : dict[Node, int]

        Returns
        -------
        complex

        """
        pot_val = 1
        for node in self.bnet.nodes:
            pot = node.potential
            states = tuple(annotated_story[v] for v in pot.ord_nodes)
            pot_val *= pot[states]
        return pot_val

from examples_cbnets.HuaDar import *
if __name__ == "__main__":

    bnet = HuaDar.build_bnet()

    # introduce some evidence
    # bnet.get_node_named("D").active_states = [0]
    # bnet.get_node_named("G").active_states = [1]

    inf_eng = EnumerationEngine(bnet, verbose=True)
    id_nums = sorted([node.id_num for node in bnet.nodes])
    node_list = [bnet.get_node_with_id_num(k) for k in id_nums]

    # this is simpler but erratic
    # node_list = list(bnet.nodes)

    pot_list = inf_eng.get_unipot_list(node_list)
    for pot in pot_list:
        print(pot, "\n")

