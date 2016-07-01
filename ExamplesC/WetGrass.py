import numpy as np

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *
from inference.EnumerationEngine import *
from inference.MCMC_Engine import *
from inference.JoinTreeEngine import *


class WetGrass:

    @staticmethod
    def build_bnet():
        """
        Builds CBnet called WetGrass with diamond shape
                Cloudy
                /    \
             Rain    Sprinkler
               \      /
               WetGrass
        All arrows pointing down
        """

        cl = BayesNode(0, name="Cloudy")
        sp = BayesNode(1, name="Sprinkler")
        ra = BayesNode(2, name="Rain")
        we = BayesNode(3, name="WetGrass")

        we.add_parent(sp)
        we.add_parent(ra)
        sp.add_parent(cl)
        ra.add_parent(cl)

        nodes = {cl, ra, sp, we}

        cl.potential = DiscreteUniPot(False, cl)  # P(a)
        sp.potential = DiscreteCondPot(False, [cl, sp])  # P(b| a)
        ra.potential = DiscreteCondPot(False, [cl, ra])
        we.potential = DiscreteCondPot(False, [sp, ra, we])

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        # off = 0
        # on = 1

        cl.potential.pot_arr[:] = [.5, .5]

        ra.potential.pot_arr[1, :] = [.5, .5]
        ra.potential.pot_arr[0, :] = [.4, .6]

        sp.potential.pot_arr[1, :] = [.7, .3]
        sp.potential.pot_arr[0, :] = [.2, .8]

        we.potential.pot_arr[1, 1, :] = [.01, .99]
        we.potential.pot_arr[1, 0, :] = [.01, .99]
        we.potential.pot_arr[0, 1, :] = [.01, .99]
        we.potential.pot_arr[0, 0, :] = [.99, .01]

        return BayesNet(nodes)


if __name__ == "__main__":
    bnet = WetGrass.build_bnet()

    # introduce some evidence
    bnet.get_node_named("WetGrass").active_states = [1]

    id_nums = sorted([node.id_num for node in bnet.nodes])
    node_list = [bnet.get_node_with_id_num(k) for k in id_nums]

    # this is simpler but erratic
    # node_list = list(bnet.nodes)

    brute_eng = EnumerationEngine(bnet)
    brute_pot_list = brute_eng.get_unipot_list(node_list)

    # print("bnet pots after brute")
    # # check that bnet pots are not modified by engine
    # for node in node_list:
    #     print(node.name)
    #     print(node.potential, "\n")

    monte_eng = MCMC_Engine(bnet)
    num_cycles = 1000
    warmup = 200
    monte_pot_list = monte_eng.get_unipot_list(
            node_list, num_cycles, warmup)

    # print("bnet pots after monte")
    # # check that bnet pots are not modified by engine
    # for node in node_list:
    #     print(node.name)
    #     print(node.potential, "\n")

    jtree_eng = JoinTreeEngine(bnet)
    jtree_pot_list = jtree_eng.get_unipot_list(node_list)

    # print("bnet pots after jtree")
    # # check that bnet pots are not modified by engine
    # for node in node_list:
    #     print(node.name)
    #     print(node.potential, "\n")

    for k in range(len(node_list)):
        print(node_list[k].name)
        print("brute engine:", brute_pot_list[k])
        print("monte engine:", monte_pot_list[k])
        print("jtree engine:", jtree_pot_list[k])
        print("\n")
