import numpy as np

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *
from inference.EnumerationEngine import *
from inference.MCMC_Engine import *
from inference.JoinTreeEngine import *


class QuWetGrass:

    @staticmethod
    def build_bnet():
        """
        Builds QBnet called QuWetGrass with diamond shape
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

        cl.potential = DiscreteUniPot(True, cl)  # P(a)
        sp.potential = DiscreteCondPot(True, [cl, sp])  # P(b| a)
        ra.potential = DiscreteCondPot(True, [cl, ra])
        we.potential = DiscreteCondPot(True, [sp, ra, we])

        # in general
        # DiscreteCondPot(True, [y1, y2, y3, x]) refers to A(x| y1, y2, y3)
        # off = 0
        # on = 1

        cl.potential.pot_arr[:] = [.5 + .1j, .5]

        ra.potential.pot_arr[1, :] = [.5 - .1j, .5 + .3j]
        ra.potential.pot_arr[0, :] = [.4, .6 - .7j]

        sp.potential.pot_arr[1, :] = [.7 + 3.j, .3 - 1.j]
        sp.potential.pot_arr[0, :] = [.2 + .5j, .8]

        we.potential.pot_arr[1, 1, :] = [.01 + 1j, .99]
        we.potential.pot_arr[1, 0, :] = [.01 - 5.j, .99]
        we.potential.pot_arr[0, 1, :] = [.01, .99 + 2.3j]
        we.potential.pot_arr[0, 0, :] = [.99, .01 - .01j]

        cl.potential.tr_normalize_self()
        ra.potential.tr_normalize_self()
        sp.potential.tr_normalize_self()
        we.potential.tr_normalize_self()

        return BayesNet(nodes)

if __name__ == "__main__":
    bnet = QuWetGrass.build_bnet()
    brute_eng = EnumerationEngine(bnet, is_quantum=True)
    # introduce some evidence
    bnet.get_node_named("WetGrass").active_states = [1]
    node_list = brute_eng.bnet_ord_nodes
    brute_pot_list = brute_eng.get_unipot_list(node_list)

    bnet = QuWetGrass.build_bnet()
    monte_eng = MCMC_Engine(bnet, is_quantum=True)
    # introduce some evidence
    bnet.get_node_named("WetGrass").active_states = [1]
    num_cycles = 1000
    warmup = 200
    node_list = monte_eng.bnet_ord_nodes
    monte_pot_list = monte_eng.get_unipot_list(
            node_list, num_cycles, warmup)

    bnet = QuWetGrass.build_bnet()
    jtree_eng = JoinTreeEngine(bnet, is_quantum=True)
    # introduce some evidence
    bnet.get_node_named("WetGrass").active_states = [1]
    node_list = jtree_eng.bnet_ord_nodes
    jtree_pot_list = jtree_eng.get_unipot_list(node_list)

    for k in range(len(node_list)):
        print(node_list[k].name)
        print("brute engine:", brute_pot_list[k])
        print("monte engine:", monte_pot_list[k])
        print("jtree engine:", jtree_pot_list[k])
        print("\n")

