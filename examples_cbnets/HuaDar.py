import numpy as np

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *
from inference.EnumerationEngine import *
from inference.MCMC_Engine import *
from inference.JoinTreeEngine import *


class HuaDar:

    @staticmethod
    def build_bnet():
        """
        Builds CBnet in accompanying gif : bnet_HuangDarwiche.gif

        From "Inference Belief Networks: A Procedural Guide", by C.Huang and
        A. Darwiche

        """

        a_node = BayesNode(0, name="A")
        b_node = BayesNode(1, name="B")
        c_node = BayesNode(2, name="C")
        d_node = BayesNode(3, name="D")
        e_node = BayesNode(4, name="E")
        f_node = BayesNode(5, name="F")
        g_node = BayesNode(6, name="G")
        h_node = BayesNode(7, name="H")

        b_node.add_parent(a_node)
        c_node.add_parent(a_node)
        d_node.add_parent(b_node)
        e_node.add_parent(c_node)
        f_node.add_parent(d_node)
        f_node.add_parent(e_node)
        g_node.add_parent(c_node)
        h_node.add_parent(e_node)
        h_node.add_parent(g_node)

        nodes = {a_node, b_node, c_node, d_node, e_node,
                f_node, g_node, h_node}

        a_node.potential = DiscreteUniPot(False, a_node)  # P(a)
        b_node.potential = DiscreteCondPot(False, [a_node, b_node])  # P(b| a)
        c_node.potential = DiscreteCondPot(False, [a_node, c_node])
        d_node.potential = DiscreteCondPot(False, [b_node, d_node])
        e_node.potential = DiscreteCondPot(False, [c_node, e_node])

        # P(f|d, e)
        f_node.potential = DiscreteCondPot(False, [d_node, e_node, f_node])

        g_node.potential = DiscreteCondPot(False, [c_node, g_node])
        h_node.potential = DiscreteCondPot(False, [e_node, g_node, h_node])

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        # off = 0
        # on = 1

        a_node.potential.pot_arr[:] = [.5, .5]

        b_node.potential.pot_arr[1, :] = [.5, .5]
        b_node.potential.pot_arr[0, :] = [.4, .6]

        c_node.potential.pot_arr[1, :] = [.7, .3]
        c_node.potential.pot_arr[0, :] = [.2, .8]

        d_node.potential.pot_arr[1, :] = [.9, .1]
        d_node.potential.pot_arr[0, :] = [.5, .5]

        e_node.potential.pot_arr[1, :] = [.3, .7]
        e_node.potential.pot_arr[0, :] = [.6, .4]

        f_node.potential.pot_arr[1, 1, :] = [.01, .99]
        f_node.potential.pot_arr[1, 0, :] = [.01, .99]
        f_node.potential.pot_arr[0, 1, :] = [.01, .99]
        f_node.potential.pot_arr[0, 0, :] = [.99, .01]

        g_node.potential.pot_arr[1, :] = [.8, .2]
        g_node.potential.pot_arr[0, :] = [.1, .9]

        h_node.potential.pot_arr[1, 1, :] = [.05, .95]
        h_node.potential.pot_arr[1, 0, :] = [.95, .05]
        h_node.potential.pot_arr[0, 1, :] = [.95, .05]
        h_node.potential.pot_arr[0, 0, :] = [.95, .05]

        return BayesNet(nodes)

if __name__ == "__main__":
    bnet = HuaDar.build_bnet()
    brute_eng = EnumerationEngine(bnet)
    # introduce some evidence
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]
    node_list = brute_eng.bnet_ord_nodes
    brute_pot_list = brute_eng.get_unipot_list(node_list)

    # bnet = HuaDar.build_bnet()
    monte_eng = MCMC_Engine(bnet)
    # introduce some evidence
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]
    num_cycles = 1000
    warmup = 200
    node_list = monte_eng.bnet_ord_nodes
    monte_pot_list = monte_eng.get_unipot_list(
            node_list, num_cycles, warmup)

    # bnet = HuaDar.build_bnet()
    jtree_eng = JoinTreeEngine(bnet)
    # introduce some evidence
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]
    node_list = jtree_eng.bnet_ord_nodes
    jtree_pot_list = jtree_eng.get_unipot_list(node_list)

    for k in range(len(node_list)):
        print(node_list[k].name)
        print("brute engine:", brute_pot_list[k])
        print("monte engine:", monte_pot_list[k])
        print("jtree engine:", jtree_pot_list[k])
        print("\n")
