import numpy as np

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *


class SimpleTree7nd:

    @staticmethod
    def build_bnet():
        """
        Builds simple 7 node binary tree
                     a0
                   /    \
                  b0    b1
                 /  \  /  \
                c0  c1,c2 c3

        All arrows pointing down
        """

        a0 = BayesNode(0, name="a0")

        b0 = BayesNode(1, name="b0")
        b1 = BayesNode(2, name="b1")

        c0 = BayesNode(3, name="c0")
        c1 = BayesNode(4, name="c1")
        c2 = BayesNode(5, name="c2")
        c3 = BayesNode(6, name="c3")

        a0.add_children([b0, b1])
        b0.add_children([c0, c1])
        b1.add_children([c2, c3])

        nodes = {a0, b0, b1, c0, c1, c2, c3}

        a0.potential = DiscreteUniPot(False, a0)  # P(a)

        b0.potential = DiscreteCondPot(False, [a0, b0])  # P(b| a)
        b1.potential = DiscreteCondPot(False, [a0, b1])

        c0.potential = DiscreteCondPot(False, [b0, c0])
        c1.potential = DiscreteCondPot(False, [b0, c1])
        c2.potential = DiscreteCondPot(False, [b1, c2])
        c3.potential = DiscreteCondPot(False, [b1, c3])

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        # off = 0
        # on = 1

        a0.potential.pot_arr = np.array([.7, .3])

        b0.potential.pot_arr = np.array([[.5, .5], [.4, .6]])
        b1.potential.pot_arr = np.array([[.2, .8], [.4, .6]])

        c0.potential.pot_arr = np.array([[.3, .7], [.4, .6]])
        c1.potential.pot_arr = np.array([[.5, .5], [.9, .1]])
        c2.potential.pot_arr = np.array([[.8, .2], [.4, .6]])
        c3.potential.pot_arr = np.array([[.5, .5], [.6, .4]])

        return BayesNet(nodes)


if __name__ == "__main__":
    bnet = SimpleTree7nd.build_bnet()
    for nd in bnet.nodes:
        print('\n node', nd.name)
        print(nd.potential)
    bnet.draw(algo_num=1)
