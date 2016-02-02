# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import copy as cp

from Dag import *
# from BayesNode import *


class BayesNet(Dag):
    """
    This class embodies a CBnet or QBnet.  It's a DAG (Direct
    Acyclic Graph) whose nodes are of type BayesNode.

    Attributes
    ----------
    nodes : set[BayesNode]
    num_of_nodes : int

    """

    def __init__(self, nodes):
        """
        Constructor.

        Parameters
        ----------
        nodes : set[BayesNode]

        Returns
        -------

        """
        Dag.__init__(self, nodes)

from ExamplesC.HuaDar import *
if __name__ == "__main__":
    bnet = HuaDar.build_bnet()
    for node in bnet.nodes:
        print("name: ", node.name)
        print("parents: ", [x.name for x in node.parents])
        print("children: ", [x.name for x in node.children])
        print("pot_arr: \n", node.potential.pot_arr)
        print("\n")


