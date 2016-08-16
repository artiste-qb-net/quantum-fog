# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import copy as cp
import networkx as nx

from graphs.Dag import *
# from nodes.BayesNode import *
from Qubifer import *
from potentials.DiscreteUniPot import *


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

    def get_vtx_to_state_names(self):
        """
        Returns a dictionary mapping node names to the list of their state
        names

        Returns
        -------
        dict[str, list[str]]

        """

        return {nd.name: nd.state_names for nd in self.nodes}

    @staticmethod
    def new_from_nx_graph(nx_graph):
        """
        Returns a BayesNet constructed from nx_graph.

        Parameters
        ----------
        nx_graph : networkx Graph

        Returns
        -------
        BayesNet

        """
        dagger = Dag.new_from_nx_graph(nx_graph)
        return BayesNet(dagger.nodes)

    @staticmethod
    def read_bif(path, is_quantum):
        """
        Reads a bif file using our stand-alone class Qubifer and returns a
        BayesNet. bif and dot files complement each other. bif: graphical
        info No, pot info Yes. dot: graphical info Yes, pot info No. By pots
        I mean potentials, the transition matrices of the nodes. (aka CPTs,
        etc.)

        Parameters
        ----------
        path : str
        is_quantum : bool

        Returns
        -------
        BayesNet

        """

        qb = Qubifer(is_quantum)
        qb.read_bif(path)
        k = -1
        nodes = set()
        name_to_nd = {}
        for nd_name in qb.nd_sizes:
            k += 1
            node = BayesNode(k, nd_name)
            node.state_names = qb.states[nd_name]
            nodes |= {node}
            name_to_nd[nd_name] = node
        for nd_name, pa_name_list in qb.parents.items():
            node = name_to_nd[nd_name]
            for pa_name in pa_name_list:
                pa = name_to_nd[pa_name]
                node.add_parent(pa)

        for nd_name, parent_names in qb.parents.items():
            node = name_to_nd[nd_name]
            num_pa = len(parent_names)
            parents = [name_to_nd[pa_name] for pa_name in parent_names]
            if num_pa == 0:
                node.potential = DiscreteUniPot(is_quantum, node)
            else:
                node.potential = DiscreteCondPot(
                    is_quantum, parents + [node])
            node.potential.pot_arr = qb.pot_arrays[nd_name]

        return BayesNet(nodes)

    def write_bif(self, path, is_quantum):
        """
        Writes a bif file using Qubifer class. Complements read_bif().

        Parameters
        ----------
        path : str
        is_quantum : bool

        Returns
        -------

        """
        qb = Qubifer(is_quantum)
        for node in self.nodes:
            qb.nd_sizes[node.name] = node.size
            qb.states[node.name] = node.state_names
            parent_names = \
                [nd.name for nd in node.potential.ord_nodes[:-1]]
            qb.parents[node.name] = parent_names
            qb.pot_arrays[node.name] = node.potential.pot_arr
        qb.write_bif(path)


from examples_cbnets.HuaDar import *
from examples_qbnets.QuWetGrass import *
if __name__ == "__main__":
    bnet = HuaDar.build_bnet()
    for node in bnet.nodes:
        print("name: ", node.name)
        print("parents: ", [x.name for x in node.parents])
        print("children: ", [x.name for x in node.children])
        print("pot_arr: \n", node.potential.pot_arr)
        print("\n")

    bnet.draw(algo_num=2)

    path1 = '../examples_cbnets/dot_test1.dot'
    path2 = '../examples_cbnets/dot_test2.dot'
    bnet.write_dot(path1)
    new_bnet = BayesNet.read_dot(path1)
    new_bnet.write_dot(path2)

    path = '../examples_cbnets/HuaDar.bif'
    path1 = '../examples_cbnets/HuaDar1.bif'
    new_bnet = BayesNet.read_bif(path, False)
    new_bnet.write_bif(path1, False)

    path = '../examples_qbnets/QuWetGrass.bif'
    path1 = '../examples_qbnets/QuWetGrass1.bif'
    new_bnet = BayesNet.read_bif(path, True)
    new_bnet.write_bif(path1, True)

