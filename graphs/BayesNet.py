# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import copy as cp
import networkx as nx
import pandas as pd

from graphs.Dag import *
from nodes.BayesNode import *
from BifTool import *
from DotTool import *
from potentials.DiscreteUniPot import *


class BayesNet(Dag):
    """
    This class embodies a CBnet or QBnet.  It's a DAG (Direct
    Acyclic Graph) whose nodes are of type BayesNode.

    Attributes
    ----------

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

    def __deepcopy__(self, memo):
        """
        Creates deep copy of self.

        Parameters
        ----------
        memo : dict

        Returns
        -------
        BayesNet

        """
        nd_to_new_nd = {}
        for nd in self.nodes:
            new_node = BayesNode(nd.id_num, nd.name)
            # important to set new node sizes before
            # constructing new node potentials
            new_node.size = nd.size
            nd_to_new_nd[nd] = new_node
        for nd, new_nd in nd_to_new_nd.items():
            new_nd.neighbors = set([nd_to_new_nd[nd1]
                                    for nd1 in nd.neighbors])
            new_nd.topo_index = nd.topo_index
            new_nd.visited = nd.visited
            new_nd.children = set([nd_to_new_nd[nd1]
                                   for nd1 in nd.children])
            new_nd.parents = set([nd_to_new_nd[nd1]
                                  for nd1 in nd.parents])
            # not in Dag:
            new_nd.active_states = [x for x in nd.active_states]
            new_pot_arr = cp.deepcopy(nd.potential.pot_arr)
            new_ord_nodes = [nd_to_new_nd[nd1]
                             for nd1 in nd.potential.ord_nodes]
            new_nd.potential = Potential(nd.potential.is_quantum,
                                         ord_nodes=new_ord_nodes,
                                         pot_arr=new_pot_arr)
            new_nd.state_names = [x for x in nd.state_names]
            # print("8899t", new_nd.name, new_nd.size,
            #       new_nd.potential.pot_arr.shape)
        return BayesNet(set(nd_to_new_nd.values()))

    def add_nodes(self, nodes):
        """
        Add a set of nodes.

        Parameters
        ----------
        nodes : set[BayesNode]

        Returns
        -------
        None

        """
        assert all([isinstance(nd, BayesNode) for nd in nodes])
        Graph.add_nodes(self, nodes)
        self.topological_sort()

    def get_vtx_to_state_names(self):
        """
        Returns a dictionary mapping node names to the list of their state
        names

        Returns
        -------
        dict[str, list[str]]

        """

        return {nd.name: nd.state_names for nd in self.nodes}

    def import_nd_state_names(self, vtx_to_states):
        """
        Enters vtx_to_states information into bnet.

        Parameters
        ----------
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.

        Returns
        -------
        None

        """
        for nd in self.nodes:
            nd.state_names = vtx_to_states[nd.name]
            nd.size = len(nd.state_names)

    def learn_nd_state_names(self, states_df):
        """
        Compiles an alphabetically ordered list of the unique names in each
        column of states_df and makes those the state names of the
        corresponding node in bnet.

        This function inserts into self the list of state names in
        alphabetical order. If self is an empirical net learned from
        states_df, self will only match the true, original bnet if state
        names of true net are in alphabetical order too.

        Parameters
        ----------
        states_df : pandas.DataFrame

        Returns
        -------
        None

        """
        for nd in self.nodes:
            # must turn numpy array to list
            nd.state_names = sorted(list(pd.unique(states_df[nd.name])))
            nd.size = len(nd.state_names)

    @staticmethod
    def new_from_nx_graph(nx_graph):
        """
        Returns a BayesNet constructed from nx_graph.

        Parameters
        ----------
        nx_graph : networkx DiGraph

        Returns
        -------
        BayesNet

        """
        new_g = BayesNet(set())
        for k, name in enumerate(nx_graph.nodes()):
            new_g.add_nodes({BayesNode(k, name=name)})

        node_list = list(new_g.nodes)
        for nd in node_list:
            for ch_name in nx_graph.successors(nd.name):
                nd.add_child(new_g.get_node_named(ch_name))
        return new_g

    def get_nx_graph(self):
        """
        This method returns an nx.DiGraph with the same structure as self.

        Returns
        -------
        nx.DiGraph

        """
        nx_graph = nx.DiGraph()
        for pa_nd in self.nodes:
            for ch_nd in pa_nd.children:
                nx_graph.add_edge(pa_nd.name, ch_nd.name)
        return nx_graph

    @staticmethod
    def read_bif(path, is_quantum):
        """
        Reads a bif file using our stand-alone class BifTool and returns a
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

        bt = BifTool(is_quantum)
        bt.read_bif(path)
        nodes = set()
        name_to_nd = {}
        for k, nd_name in enumerate(bt.nd_sizes.keys()):
            node = BayesNode(k, nd_name)
            node.state_names = bt.states[nd_name]
            node.size = len(node.state_names)
            node.forget_all_evidence()
            nodes |= {node}
            name_to_nd[nd_name] = node
        for nd_name, pa_name_list in bt.parents.items():
            node = name_to_nd[nd_name]
            for pa_name in pa_name_list:
                pa = name_to_nd[pa_name]
                node.add_parent(pa)

        for nd_name, parent_names in bt.parents.items():
            node = name_to_nd[nd_name]
            num_pa = len(parent_names)
            parents = [name_to_nd[pa_name] for pa_name in parent_names]
            if num_pa == 0:
                node.potential = DiscreteUniPot(is_quantum, node)
            else:
                node.potential = DiscreteCondPot(
                    is_quantum, parents + [node])
            node.potential.pot_arr = bt.pot_arrays[nd_name]

        return BayesNet(nodes)

    def write_bif(self, path, is_quantum):
        """
        Writes a bif file using BifTool class. Complements read_bif().

        Parameters
        ----------
        path : str
        is_quantum : bool

        Returns
        -------

        """
        bt = BifTool(is_quantum)
        for node in self.nodes:
            bt.nd_sizes[node.name] = node.size
            bt.states[node.name] = node.state_names
            parent_names = \
                [nd.name for nd in node.potential.ord_nodes[:-1]]
            bt.parents[node.name] = parent_names
            bt.pot_arrays[node.name] = node.potential.pot_arr
        bt.write_bif(path)

    def gv_draw(self, jupyter=True):
        """
        This method uses graphviz to draw self. It creates a temporary file
        called tempo.png with a png of self. If jupyter=True, it embeds the png
        in a jupyter notebook. If jupyter=False, it opens a window showing the
        png.

        Parameters
        ----------
        jupyter : bool

        Returns
        -------
        None

        """
        path = "tempo.dot"
        DotTool.write_dot_file_from_nx_graph(self.get_nx_graph(), path)
        DotTool.draw(path, jupyter=jupyter)

    def __str__(self):
        """
        Specifies the string outputted by print(obj) where obj is an object
        of BayesNet.

        Returns
        -------
        str

        """
        st = ""
        for nd in self.nodes:
            st += nd.name \
                  + ", parents=" \
                  + str([x.name for x in nd.parents]) \
                  + ", children=" \
                  + str([x.name for x in nd.children]) \
                  + "\n" \
                  + str(nd.potential) \
                  + "\n\n"
        return st


if __name__ == "__main__":
    from examples_cbnets.HuaDar import *
    # from examples_qbnets.QuWetGrass import *

    def main():

        bnet = HuaDar.build_bnet()
        for node in bnet.nodes:
            print("name: ", node.name)
            print("parents: ", [x.name for x in node.parents])
            print("children: ", [x.name for x in node.children])
            print("pot_arr: \n", node.potential.pot_arr)
            print("\n")

        bnet.draw(algo_num=2)

        path1 = '../examples_cbnets/bnet1.dot'
        path2 = '../examples_cbnets/bnet2.dot'
        bnet.write_dot(path1)
        new_bnet = BayesNet.read_dot(path1)
        new_bnet.write_dot(path2)

        path = '../examples_cbnets/HuaDar.bif'
        path1 = '../examples_cbnets/HuaDar1.bif'
        new_bnet = BayesNet.read_bif(path, False)
        new_bnet.write_bif(path1, False)

        # path = '../examples_qbnets/QuWetGrass.bif'
        # path1 = '../examples_qbnets/QuWetGrass1.bif'
        # new_bnet = BayesNet.read_bif(path, True)
        # new_bnet.write_bif(path1, True)

        nx_graph = new_bnet.get_nx_graph()
        print(nx_graph)

        copy_bnet = cp.deepcopy(new_bnet)
        print("copy_bnet\n", copy_bnet)

    main()

