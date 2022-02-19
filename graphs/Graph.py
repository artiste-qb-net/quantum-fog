# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


import pydotplus as pdp
import networkx as nx
import matplotlib.pyplot as plt
# import graphviz as gv
# from graphviz import dot


class Graph:

    """
    Graph is the parent of all other graph classes. It's just a set of
    nodes. The nodes themselves will keep track of their neighbors, parents,
    children, etc. In general, Quantum Fog will use the words 'nodes' and
    'subnodes' for sets of nodes. For lists of nodes, we will use either
    'node_list' or 'ord_nodes'.

    Attributes
    ----------
    nodes : set[Node]
    num_nodes : int
        number of nodes.

    """

    def __init__(self, nodes):
        """
        Constructor

        Parameters
        ----------
        nodes : set[Node]

        Returns
        -------

        """
        # make copy of 'nodes'
        self.nodes = set(nodes)
        self.num_nodes = len(self.nodes)

    def add_nodes(self, nodes):
        """
        Add a set 'nodes' to existing set 'self.nodes'.

        Parameters
        ----------
        nodes : set[Node]

        Returns
        -------
        None

        """
        self.nodes |= nodes
        self.num_nodes = len(self.nodes)

    def has_node(self, node):
        """
        Answer the question of whether 'node' is in 'self.nodes'.

        Parameters
        ----------
        node : Node

        Returns
        -------
        bool

        """
        return node in self.nodes

    def contains(self, nodes):
        """
        Returns True iff set 'nodes' is a subset of 'self.nodes'.

        Parameters
        ----------
        nodes : set[Node]

        Returns
        -------
        bool

        """
        return self.nodes >= nodes

    def unmark_all_nodes(self):
        """
        Set the 'visited' flag of all nodes to False. 'visited' is an
        attribute of class Node.

        Returns
        -------
        None

        """
        for node in self.nodes:
            node.visited = False

    def get_node_named(self, name):
        """
        Tries to find the node called 'name'.

        Parameters
        ----------
        name : str

        Returns
        -------
        Node

        """
        for node in self.nodes:
            if node.name == name:
                return node
        assert False, "There is no node named " + name

    def get_node_with_id_num(self, id_num):
        """
        Tries to find the node with this id_num.

        Parameters
        ----------
        id_num : int

        Returns
        -------
        Node

        """
        for node in self.nodes:
            if node.id_num == id_num:
                return node
        assert False, "There is no node with id_num " + str(id_num)

    def get_node_with_topo_index(self, topo_index):
        """
        Tries to find the node with this topo_index.

        Parameters
        ----------
        topo_index : int

        Returns
        -------
        Node

        """
        for node in self.nodes:
            if node.topo_index == topo_index:
                return node
        assert False, "There is no node with topo_index " + str(topo_index)

    @staticmethod
    def new_from_nx_graph(nx_graph):
        """
        Returns a Graph constructed from nx_graph.

        Parameters
        ----------
        nx_graph : networkx Graph

        Returns
        -------
        Graph

        """
        new_g = Graph(set())
        for k, name in enumerate(nx_graph.nodes()):
            new_g.add_nodes({Node(k, name=name)})

        node_list = list(new_g.nodes)

        for k, nd1 in enumerate(node_list):
            for nd2 in node_list[k+1:]:
                if nd1.name in nx_graph.neighbors(nd2.name):
                    nd1.add_neighbor(nd2)
        return new_g

    def get_nx_graph(self):
        """
        Returns an nx_graph built from self info.

        Returns
        -------
        networkx Graph

        """

        node_list = list(self.nodes)
        nx_graph = nx.Graph()
        for k, nd1 in enumerate(node_list):
            for nd2 in node_list[k+1:]:
                if nd2.has_neighbor(nd1):
                    nx_graph.add_edge(nd1.name, nd2.name)
        return nx_graph

    def draw(self, algo_num, **kwargs):
        """
        This method works both for undirected and directed graphs. It
        produces a plot of the self graph using only standard Python
        libraries like matplotlib. Does not require that the fantastic &
        free graphviz app be installed on your computer. If you want to use
        graphviz to get a more finely tuned plot, use the write_dot() method
        to generate a dot file which you can then fiddle with using the
        graphviz app itself.

        Parameters
        ----------
        algo_num : int
            From 1 to 6, algorithm used to determine node positions

        **kwargs : unpacked dictionary
            Look at the numerous keyword arguments of the function
            draw_networkx() in the networkx package. You can pass the same
            key-value pairs into this draw() method.

        Returns
        -------

        """
        # each of these is a networkx function for calculating node positions
        node_pos = {
            1: nx.circular_layout,
            2: nx.fruchterman_reingold_layout,
            3: nx.random_layout,
            4: nx.shell_layout,
            5: nx.spring_layout,
            6: nx.spectral_layout
        }
        nx_graph = self.get_nx_graph()
        pos = node_pos[algo_num](nx_graph)
        nx.draw_networkx(nx_graph, pos=pos, node_color='white', **kwargs)
        plt.axis('off')
        plt.show()

    def write_dot(self, path):
        """
        This produces a very basic .dot file that can be opened with
        graphviz to fine tune its layout details.

        Parameters
        ----------
        path : str
            eg. for Windows, you can use an absolute path like
            'C:/Documents and Settings/ROBERT/Desktop/tempo.dot' or a
            relative one like '../examples_cbnets/tempo.dot'

        Returns
        -------
        None

        """
        nx.nx_pydot.write_dot(self.get_nx_graph(), path)

    @classmethod
    def read_dot(cls, path):
        """
        Reads dot file and returns new cls object, where cls can be Graph,
        Dag, or BayesNet.

        Parameters
        ----------
        path : str
            eg. for Windows, you can use an absolute path like
            'C:/Documents and Settings/ROBERT/Desktop/tempo.dot' or a
            relative one like '../examples_cbnets/tempo.dot'

        Returns
        -------
        cls

        """
        nx_graph = nx.nx_pydot.read_dot(path)
        return cls.new_from_nx_graph(nx_graph)

    def print_neighbors(self):
        """
        Print neighbors of each node

        Returns
        -------
        None

        """
        for node in self.nodes:
            print("name: ", node.name)
            print("neighbors: ",
                  sorted([x.name for x in node.neighbors]))
            print("\n")

    def __str__(self):
        """
        Specifies the string outputted by print(obj) where obj is an object
        of Graph.

        Returns
        -------
        str

        """
        st = ""
        for nd in self.nodes:
            st += nd.name \
                  + ", neighbors=" \
                  + str([x.name for x in nd.neighbors]) \
                  + "\n\n"
        return st


if __name__ == "__main__":
    from nodes.Node import *

    def main():
        p1 = Node(0, "p1")
        p2 = Node(1, "p2")
        center = Node(2, "center")
        c1 = Node(3, "c1")
        c2 = Node(4, "c2")

        g = Graph({p1})
        g.add_nodes({p2, center, c1, c2})
        assert g.has_node(p1)
        assert g.contains({p1, center, c2})

        center.add_neighbor(p1)
        center.add_neighbor(p2)
        center.add_neighbor(c1)
        center.add_neighbor(c2)

        g.draw(algo_num=1)

        path1 = '../examples_cbnets/graph1.dot'
        path2 = '../examples_cbnets/graph2.dot'
        g.write_dot(path1)
        new_g = Graph.read_dot(path1)
        new_g.write_dot(path2)
    main()
