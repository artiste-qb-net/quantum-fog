# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import copy as cp
import networkx as nx


from graphs.Graph import *
from MyExceptions import BadGraphStructure


class Dag(Graph):
    """
    A Dag is a Graph. DAG = Directed Acyclic Graph. A mere Graph is
    undirected and is composed of Node's. A Dag is directed and is composed
    of DirectedNode's.

    Attributes
    ----------
    nodes : set[DirectedNode]

    """

    def __init__(self, nodes):
        """
        Constructor

        Parameters
        ----------
        nodes : set[DirectedNode]

        Returns
        -------

        """
        Graph.__init__(self, nodes)
        self.topological_sort()

    def detect_two_node_cycle(self):
        """
        Detects a 2 node cycle. That is, when 2 nodes are both parents and
        children of each other.

        Returns
        -------
        None

        """
        for node in self.nodes:
            overlap = node.parents & node.children
            if overlap:
                raise BadGraphStructure("two node cycle detected")

    def topological_sort(self):
        """
        Assigns topo_index to each node. This is an int between 0 and number
        of nodes -1, such that parents have lower topo_index than their
        children. The node with the lowest topo_index is always a root node.
        So this could also be called a chronological or birthday sort,
        oldest nodes first. Exception is raised if graph has cycles and
        cannot be ordered topologically.

        Returns
        -------
        None

        """

        self.detect_two_node_cycle()
        sorted_set = set()
        i = 0
        num_unsorted_nds = len(self.nodes)
        while len(self.nodes) > 0:
            if num_unsorted_nds <= 0:
                raise BadGraphStructure("Graph must be acyclic")
            num_unsorted_nds -= 1
            for node in self.nodes:
                if sorted_set >= node.parents:
                    sorted_set.add(node)
                    self.nodes.remove(node)
                    node.topo_index = i
                    i += 1
                    break
        self.nodes = sorted_set

    def undirect(self):
        """
        This just goes to each node and sets its neighbor set to be
        the union of parents and children.

        Returns
        -------
        None

        """
        for node in self.nodes:
            node.undirect()

    def add_nodes(self, nodes):
        """
        Add a set of nodes.

        Parameters
        ----------
        nodes : set[DirectedNode]

        Returns
        -------
        None

        """
        assert all([isinstance(nd, DirectedNode) for nd in nodes])
        Graph.add_nodes(self, nodes)
        self.topological_sort()

    @staticmethod
    def new_from_nx_graph(nx_graph):
        """
        Returns a Dag constructed from nx_graph.

        Parameters
        ----------
        nx_graph : networkx DiGraph

        Returns
        -------
        Dag

        """
        new_g = Dag(set())
        for k, name in enumerate(nx_graph.nodes()):
            new_g.add_nodes({DirectedNode(k, name=name)})

        node_list = list(new_g.nodes)
        for nd in node_list:
            for ch_name in nx_graph.successors(nd.name):
                nd.add_child(new_g.get_node_named(ch_name))
        return new_g

    def get_nx_graph(self):
        """
        Returns an nx_graph built from self info.

        Returns
        -------
        networkx Graph

        """

        node_list = list(self.nodes)
        nx_graph = nx.DiGraph()
        for nd in node_list:
            nx_graph.add_node(nd.name)
            for ch in nd.children:
                nx_graph.add_edge(nd.name, ch.name)
        return nx_graph

    def __str__(self):
        """
        Specifies the string outputted by print(obj) where obj is an object
        of Dag.

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
                  + "\n\n"
        return st


if __name__ == "__main__":
    from nodes.DirectedNode import *

    def main():
        p1 = DirectedNode(0, "p1")
        p2 = DirectedNode(1, "p2")
        center = DirectedNode(2, "center")
        c1 = DirectedNode(3, "c1")
        c2 = DirectedNode(4, "c2")

        center.add_parent(p1)
        center.add_parent(p2)
        center.add_child(c1)
        center.add_child(c2)
        g = Dag({p1})
        g.add_nodes({p2, center, c1, c2})
        c2.add_parent(p2)

        g.draw(algo_num=1)

        path1 = '../examples_cbnets/dag1.dot'
        path2 = '../examples_cbnets/dag2.dot'
        g.write_dot(path1)
        new_g = Dag.read_dot(path1)
        new_g.write_dot(path2)

        try:
            test = 2
            if test == 1:
                print("introduce 2 node cycle")
                c2.add_child(p2)
                g.topological_sort()
            elif test == 2:
                c2.remove_parent(p2)
                print("introduce 3 node cycle")
                c2.add_child(p2)
                g.topological_sort()
        except BadGraphStructure as txt:
            print(txt)
    main()

