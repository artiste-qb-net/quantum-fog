# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import copy as cp

from Graph import *
from MyExceptions import BadGraphStructure


class Dag(Graph):
    """
    A Dag is a Graph. DAG = Directed Acyclic Graph. A mere Graph is
    undirected and is composed of Node's. A Dag is directed and is composed
    of DirectedNode's.

    Attributes
    ----------
    nodes : set[DirectedNode]
    num_nodes : int
        number of nodes.


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
        self.num_nodes = len(self.nodes)
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
                raise BadGraphStructure(
                    "two node cycle detected")

    def topological_sort(self):
        """
        Orders nodes (permutes their indices) such that no node is before
        any of its parents. Node with lowest index number is a root node.
        Exception is raised if graph has cycles and cannot be ordered
        topologically.

        Returns
        -------
        None

        """

        self.detect_two_node_cycle()
        sorted_set = set()
        i = 0
        tot_iter = len(self.nodes)
        while len(self.nodes) > 0:
            if tot_iter <= 0:
                raise BadGraphStructure(
                    "Graph must be acyclic")
            tot_iter -= 1
            for node in self.nodes:
                if sorted_set >= node.parents:
                    sorted_set.add(node)
                    self.nodes.remove(node)
                    node.index = i
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
        Graph.add_nodes(self, nodes)
        self.topological_sort()

from DirectedNode import *
if __name__ == "__main__":
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

