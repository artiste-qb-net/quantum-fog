# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


# from Utilities import *


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
        self.nodes = nodes

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
        Answer the question of whether set 'nodes' is a subset of
        'self.nodes'.

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
        Set the 'visited' flag of all nodes to False.

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

    def get_node_with_index(self, index):
        """
        Tries to find the node with this index.

        Parameters
        ----------
        index : int

        Returns
        -------
        Node

        """
        for node in self.nodes:
            if node.index == index:
                return node
        assert False, "There is no node with index " + str(index)

from Node import *
if __name__ == "__main__":
    p1 = Node(0, "p1")
    p2 = Node(1, "p2")
    center = Node(2, "center")
    c1 = Node(3, "c1")
    c2 = Node(4, "c2")

    g = Graph({p1})
    g.add_nodes({p2, center, c1, c2})
    assert(g.has_node(p1))
    assert(g.contains({p1, center, c2}))
