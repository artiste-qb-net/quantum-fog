# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


class Node:
    """
    In its most basic form, a Graph is just a set of Node's. A BayesNode is
    a DirectedNode is a Node. Similarly, a BayesNet is a Dag is a Graph.


    Attributes
    ----------
    id_num : int
        Should be an int that is immutable and different for each node.
    name : str
        Optional, preferably different for each node.
    neighbors : set[Nodes]
    topo_index : int
        int between 0 and num of nodes-1. parents always have lower
        topo_index than chidren.
    visited : bool

    """

    def __init__(self, id_num, name="blank"):
        """
        Constructor

        Parameters
        ----------
        id_num : int
        name : str

        Returns
        -------

        """

        self.id_num = id_num
        self.topo_index = id_num
        self.name = name
        self.neighbors = set()
        self.visited = False

    def __lt__(self, other):
        """
        Defines a < operator for Node's. Used in topological sorting.

        Parameters
        ----------
        other : Node

        Returns
        -------
        bool

        """

        return self.topo_index < other.topo_index

    def add_neighbor(self, node):
        """
        Make 'node' a neighbor of self and vice versa if they aren't the
        same node.

        Parameters
        ----------
        node : Node

        Returns
        -------
        None

        """
        if (node not in self.neighbors) and (self != node):
            self.neighbors.add(node)
            node.neighbors.add(self)

    def add_neighbors(self, node_list):
        """
        Add neighbor nodes from a list of them.

        Parameters
        ----------
        node_list : list(Node)

        Returns
        -------
        None

        """
        for nd in node_list:
            self.add_neighbor(nd)

    def remove_neighbor(self, node):
        """
        Remove 'node' from the list of neighbors, effectively deleting
        that edge from the graph.

        Parameters
        ----------
        node : Node

        Returns
        -------
        None

        """

        self.neighbors.remove(node)
        node.neighbors.remove(self)

    def has_neighbor(self, node):
        """
        Check if 'node' is a neighbor.

        Parameters
        ----------
        node : Node

        Returns
        -------
        bool

        """
        return node in self.neighbors

if __name__ == "__main__":
    def main():
        a = Node(0, name="a")
        b = Node(1, name="b")
        c = Node(2, name="c")

        a.add_neighbor(b)
        a.add_neighbor(c)
        aa = a
        assert aa == a
        assert a != b
        assert a.has_neighbor(b)
        a.remove_neighbor(b)
        assert not a.has_neighbor(b)
    main()


