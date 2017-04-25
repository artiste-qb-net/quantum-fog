# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np

from nodes.BayesNode import *


class Star:
    """
    A Star contains a focus node and some extra information about edges of
    its neighboring nodes. Since it is "centered" on one node, we call it a
    star. (PBNT calls this class an InducedCluster). In our terminology,
    once one adds the missing edges to a star, it becomes a preclique. If a
    preclique is maximal (meaning that there are no cliques that contain
    it), it is a clique.

    Attributes
    ----------
    medges : list[set[BayesNode]]
        medges stands for missing edges
    node : BayesNode
    num_medges : int
        number of missing edges
    weight : float

    """

    def __init__(self, node):
        """
        Constructor

        Parameters
        ----------
        node : BayesNode

        Returns
        -------

        """

        self.node = node
        self.medges = Star.get_missing_edges_of_node(node)
        self.num_medges = len(self.medges)
        self.weight = self.compute_weight()

    def __lt__(self, other):
        """
        Overrides <. Used by TriangulatedGraph to order Stars in a
        priority queue.

        Parameters
        ----------
        other : Star

        Returns
        -------
        bool

        """
        # less than means that it is better (pick it first)
        if self.num_medges < other.num_medges:
            return True
        elif self.num_medges == other.num_medges \
                and self.weight < other.weight:
            return True
        elif self.num_medges == other.num_medges \
                and self.weight == other.weight:
            # PBMT returned False here always.
            # This would lead to erratic behavior
            return self.node.id_num < other.node.id_num
        else:
            return False

    def recompute(self):
        """
        Recalculate Star internals after a node has been removed from the
        graph.

        Returns
        -------
        None

        """
        self.medges = Star.get_missing_edges_of_node(self.node)
        self.num_medges = len(self.medges)
        self.weight = self.compute_weight()

    def compute_weight(self):
        """
        Compute self.weight

        Returns
        -------
        float

        """
        # merge two lists and covert to np.array so can
        # multiply its entries using np.prod
        return np.prod(np.array(
            [nd.size for nd in self.node.neighbors] +
            [self.node.size], dtype=np.float64
            ))

    @staticmethod
    def get_missing_edges_of_node(nd):
        """
        Returns list of missing edges (medges) of node 'nd". By missing
        edges we mean edges that don't exist between pairs of 'nd' parents.
        By an edge we mean a set of two nodes.

        Parameters
        ----------
        nd : BayesNode

        Returns
        -------
        list[set[BayesNode]]

        """
        edges = list()
        neig_list = list(nd.neighbors)
        for k in range(len(neig_list)):
            n1 = neig_list[k]
            for n2 in neig_list[k+1:]:
                if not n2.has_neighbor(n1):
                    edges.append({n1, n2})
        return edges

if __name__ == "__main__":

    a = BayesNode(0, name="a")
    b = BayesNode(1, name="b")
    c = BayesNode(2, name="c")

    a.add_neighbor(b)
    a.add_neighbor(c)

    assert Star.get_missing_edges_of_node(a) == [{b, c}]
