# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


# from nodes.BayesNode import *
# from potentials.Potential import *
from graphs.Sepset import *


class Clique(Node):
    """
    Clique inherits from Node. Cliques are clusters of subnodes. Each clique
    acts as a single node of a JoinTree. A JoinTree is an undirected graph
    whereas a BayesNet is a directed one.

    Attributes
    ----------
    potential : Potential
    sepsets : set[Sepset]
        insert into this set one sepset for each
        clique adjacent to the self clique.
    subnodes : set[BayesNode]

    id_num : int
    topo_index : int
    name : str
    neighbors : set[Node]
    visited : bool

    """

    def __init__(self, id_num, subnodes):
        """
        Constructor. Note that a clique is a Node that contains subnodes.
        Make the name of self the concatenation of the names of its subnodes
        in alphabetical order.

        Parameters
        ----------
        id_num : int
            Unique int to identify each clique.
        subnodes : set[Node]

        Returns
        -------

        """
        self.subnodes = subnodes
        joe = '_'.join(sorted([nd.name for nd in subnodes]))
        Node.__init__(self, id_num, name=joe)
        self.sepsets = set()
        self.potential = None

    def add_sepset(self, sepset):
        """
        Add sepset.

        Parameters
        ----------
        sepset : Sepset

        Returns
        -------
        None

        """
        self.sepsets.add(sepset)

    def set_pot_to_one(self, is_quantum):
        """
        Set potential to one.

        Parameters
        ----------
        is_quantum : bool
            False->CBnet, True->QBnet

        Returns
        -------
        None

        """
        # insert subnodes in arbitrary order
        self.potential = Potential(is_quantum,
                            list(self.subnodes), bias=1)

    def contains(self, nd_set):
        """
        Checks if set self.subnodes contains nd_set.

        Parameters
        ----------
        nd_set : set[Node]

        Returns
        -------
        bool

        """

        return self.subnodes >= nd_set

if __name__ == "__main__":
    print(5)
