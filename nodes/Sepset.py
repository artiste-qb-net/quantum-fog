# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


from nodes.Clique import *
from potentials.Potential import *
import heapq as he


class Sepset(Node):
    """
    A Sepset is a Node which contains subnodes. Every Sepset sits between 2
    Cliques in a JoinTree. The subnodes set of a sepset is the intersection
    of the subnodes sets of those two Cliques. The potential of a sepset
    pertains to its subnodes. A sepset is used when passing messages between
    its two cliques.

    Attributes
    ----------
    clique_x : Clique
    clique_y : Clique
    cost : float
    flag : bool
    mass : float
    potential : Potential
    subnodes : set[Node]

    """

    def __init__(self, id_num, clique_x, clique_y, subnodes):
        """
        Constructor


        Parameters
        ----------
        id_num : int
        clique_x : Clique
        clique_y : Clique
        subnodes : set[BayesNode]
            the intersection of the subnodes of clique_x and clique_y.

        Returns
        -------

        """
        assert len(subnodes) > 0
        mashup = '_'.join(sorted([nd.name for nd in subnodes]))
        Node.__init__(self, id_num, name=mashup)
        self.clique_x = clique_x
        self.clique_y = clique_y
        self.subnodes = subnodes
        self.potential = None
        self.flag = False
        self.mass = len(self.subnodes)
        cost_x = np.prod(np.array(
                [node.size for node in clique_x.subnodes],
                dtype=np.float64))
        cost_y = np.prod(np.array(
                [node.size for node in clique_y.subnodes],
                dtype=np.float64))
        self.cost = cost_x + cost_y

    def __lt__(self, other):
        """
        Overrides <. Used by JoinTree to order sepsets in a priority queue.

        Parameters
        ----------
        other : Sepset

        Returns
        -------
        bool

        """
        # for python heap, heap[0], the element given by pop(),
        # is the smallest element
        if self.mass > other.mass:
            return True
        elif self.mass == other.mass and self.cost < other.cost:
            return True
        elif self.mass == other.mass and \
                self.cost == other.cost:
            # PBMT returned False here always.
            # This would lead to erratic behavior
            return self.id_num < other.id_num
        else:
            return False

    def set_pot_to_one(self, is_quantum):
        """
        Sets self.potential to one. Needs to know is_quantum to decide
        whether to use a numpy array with float64 or complex128 entries.
        Sets pot to one over all states, not just the active ones.

        Parameters
        ----------
        is_quantum : bool

        Returns
        -------
        None

        """
        # insert subnodes into pot in arbitrary order
        self.potential = Potential(is_quantum,
                        list(self.subnodes), bias=1)

    @staticmethod
    def create_sepset_heap(cliques):
        """
        Create a sepset (with a unique id) for every pair of cliques in
        'cliques' that have a non-empty intersection. Insert those sepsets
        into a priority queue.

        Parameters
        ----------
        cliques : list[Clique]

        Returns
        -------
        list

        """

        sepset_heap = []
        id_num = 0
        for i in range(len(cliques) - 1):
            c1 = cliques[i]
            for c2 in cliques[i + 1:]:
                # PBNT was creating empty sepsets
                subnodes = c1.subnodes & c2.subnodes
                if len(subnodes) > 0:
                    sepset = Sepset(id_num, c1, c2, subnodes)
                    id_num += 1
                    he.heappush(sepset_heap, sepset)
        return sepset_heap

    def get_other_clique(self, clique):
        """
        If 'clique' is clique_x or clique_y, then return the other one.

        Parameters
        ----------
        clique : Clique

        Returns
        -------
        Clique

        """
        if clique is self.clique_x:
            return self.clique_y
        elif clique is self.clique_y:
            return self.clique_x
        else:
            assert False, "clique does not belong to sepset"

if __name__ == "__main__":
    print(5)
