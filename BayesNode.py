# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

from DirectedNode import *


class BayesNode(DirectedNode):
    """
    A BayesNode is a DirectedNode wih additional information such as a
    Potential. A Potential is either a PD in the CBnet case, or a PAD in the
    QBnet case. abbreviations in MyConstants.py. value of node = state of node.


    Attributes
    ----------
    __active_states : list[int]
        When all states are active, use range(self.size)
    clique : Clique
        In a JoinTree, the family of any node belongs
        to exactly one clique, stored here. The family of a node
        is the node and its parents. The nodes on which a
        node's potential depends are its family.
    potential : Potential
    size : int
        number of values = number of states of node =
        size of node
    state_names : list[str]

    children : set[Node]
    neighbors : set[Node]
    parents : set[Node]
    id_num : int
    index : int
    name : str
    visited : bool
    """

    def __init__(self, id_num, name="blank"):
        """
        Constructor.

        Parameters
        ----------
        id_num : int
            This should be a different int for each node.
        name : str
            The name of the node. Optional.

        Returns
        -------

        """

        DirectedNode.__init__(self, id_num, name)
        self.size = 2
        self.state_names = ["state0", "state1"]
        self.clique = None
        self.potential = None
        self.__active_states = [0, 1]  # underscore to use @property

    def set_potential(self, pot):
        """

        Parameters
        ----------
        pot : Potential

        Returns
        -------
        None

        """
        self.potential = pot
        assert(pot.nd_sizes[-1] == self.size)

    def resize(self, size):
        """
        Add or remove nodes taking care to change everything that depends on
        number of nodes.

        Parameters
        ----------
        size : int

        Returns
        -------
        None

        """
        if size < self.size:
            self.state_names = self.state_names[:size]
        elif size == self.size:
            pass
        else:
            self.state_names += [
                "state" + str(k) for k in range(self.size, size)]
        self.size = size

    def set_state_name(self, position, name):
        """
        Changes name of state of node.

        Parameters
        ----------
        position : int
        name : str

        Returns
        -------
        None

        """
        assert(0 <= position < self.size)
        self.state_names[position] = name

    def forget_all_evidence(self):
        """
        Expand list of self's active states to include all possible states from
        0 to size-1.

        Returns
        -------
        None

        """
        self.active_states = range(self.size)

    def get_active_states(self):
        """
        Getter for active state list.

        Returns
        -------
        list[int]

        """
        return self.__active_states

    def set_active_states(self, states):
        """
        Setter for active state list.

        Parameters
        ----------
        states : list[int]

        Returns
        -------
        None

        """
        assert(max(states) < self.size and min(states) >= 0)
        assert(len(states) >= 1)
        self.__active_states = states

    active_states = property(get_active_states, set_active_states)

# These imports are here after the class definition
# to avoid import loops.
# The idea is to define the class before these imports occur.
from Potential import *
from DiscreteCondPot import *
if __name__ == "__main__":
    a = BayesNode(0, "alice")
    print("a id_num=", a.id_num)
    print(a.state_names)

    a.resize(4)
    print(a.state_names)

    a.resize(3)
    print(a.state_names)

    a.set_state_name(2, "horse")
    print(a.state_names)

