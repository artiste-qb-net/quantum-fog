# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import numpy as np
# import copy as cp

from Potential import *
# from DiscreteUniPot import *


class DiscreteCondPot(Potential):
    """
    CondPot = Conditional Potential. A CondPot is a Potential that stores a
    node as a focus node and checks to make sure that that node remains the
    last node of its ord_nodes and pot_arr. CondPots can hold either
    conditional PDs like P(x| y_1, y_2, ..) or conditional PADs like A(x|
    y_1, y_2, ...), where x is the focus node. abbreviations in
    MyConstants.py. CondPots need not be normalized, but there is a method
    in the class to normalize them. Normalization depends on whether we are
    dealing with CNets or QNets, which correspondS to is_quantum= False or
    True respectively.

    Attributes
    ----------
    focus_node : Node

    is_quantum : bool
    nd_sizes : int
    nodes : set[Node]
    num_nodes : int
    ord_nodes : list[Node]
    pot_arr : numpy.ndarray

    """

    def __init__(self, is_quantum, ord_nodes, pot_arr=None, bias=1):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        ord_nodes : list[Node]
        pot_arr : numpy.ndarray
        bias : complex

        Returns
        -------

        """
        Potential.__init__(self, is_quantum, ord_nodes, pot_arr, bias)
        self.focus_node = self.ord_nodes[-1]

    def get_probs_from_amps(self):
        """
        First checks that is_quantum=True and if so returns a new
        DiscreteCondPot calculated by taking the magnitude squared of
        self.pot_arr. Thus, this function gets probabilities from amplitudes.

        Returns
        -------
        DiscreteCondPot

        """
        assert self.is_quantum
        arr = self.pot_arr
        arr2 = (arr*np.conjugate(arr)).real
        return DiscreteCondPot(
            False, self.ord_nodes, pot_arr=arr2)

    def normalize_self(self):
        """
        This normalizes pot_arr so that it becomes a conditional potential (
        a conditional PD for is_quantum=False or a conditional PAD for
        is_quantum=True) of last node given the others. Last node
        corresponds to last axis which corresponds to innermost bracket of
        pot_arr.

        Parameters
        ----------

        Returns
        -------
        None

        """

        # print("inside normalize_self")
        # print("pot before", self)

        assert(self.focus_node == self.ord_nodes[-1])

        if self.num_nodes == 1:
            if not self.is_quantum:
                d = self.pot_arr.sum()
            else:
                d = np.linalg.norm(self.pot_arr)
            if abs(d) > TOL:
                self.pot_arr /= d
            else:
                raise ZeroDivisionError
        else:
            ind_gen = cartesian_product(self.nd_sizes[:-1])
            axes = list(range(self.num_nodes - 1))
            for indices in ind_gen:
                slicex = self.get_slicex_ax(indices, axes)
                arr = self.pot_arr[slicex]
                if not self.is_quantum:
                    d = arr.sum()
                else:
                    d = np.sqrt((arr*np.conjugate(arr)).sum())
                if abs(d) > TOL:
                    self.pot_arr[slicex] /= d
                else:
                    raise ZeroDivisionError

        # print("pot after", self, "\n")

    def __deepcopy__(self, memo):
        """
        We want deepcopy to produce a copy of pot_arr but not of the nodes
        in self.nodes so need to override the usual deepcopy.

        Parameters
        ----------
        memo :

        Returns
        -------
        DiscreteCondPot

        """
        copy_pot_arr = cp.deepcopy(self.pot_arr)
        return DiscreteCondPot(self.is_quantum,
                    ord_nodes=self.ord_nodes, pot_arr=copy_pot_arr)


if __name__ == "__main__":
    x = [0, 1, 2, 3, 4]
    print(x[:-1])

