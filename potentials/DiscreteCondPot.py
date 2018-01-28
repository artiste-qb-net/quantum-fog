# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# import numpy as np
# import copy as cp

from potentials.Potential import *
# from DiscreteUniPot import *
import Utilities as ut
from MyExceptions import UnNormalizablePot


class DiscreteCondPot(Potential):
    """
    CondPot = Conditional Potential. A CondPot is a Potential that stores a
    node as a focus node and checks to make sure that that node remains the
    last node of its ord_nodes and pot_arr. CondPots can hold either
    conditional PDs like P(x| y_1, y_2, ..) or conditional PADs like A(x|
    y_1, y_2, ...), where x is the focus node. abbreviations in
    abbreviations.md. CondPots need not be normalized, but there is a method
    in the class to normalize them. Normalization depends on whether we are
    dealing with CNets or QNets, which corresponds to is_quantum=False or
    True respectively.

    Attributes
    ----------
    focus_node : Node
        last node in ord_nodes

    """

    def __init__(self, is_quantum, ord_nodes, pot_arr=None, bias=1):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        ord_nodes : list[BayesNode]
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
        return DiscreteCondPot(False, self.ord_nodes, pot_arr=arr2)

    def normalize_self(self, postpone=False, returns=False):
        """
        This normalizes pot_arr so that it becomes a conditional potential (
        a conditional PD for is_quantum=False or a conditional PAD for
        is_quantum=True) of last node given the others. Last node
        corresponds to last axis which corresponds to innermost bracket of
        pot_arr. If returns=True, returns the normalization constants in a
        dictionary with the input states as keys. If postpone=True, does not
        apply normalization constants to pot_arr.

        Parameters
        ----------
        postpone : bool
        returns : bool

        Returns
        -------
        None | float | dict[str, float]

        """

        # print("inside tr_normalize_self")
        # print("pot before", self)

        assert self.focus_node == self.ord_nodes[-1]

        if self.num_nodes == 1:
            if not self.is_quantum:
                d = self.pot_arr.sum()
            else:
                d = np.linalg.norm(self.pot_arr)
            if not postpone:
                if abs(d) > 1e-6:
                    self.pot_arr /= d
                else:
                    raise UnNormalizablePot(())
            if returns:
                return d
        else:
            totals = {}
            ind_gen = ut.cartesian_product(self.nd_sizes[:-1])
            axes = list(range(self.num_nodes - 1))
            for indices in ind_gen:
                slicex = self.slicex_from_axes(indices, axes)
                arr = self.pot_arr[slicex]
                if not self.is_quantum:
                    d = arr.sum()
                else:
                    d = np.linalg.norm(arr)
                if returns:
                    # this works but using indices as key is briefer
                    # name_tuple = str(tuple(self.ord_nodes[k].state_names[r]
                    #               for k, r in enumerate(indices)))
                    # name_tuple = ut.fix(name_tuple, "'", '')
                    # totals[name_tuple] = d

                    totals[indices] = d
                if not postpone:
                    if abs(d) > 1e-6:
                        self.pot_arr[slicex] /= d
                    else:
                        print('****************un-normalizable pot')
                        print('sick pot:')
                        print([node.name for node in self.ord_nodes])
                        print(self.pot_arr)
                        raise UnNormalizablePot(indices)
            if returns:
                return totals

        # print("pot after", self, "\n")

    def get_total_probs(self, brief=False):
        """
        This function is just a narrower version of tr_normalize_self(). When
        brief=False, it returns a dictionary giving total prob for each
        input state. When brief=True, it returns a dictionary with only the
        total probs that are less than 1.

        Parameters
        ----------
        brief : bool

        Returns
        -------
        float | dict[str, float]

        """
        d = self.normalize_self(postpone=True, returns=True)
        if brief:
            d = dict((name, prob) for name, prob in d.items() if prob < 1-1e-6)
        return d

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
    def main():
        x = [0, 1, 2, 3, 4]
        print(x[:-1])
    main()


