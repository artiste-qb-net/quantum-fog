# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np
import copy as cp

import Utilities as ut
from nodes.BayesNode import *


class Potential:
    """
    Potentials are basically just functions of several nodes = random
    variables. A pot contains both a list of ordered nodes ('ord_nodes') and
    a numpy array ('pot_arr'). When we permute the ord_nodes of the pot,
    we also apply the corresponding numpy transposition to its pot_arr. We
    define as equal all the pots reachable from any starting pot by this
    symmetry operation.

    For is_quantum=False (resp., True), pot_arr is a numpy array of dtype
    float64 (resp., complex128)

    nd_sizes is a list of the sizes of the nodes in ord_nodes. pot_arr.shape
    = nd_sizes

    potential[index] yields pot_arr[index]

    A DiscreteUniPot is a DiscreteCondPot is a Potential.

    The Potential class is where most of the magical numpy functionality of
    QuantumFog resides.

    IMP. when labeling sets of nodes, we will call them just 'nodes' or
    'subnodes' if they are a set and order doesn't matter. We will call them
    'ord_nodes' or 'node_list' if they are in a list and order does matter.
    This is especially important in the Potential class were we will use
    both a 'nodes' and 'ord_nodes'.


    Attributes
    ----------
    is_quantum : bool
    nd_sizes : list[int]
        sizes of nodes in ord_nodes
    nodes : set[BayesNode]
        set(ord_nodes)
    num_nodes:
        len(nodes)
    ord_nodes : list[BayesNode]
        nodes in this list are in 1-1 correspondence with axes of pot_arr
    pot_arr : numpy.ndarray
        potential's array. shape=nd_sizes

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
            if pot_arr is None as input, all its entries are set to bias

        Returns
        -------

        """
        assert len(ord_nodes) > 0
        self.is_quantum = is_quantum
        self.ord_nodes = ord_nodes
        self.nodes = set(ord_nodes)
        self.num_nodes = len(self.ord_nodes)
        self.nd_sizes = [node.size for node in self.ord_nodes]
        if isinstance(pot_arr, np.ndarray):
            self.pot_arr = pot_arr
            test = (np.shape(pot_arr) == np.array(self.nd_sizes)).all()
            assert test, "Node sizes do not match shape of pot_arr"
        else:
            self.set_all_entries_to(bias)

    def is_joint_prob_dist(self):
        """
        Returns True if pot is a classical joint prob distribution of all
        its nodes.

        Returns
        -------
        bool

        """
        classical = ~self.is_quantum
        nneg = (self.pot_arr >= 0).all()
        sum_one = (abs(self.get_new_marginal([])-1) < 1e-6)
        return classical & nneg & sum_one

    def set_all_entries_to(self, val):
        """
        Sets all entries of pot_arr to val.

        Parameters
        ----------
        val : float | complex

        Returns
        -------
        None

        """
        if not self.is_quantum:
            ty = np.float64
        else:
            ty = np.complex128
        self.pot_arr = np.zeros(self.nd_sizes, dtype=ty) + val

    def set_to_random(self, max_int=None):
        """
        Sets all entries of pot_arr to random value.

        Parameters
        ----------
        max_int : int|None
            if max_int=None, then use random floats in [0, 1) for real and
            imaginary part of entries. If max_int is an int, then use random
            ints in [0, max_int) for real and imaginary part of entries.

        Returns
        -------
        None

        """
        self.set_all_entries_to(0)
        if max_int:
            # adding in place doesn't change dtype of self.pot_arr
            self.pot_arr += np.random.randint(0, max_int, size=self.nd_sizes)
            if self.is_quantum:
                self.pot_arr += \
                    1j*np.random.randint(0, max_int, size=self.nd_sizes)
        else:
            self.pot_arr += np.random.rand(*self.nd_sizes)
            if self.is_quantum:
                self.pot_arr += 1j*np.random.rand(*self.nd_sizes)

    def mask_self(self):
        """
        Multiply pot_arr entries times zero for those entries that are
        forbidden by the active states constraint of the ord_nodes.

        Returns
        -------
        None

        """
        for node in self.ord_nodes:
            for st in range(node.size):
                if st not in node.active_states:
                    slicex = self.slicex_from_nds([st], [node])
                    self.pot_arr[slicex] = 0.

    def get_new_marginal(self, fin_node_list):
        """
        Returns a new potential (marginal) obtained by summing self.pot_arr
        over states of all nodes except those in fin_node_list. fin=final.
        Note this does not modify pot_arr. This function works even if
        fin_node_list=[].

        Parameters
        ----------
        fin_node_list : list[BayesNode]

        Returns
        -------
        float | Potential

        """
        assert self.nodes >= set(fin_node_list)
        if not fin_node_list:
            return self.pot_arr.sum()

        fin_pot = Potential(self.is_quantum, fin_node_list, bias=0)

        fin_axes = list(range(fin_pot.num_nodes))
        ind_gen = ut.cartesian_product(fin_pot.nd_sizes)
        for fin_indices in ind_gen:
            slicex = self.slicex_from_nds(fin_indices, fin_node_list)
            fin_slicex = fin_pot.slicex_from_axes(fin_indices, fin_axes)
            fin_pot[fin_slicex] = self[slicex].sum()
        return fin_pot

    def get_axes(self, node_list):
        """
        Returns a list of the positions in ord_nodes of the nodes in
        node_list.

        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        list[int]

        """

        assert self.nodes >= set(node_list)
        return [self.ord_nodes.index(node) for node in node_list]

    def slicex_from_axes(self, indices, axes):
        """
        slicex is a portmanteau that stands for slice index. This function
        works hand in hand with __getitem__ and __setitem__ which override
        getting and setting via [ ]. It takes in a list of indices and a
        list of axes both of the same length and in 1-1 correspondence. It
        returns a slicex generated by padding the list 'indices' with extra
        slice(None) indices.

        Parameters
        ----------
        indices : list[int]
        axes : list[int]

        Returns
        -------
        tuple

        """
        assert len(indices) == len(axes)
        assert len(axes) <= self.num_nodes

        padded_indices = [slice(None)]*self.num_nodes

        for k in range(len(axes)):
            padded_indices[axes[k]] = indices[k]

        # Will make slicex a tuple to avert advanced indexing
        # We want just a basic slice
        return tuple(padded_indices)

    def slicex_from_nds(self, indices, node_list):
        """
        The _nds version of this function has node_list as argument,
        the _axes version has axes instead, but they return the same thing.

        Parameters
        ----------
        indices : list[int]
        node_list : list[BayesNode]

        Returns
        -------
        tuple

        """

        return self.slicex_from_axes(indices, self.get_axes(node_list))

    def set_to_transpose(self, node_list):
        """
        node_list should be a permutation of self.ord_nodes. Like numpy
        does, we will use the words "permutation" and "transpose"
        interchangeably. This function replaces ord_nodes by node_list and
        applies corresponding numpy transposition to pot_arr.


        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        None

        """

        assert set(node_list) == self.nodes
        axes = self.get_axes(node_list)

        # this didn't work
        # self.pt_arr.transpose(axes)

        self.pot_arr = np.transpose(self.pot_arr, axes)
        self.ord_nodes = node_list
        self.nd_sizes = [node.size for node in self.ord_nodes]

    def cc(self):
        """
        Returns new Potential whose pot_arr is the complex conjugate of
        self.pot_arr

        Returns
        -------
        Potential

        """
        return Potential(self.is_quantum,
            self.ord_nodes, np.conjugate(self.pot_arr))

    @staticmethod
    def cc_of(pot):
        """
        Returns new Potential whose pot_arr is the complex conjugate of
        pot.pot_arr

        Parameters
        ----------
        pot : Potential

        Returns
        -------
        Potential

        """
        return Potential(pot.is_quantum,
            pot.ord_nodes, np.conjugate(pot.pot_arr))

    @staticmethod
    def mag(pot):
        """
        mag= magnitude. Returns the norm of self.pot_arr, where by norm we
        mean the usual norm used in Quantum Mechanics, called either
        Frobenius or 2-norm, \sqrt(\sum_index  abs(pot_arr[index])^2)

        Parameters
        ----------
        pot : Potential

        Returns
        -------
        float

        """
        return np.linalg.norm(pot.pot_arr)

    @staticmethod
    def distance(pot1, pot2):
        """
        Returns the mag of (pot1 - pot2).

        Parameters
        ----------
        pot1 : Potential
        pot2 : Potential

        Returns
        -------
        float

        """
        return Potential.mag(pot1 - pot2)

    def __getitem__(self, slicex):
        """
        Overrides the [] getter. This allows us to use a slicex as
        an index of a Potential object.

        Parameters
        ----------
        slicex : tuple

        Returns
        -------
        float | complex


        """

        return self.pot_arr[slicex]

    def __setitem__(self, slicex, value):
        """
        Overrides the [] setter. This allows us to use a slicex as
        an index of a Potential object.


        Parameters
        ----------
        slicex : tuple
        value : float | complex

        Returns
        -------
        None

        """
        self.pot_arr[slicex] = value

    def __eq__(self, other):
        """
        Overrides ==. This defines what it means for two pots to be equal.
        The 2 pots don't have to have the same pot_arr. As long as one can
        be converted to the other using set_to_transpose(), they are
        considered equal.

        Parameters
        ----------
        other : Potential

        Returns
        -------
        bool

        """
        return Potential.distance(self, other) < 1e-6

    def __ne__(self, other):
        """
        Overrides !=

        Parameters
        ----------
        other : Potential

        Returns
        -------
        bool

        """
        return not self.__eq__(other)

    @staticmethod
    def __safe_truediv(xx, yy):
        """
        Used instead of __truediv__ for pots. Needed when dividing entrywise
        two pot_arr's yields either -inf, inf or nan.

        Parameters
        ----------
        xx : numpy.ndarray
        yy : numpy.ndarray

        Returns
        -------
        numpy.ndarray

        """

        # clean denominator up to avoid round-off error
        # print(yy)
        # yy[abs(yy)<1e-6] = 0

        with np.errstate(all='ignore'):

            # print("\ndivide")
            # print("xx", xx)
            # print("yy", yy)

            new = xx/yy
            # np.isfinite(new) returns array of same size as new
            # with True iff component not -inf, inf or nan
            # and ~ negates each component
            new[~ np.isfinite(new)] = 0
        return new

    @staticmethod
    def __safe_itruediv(xx, yy):
        """
        Used instead of __itruediv__ (in place true division) for pots. 
        Needed when dividing entrywise two pot_arr's yields either -inf, 
        inf or nan. 

        Parameters
        ----------
        xx : numpy.ndarray
        yy : numpy.ndarray

        Returns
        -------
        numpy.ndarray


        """

        # clean denominator up to avoid round-off error
        # yy[abs(yy)<1e-6] = 0

        with np.errstate(all='ignore'):

            # print("\nin place divide")
            # print("xx", xx)
            # print("yy", yy)

            xx /= yy
            xx[~ np.isfinite(xx)] = 0
        return xx

    def pot_op(self, right, arr_op):
        """
        This private method will be used to override binary operators
        __add__, __sub__, __mult__ and __truediv__ for pots. A re-alignment
        of the pot axes is required before applying the binary operator to
        two numpy pot_arr's.

        Parameters
        ----------
        right : Potential
        arr_op : wrapper_descriptor
            This is going to be either
            np.ndarray.[__add__, __sub__, __mul__],
            Potential.__safe_truediv

        Returns
        -------
        Potential

        """
        if isinstance(right, (int, float, complex)):
            new_pot_arr = arr_op(self.pot_arr, right)
            new = Potential(self.is_quantum, self.ord_nodes,
                            pot_arr=new_pot_arr)
        else:
            # nlist = node list

            self_only_nlist = list(self.nodes - right.nodes)
            overlap_nlist = list(self.nodes & right.nodes)
            right_only_nlist = list(right.nodes - self.nodes)

            la = len(self_only_nlist)
            lb = len(overlap_nlist)
            lc = len(right_only_nlist)

            new = Potential(self.is_quantum,
                self_only_nlist + overlap_nlist + right_only_nlist)

            self_copy = cp.deepcopy(self)
            right_copy = cp.deepcopy(right)
            self_copy.set_to_transpose(
                self_only_nlist + overlap_nlist)
            right_copy.set_to_transpose(
                overlap_nlist + right_only_nlist)

            self_slicex = [slice(None)]*(la+lb)
            self_slicex += [np.newaxis]*lc
            self_slicex = tuple(self_slicex)

            right_slicex = [np.newaxis]*la
            right_slicex += [slice(None)]*(lb+lc)
            right_slicex = tuple(right_slicex)

            # numpy array magic
            new.pot_arr = arr_op(
                self_copy[self_slicex], right_copy[right_slicex])
        return new

    def pot_iop(self, right, arr_iop):
        """
        This private method will be used to override the in place binary
        operators __iadd__, __isub__, __imult__ and __itruediv__ for pots. A
        re-alignment of the pot axes is required before applying the binary
        operator to two numpy pot_arr's.

        Parameters
        ----------
        right : Potential
        arr_iop : wrapper_descriptor
            This is going to be either
            np.ndarray.[__iadd__, __isub__, __imult__],
            Potential.__safe_itruediv

        Returns
        -------
        Potential

        """

        if isinstance(right, (int, float, complex)):
            arr_iop(self.pot_arr, right)
        else:
            assert self.nodes >= right.nodes,\
                "can't add or mult *in place* unless self node set " \
                "contains right node set"
            # nlist = node list
            self_only_nlist = list(self.nodes - right.nodes)

            la = len(self_only_nlist)
            lb = len(right.ord_nodes)

            self.set_to_transpose(
                self_only_nlist + right.ord_nodes)

            self_slicex = [slice(None)]*(la+lb)
            self_slicex = tuple(self_slicex)

            right_slicex = [np.newaxis]*la
            right_slicex += [slice(None)]*lb
            right_slicex = tuple(right_slicex)

            # numpy array magic
            arr_iop(self[self_slicex], right[right_slicex])
        return self

    def __add__(self, right):
        """
        Entrywise addition (+) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_op(right, np.ndarray.__add__)

    def __iadd__(self, right):
        """
        Entrywise in place addition (+=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_iop(right, np.ndarray.__iadd__)

    def __sub__(self, right):
        """
        Entrywise subtraction (-) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_op(right, np.ndarray.__sub__)

    def __isub__(self, right):
        """
        Entrywise in place subtraction (-=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_iop(right, np.ndarray.__isub__)

    def __mul__(self, right):
        """
        Entrywise multiplication (*) of self and right. self and right can
        be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_op(right, np.ndarray.__mul__)

    def __imul__(self, right):
        """
        Entrywise in place multiplication (*=) of self and right. self node
        set must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_iop(right, np.ndarray.__imul__)

    def __truediv__(self, right):
        """
        Entrywise division (/) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_op(right, Potential.__safe_truediv)

    def __itruediv__(self, right):
        """
        Entrywise in place division (/=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.pot_iop(right, Potential.__safe_itruediv)

    def __deepcopy__(self, memo):
        """
        We want deepcopy to produce a copy of pot_arr but not of the nodes
        in self.nodes so need to override the usual deepcopy.

        Parameters
        ----------
        memo :

        Returns
        -------
        Potential

        """
        copy_pot_arr = cp.deepcopy(self.pot_arr)
        return Potential(self.is_quantum,
                    ord_nodes=self.ord_nodes, pot_arr=copy_pot_arr)

    def __str__(self):
        """
        What string is outputted by print(pot) where pot is an object of
        Potential? The Shadow and __str__ know.

        Returns
        -------
        str

        """
        return str([node.name for node in self.ord_nodes]) \
            + "\n" + str(self.pot_arr)

if __name__ == "__main__":
    def main():
        with np.errstate(all='ignore'):
            x = np.array([2, 0+0j])/np.array([1, 0])
            x[~ np.isfinite(x)] = 0
        print("[2, 0+0j]/[1, 0]=", x)

        with np.errstate(all='ignore'):
            x = np.array([2, 5])/np.array([1, 0])
            x[~ np.isfinite(x)] = 0
        print("[2, 5]/[1, 0]=", x)

        # define some nodes
        a_node = BayesNode(0, name="A")
        b_node = BayesNode(1, name="B")
        c_node = BayesNode(2, name="C")
        d_node = BayesNode(3, name="D")
        e_node = BayesNode(4, name="E")
        f_node = BayesNode(5, name="F")
        g_node = BayesNode(6, name="G")

        print("\n-----------------define some pots")

        # Make sure if entries of numpy array are integers, to specify
        # dtype=float64 or numpy will cast them as integers and this will lead
        # to type casting errors

        ar_ab = np.arange(4, dtype=np.float64).reshape(2, 2)
        pot_ab = Potential(False, [a_node, b_node], ar_ab)
        print("pot_ab:", pot_ab)

        ar_ab2 = np.arange(0, 40, 10, dtype=np.float64).reshape(2, 2)
        pot_ab2 = Potential(False, [a_node, b_node], ar_ab2)
        print("pot_ab2:", pot_ab2)

        ar_bc = np.arange(0, 40, 10, dtype=np.float64).reshape(2, 2)
        pot_bc = Potential(False, [b_node, c_node], ar_bc)
        print("pot_bc:", pot_bc)

        ar_abc = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
        pot_abc = Potential(False, [a_node, b_node, c_node], ar_abc)
        print("pot_abc:", pot_abc)

        ar_bcd = np.arange(0, 80, 10, dtype=np.float64).reshape((2, 2, 2))
        pot_bcd = Potential(False, [b_node, c_node, d_node], ar_bcd)
        print("pot_bcd:", pot_bcd)

        # no need to specify dtype here because using decimal points
        ar_ecg = np.array(
            [[[0., 0.60000002],
            [0., 0.30000001]],
            [[0., 0.40000001],
            [0., 0.69999999]]])
        pot_ecg = Potential(False, [e_node, c_node, g_node], ar_ecg)
        print("pot_ecg:", pot_ecg)

        print("\n-----------------try transpose, distance and ==")

        new_abc = cp.deepcopy(pot_abc)
        new_abc.set_to_transpose([b_node, a_node, c_node])
        print("pot_abc:", pot_abc)
        print("new_abc:", new_abc)
        assert pot_abc == new_abc
        assert pot_abc != (new_abc + 5)

        print("distance(pot_abc, new_abc)=",
                        Potential.distance(new_abc, pot_abc))
        print("pot_abc == new_abc?", pot_abc == new_abc)

        print("distance(pot_abc, pot_bc)=",
            Potential.distance(pot_abc, pot_bc))
        print("pot_abc == pot_bc?", pot_abc == pot_bc)

        print("\n-----------------try add, sub, mult")

        print('pot_ab:', pot_ab)
        print('pot_ab2:', pot_ab2)
        print("pot_ab + 5:", pot_ab + 5)
        print("pot_ab - 5:", pot_ab - 5)
        print("pot_ab * 5:", pot_ab * 5)
        print("pot_ab + pot_ab2:", pot_ab + pot_ab2)
        print("pot_ab - pot_ab2:", pot_ab - pot_ab2)
        print("pot_ab * pot_ab2:", pot_ab * pot_ab2)
        print("pot_ab + pot_bc:", pot_ab + pot_bc)
        print("pot_ab - pot_bc:", pot_ab - pot_bc)
        print("pot_ab * pot_bc:", pot_ab * pot_bc)

        print("\n-----------------try iadd, isub, and imul")

        new_abc = cp.deepcopy(pot_abc)
        new_abc.set_to_transpose([b_node, a_node, c_node])
        new_abc += 5
        assert new_abc == pot_abc + 5

        new_abc = cp.deepcopy(pot_abc)
        new_abc.set_to_transpose([b_node, a_node, c_node])
        new_abc += pot_bc
        assert new_abc == pot_abc + pot_bc

        new_abc = cp.deepcopy(pot_abc)
        new_abc.set_to_transpose([b_node, a_node, c_node])
        new_abc -= pot_bc
        assert new_abc == pot_abc - pot_bc

        new_abc = cp.deepcopy(pot_abc)
        new_abc.set_to_transpose([b_node, a_node, c_node])
        new_abc *= pot_bc
        assert new_abc == pot_abc*pot_bc

        print("\n-----------------try truediv")

        print("pot_ab/pot_ab2 =", pot_ab/pot_ab2)

        print("\n-----------------try itruediv")
        new_ab = cp.deepcopy(pot_ab)
        new_ab.set_to_transpose([b_node, a_node])
        new_ab /= pot_ab2
        print("pot_ab/pot_ab2=", new_ab)
        assert new_ab == pot_ab/pot_ab2

        print("\n-----------------try marginalizing")

        new_abc = cp.deepcopy(pot_abc)
        print("new_abc=", new_abc)
        new_ac = new_abc.get_new_marginal([a_node, c_node])
        print("sum_b new_abc=", new_ac)

        print("\nnew_abc=", new_abc)
        new_ab = new_abc.get_new_marginal([a_node, b_node])
        print("sum_c new_abc=", new_ab)

        print("\npot_ecg:", pot_ecg)
        pot_ge = pot_ecg.get_new_marginal([g_node, e_node])
        print("sum_c  pot_ecg=", pot_ge)
    main()
