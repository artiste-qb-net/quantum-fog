# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np
import copy as cp

import Utilities as ut
# from nodes.BayesNode import *


class Potential:
    """
    Potentials are basically just functions of several nodes = random
    variables. A pot contains both a list of ordered nodes ('ord_nodes') and
    a numpy array ('pot_arr'). When we permute the ord_nodes of the pot,
    we also apply the corresponding numpy transposition to its pot_arr.

    A DiscreteUniPot is a DiscreteCondPot is a Potential.

    For is_quantum=False (resp., True), pot_arr is a numpy array of dtype
    float64 (resp., complex128)

    potential[index] yields pot_arr[index]

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
    nodes : set[BayesNode]
    num_nodes : int
    ord_nodes : list[BayesNode]
    pot_arr : numpy.ndarray

    """

    def __init__(self, is_quantum, ord_nodes, pot_arr=None, bias=1):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        ord_nodes : list[BayesNode]
        pot_arr : numpy.ndarray
            potential's array
        bias : complex

        Returns
        -------

        """
        assert(len(ord_nodes) > 0)
        self.is_quantum = is_quantum
        self.ord_nodes = ord_nodes
        self.nodes = set(ord_nodes)
        self.num_nodes = len(self.ord_nodes)
        self.nd_sizes = [node.size
                         for node in self.ord_nodes]
        if isinstance(pot_arr, np.ndarray):
            self.pot_arr = pot_arr
            assert (
                (np.shape(pot_arr) == np.array(self.nd_sizes)).all()
            ), "Node sizes do not match shape of pot_arr"
        else:
            self.set_pot_arr_to(bias)

    def set_pot_arr_to(self, val):
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
        self.pot_arr = np.zeros(
            self.nd_sizes, dtype=ty) + val

    def set_pot_arr_to_one(self):
        """
        Sets all entries of pot_arr to one.

        Returns
        -------
        None

        """
        self.set_pot_arr_to(1.)

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
                    slicex = self.get_slicex_nd([st], [node])
                    self.pot_arr[slicex] = 0.

    def get_new_marginal(self, fin_node_list):
        """
        Returns a new potential (marginal) obtained by summing self.pot_arr
        over states of all nodes except those in fin_node_list. fin=final.
        Note this does not modify pot_arr.

        Parameters
        ----------
        fin_node_list : list[BayesNode]

        Returns
        -------
        Potential

        """
        assert(self.nodes >= set(fin_node_list))
        fin_pot = Potential(self.is_quantum, fin_node_list, bias=0)

        fin_axes = list(range(fin_pot.num_nodes))
        ind_gen = ut.cartesian_product(fin_pot.nd_sizes)
        for fin_indices in ind_gen:
            slicex = self.get_slicex_nd(fin_indices, fin_node_list)
            fin_slicex = fin_pot.get_slicex_ax(fin_indices, fin_axes)
            fin_pot[fin_slicex] = self[slicex].sum()
        return fin_pot

    def get_axes(self, node_list):
        """
        Generates a list of axes that correspond to node_list.

        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        list[int]

        """

        assert(self.nodes >= set(node_list))
        return [self.ord_nodes.index(node) for node in node_list]

    def get_slicex_ax(self, indices, axes):
        """
        slicex is a portmanteau that stands for slice index. This function
        works hand in hand with __getitem__ and __setitem__ which override
        getting and setting via [ ]. It takes in a list of indices and a
        list of axes both of the same length and order. It generates a
        slicex by padding the list 'indices' with extra slice(None) indices.

        Parameters
        ----------
        indices : list[int]
        axes : list[int]

        Returns
        -------
        tuple

        """
        assert(len(indices) == len(axes))
        assert(len(axes) <= self.num_nodes)

        padded_indices = [slice(None)]*self.num_nodes

        for k in range(len(axes)):
            padded_indices[axes[k]] = indices[k]

        # Will make slicex a tuple to avert advanced indexing
        # We want just a basic slice
        return tuple(padded_indices)

    def get_slicex_nd(self, indices, node_list):
        """
        The _nd version of this function has node_list as argument, the _ax
        version has axes instead, but they return the same thing.

        Parameters
        ----------
        indices : list[int]
        node_list : list[BayesNode]

        Returns
        -------
        tuple

        """

        return self.get_slicex_ax(
                indices, self.get_axes(node_list))

    def set_to_transpose(self, node_list):
        """
        node_list should be a permutation of self.ord_nodes. Like in numpy,
        we will words "permutation" and "transpose" interchangeably. This
        function replaces ord_nodes by node_list and applies corresponding
        numpy transposition to pot_arr.


        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        None

        """

        assert(set(node_list) == self.nodes)
        axes = self.get_axes(node_list)

        # this didn't work
        # self.pt_arr.transpose(axes)

        self.pot_arr = np.transpose(self.pot_arr, axes)
        self.ord_nodes = node_list

    def conj(self):
        """
        Returns new Potential whose pot_arr is the complex conjugate of
        self.pot_arr

        Returns
        -------
        Potential

        """
        return Potential(self.ord_nodes, np.conjugate(self.pot_arr))

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
        Used instead of __truediv__ for pots. Needed when dividing pointwise
        two pot_arr's yields either inf or nan.

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

            new = xx / yy
            # replace inf by zero
            new[new == np.inf] = 0
            # replace nan by zero
            new = np.nan_to_num(new)

            # print("xx/yy", new)

        return new

    @staticmethod
    def __safe_itruediv(xx, yy):
        """
        Used instead of __itruediv__ for pots. Needed when dividing
        pointwise two pot_arr's yields either inf or nan.

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
            # having trouble with xx /= yy
            # It appears numpy x/=y doesn't modify original x

            # print("\nin place divide")
            # print("xx", xx)
            # print("yy", yy)

            xx = xx/yy
            # replace inf by zero
            xx[xx == np.inf] = 0
            # replace nan by zero
            xx = np.nan_to_num(xx)

            # print("xx/yy", xx)
        return xx

    def __binary_op(self, right, magic, imagic):
        """
        This private method will be used to override binary operators
        __add__, __sub__, __mult__ and __truediv__ for pots. A re-alignment
        of the pot axes is required before applying the binary operator to
        two numpy pot_arr's.

        Parameters
        ----------
        right : Potential
        magic : wrapper_descriptor
            This is going to be either
            np.ndarray.[__add__, __sub__, __mul__],
            Potential.__safe_truediv
        imagic : wrapper_descriptor
            This is going to be either
            np.ndarray.[__iadd__, __isub__, __imul__],
            Potential.__safe_itruediv]

        Returns
        -------
        Potential

        """
        if isinstance(right, (int, float, complex)):
            new = cp.deepcopy(self)
            imagic(new.pot_arr, right)
        elif self.nodes == right.nodes:
            new = cp.deepcopy(self)
            new.set_to_transpose(right.ord_nodes)
            imagic(new.pot_arr, right.pot_arr)
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

            self.set_to_transpose(
                self_only_nlist + overlap_nlist)
            right.set_to_transpose(
                overlap_nlist + right_only_nlist)

            self_slicex = [slice(None)]*(la+lb)
            self_slicex += [np.newaxis]*lc
            self_slicex = tuple(self_slicex)

            right_slicex = [np.newaxis]*la
            right_slicex += [slice(None)]*(lb+lc)
            right_slicex = tuple(right_slicex)

            # numpy array magic
            new.pot_arr = magic(
                self[self_slicex], right[right_slicex])
        return new

    def __inplace_binary_op(self, right, imagic):
        """
        This private method will be used to override the in place binary
        operators __iadd__, __isub__, __imult__ and __itruediv__ for pots. A
        re-alignment of the pot axes is required before applying the binary
        operator to two numpy pot_arr's.

        Parameters
        ----------
        right : Potential
        imagic : wrapper_descriptor
            This is going to be either
            np.ndarray.[__iadd__, __isub__, __imult__],
            Potential.__safe_itruediv

        Returns
        -------
        Potential

        """

        if isinstance(right, (int, float, complex)):
            imagic(self.pot_arr, right)
        elif self.nodes == right.nodes:
            self.set_to_transpose(right.ord_nodes)
            imagic(self.pot_arr, right.pot_arr)
        else:
            assert(self.nodes >= right.nodes),\
                "can't add or mult in place unless self node set " \
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
            imagic(self[self_slicex], right[right_slicex])
        return self

    def __add__(self, right):
        """
        Pointwise addition (+) of elements in self and right. self and right
        can be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__binary_op(
            right, np.ndarray.__add__, np.ndarray.__iadd__)

    def __iadd__(self, right):
        """
        Pointwise in place addition (+=) of elements in self and right. self
        node set must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__inplace_binary_op(right, np.ndarray.__iadd__)

    def __sub__(self, right):
        """
        Pointwise subtraction (-) of elements in self and right. self and
        right can be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__binary_op(
            right, np.ndarray.__sub__, np.ndarray.__isub__)

    def __isub__(self, right):
        """
        Pointwise in place subtraction (-=) of elements in self and right.
        self node set must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__inplace_binary_op(right, np.ndarray.__isub__)

    def __mul__(self, right):
        """
        Pointwise multiplication (*) of elements in self and right. self and
        right can be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__binary_op(
            right, np.ndarray.__mul__, np.ndarray.__imul__)

    def __imul__(self, right):
        """
        Pointwise in place multiplication (*=) of elements in self and
        right. self node set must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__inplace_binary_op(right, np.ndarray.__imul__)

    def __truediv__(self, right):
        """
        Pointwise division (/) of elements in self and right. self and
        right can be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__binary_op(
            right, Potential.__safe_truediv, Potential.__safe_itruediv)

    def __itruediv__(self, right):
        """
        Pointwise in place division (/=) of elements in self and
        right. self node set must contain right node set.

        Parameters
        ----------
        right : Potential

        Returns
        -------
        Potential

        """

        return self.__inplace_binary_op(right, Potential.__safe_itruediv)

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
        Potential

        Returns
        -------
        str

        """
        return str([node.name for node in self.ord_nodes]) \
            + "\n" + str(self.pot_arr)

from nodes.BayesNode import *
if __name__ == "__main__":
    with np.errstate(all='ignore'):
        x = np.array([2, 0])/np.array([1, 0])
        x[x == np.inf] = 0
        x = np.nan_to_num(x)
    print("[2, 0]/[1, 0]=", x)

    # define some nodes
    a_node = BayesNode(0, name="A")
    b_node = BayesNode(1, name="B")
    c_node = BayesNode(2, name="C")
    d_node = BayesNode(3, name="D")
    e_node = BayesNode(4, name="E")
    f_node = BayesNode(5, name="F")
    g_node = BayesNode(6, name="G")

    print("\ndefine some pots")

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

    print("\ntry distance")
    print("distance(pot_abc, pot_bc)=",
        Potential.distance(pot_abc, pot_bc))
    print("pot_abc == pot_bc?", pot_abc == pot_bc)

    new_abc = cp.deepcopy(pot_abc)
    print("distance(pot_abc, new_abc)=",
                    Potential.distance(new_abc, pot_abc))
    print("pot_abc == new_abc?", pot_abc == new_abc)

    print("\ntry add, sub, mult")
    print("pot_ab + 5:", pot_ab + 5)
    print("pot_ab * 5:", pot_ab * 5)
    print("pot_ab + pot_bc:", pot_ab + pot_bc)
    print("pot_ab - pot_bc:", pot_ab - pot_bc)
    print("pot_ab * pot_bc:", pot_ab * pot_bc)

    print("\ntry set_to_transpose")
    new_abc = cp.deepcopy(pot_abc)
    new_abc.set_to_transpose([b_node, a_node, c_node])
    print("pot_abc:", pot_abc)
    print("new_abc:", new_abc)
    assert pot_abc == new_abc
    assert pot_abc != (new_abc + 5)

    print("\ntry iadd, isub, and imul")
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

    print("\ntry truediv")
    print("pot_ab/pot_ab2 =\n", pot_ab/pot_ab2)

    print("\ntry itruediv")
    new_ab = cp.deepcopy(pot_ab)
    new_ab.set_to_transpose([b_node, a_node])
    new_ab /= pot_ab2
    print("pot_ab/pot_ab2", new_ab)
    assert new_ab == pot_ab/pot_ab2

    print("\ntry marginalizing")
    print("new_abc=", new_abc)
    new_ac = new_abc.get_new_marginal([a_node, c_node])
    print("sum_b new_abc=\n", new_ac)

    print("pot_ecg:", pot_ecg)
    pot_ge = pot_ecg.get_new_marginal([g_node, e_node])
    print("sum_c  pot_ecg=", pot_ge)


