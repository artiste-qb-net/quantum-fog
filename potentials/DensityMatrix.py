from potentials.Potential import *
import numpy as np
import pandas as pd
import copy as cp
import Utilities as ut


class DensityMatrix:
    """
    In quantum mechanics, a density matrix is: (def) a Hermitian matrix such
    that all of its eigenvalues are non-negative and sum to one. An object
    of class DensityMatrix contains an array dmat_arr of complex numbers
    which is not shaped as a square matrix but can be reshaped thus. That
    square matrix need not satisfy (def). It need not even be Hermitian.
    It's up to the user to load into dmat_arr a matrix that satisfies (def).
    You can check if it satisfies (def) with the function is_legal_dmat().

    Think of a Potential object pot with is_quantum=True and ord_nodes = [a,
    b, c] as a pure state |psi_abc>. If dmat_abc represents
    |psi_abc><psi_abc|, then one can trace over one or more of the nodes a,
    b, c to obtain a new Density Matrix describing a mixture state. For
    example, tr_c dmat_abc = dmat_ab. One can also trace over the nodes of a
    mixture to obtain another mixture. For example, tr_b dmat_ab = dmat_a.

    A DensityMatrix object dmat contains 'ord_nodes', a list of ordered
    nodes and 'nd_sizes', a list of the sizes of the nodes in ord_nodes. The
    shape of dmat_arr is nd_sizes*2, i.e., the list nd_sizes repeated twice.
    Hence, dmat_arr contains two indices for each node in ord_nodes, and the
    axes for those two indices are len(ord_nodes) apart. When we permute the
    ord_nodes of the dmat, we also apply the corresponding numpy
    transposition to first half of the axes of dmat_arr, and to the second
    half of the axes of dmat_arr. We define as equal all the dmats reachable
    from any starting dmat by this symmetry operation.

    DensityMatrix[index] yields dmat_arr[index]

    IMP. when labeling sets of nodes, we will call them just 'nodes' or
    'subnodes' if they are a set and order doesn't matter. We will call them
    'ord_nodes' or 'node_list' if they are in a list and order does matter.
    This is especially important in the DensityMatrix class were we will use
    both a 'nodes' and 'ord_nodes'.

    Attributes
    ----------
    dmat_arr : numpy.ndarray
        density matrix's array, shape=nd_sizes*2, dtype=complex
    nd_sizes : list[int]
        sizes of nodes in ord_nodes
    nodes : set[BayesNode]
        set(ord_nodes)
    num_nodes: int
        len(nodes)
    ord_nodes : list[BayesNode]
        nodes in this list are in 1-1 correspondence with first half of axes
        of dmat_arr. They are also in 1-1 correspondence with second half of
        axes of dmat_arr.

    """

    def __init__(self, ord_nodes, dmat_arr=None, diag_val=1):
        """
        Constructor

        Parameters
        ----------
        ord_nodes : list[BayesNode]
        dmat_arr : numpy.ndarray
        diag_val : complex
            if input dmat_arr=None, sets dmat_arr to diagonal matrix with
            diagonal entries equal to diag_val

        Returns
        -------

        """
        self.ord_nodes = ord_nodes
        self.nodes = set(ord_nodes)
        self.num_nodes = len(self.ord_nodes)
        self.nd_sizes = [node.size for node in self.ord_nodes]
        if isinstance(dmat_arr, np.ndarray):
            self.dmat_arr = dmat_arr
            test = (np.shape(dmat_arr) == np.array(self.nd_sizes*2)).all()
            assert test, "Node sizes inconsistent with shape of dmat_arr"
        else:
            self.set_to_diag_mat(diag_val)

    def is_legal_dmat(self):
        """
        Returns True iff self is a legal density matrix; i.e. if dmat_arr,
        when reshaped to a square array, is a Hermitian matrix such that all
        its eigenvalues are non-negative and sum to one.

        Returns
        -------
        bool

        """
        arr = self.get_sq_array()
        is_herm = ut.is_herm(arr)
        evas = np.linalg.eigvalsh(arr)
        nneg = (evas >= 0).all()
        sum_one = (abs(evas.sum()-1) < 1e-6)
        # print(evas)
        # print(is_herm, nneg, sum_one)
        return is_herm & nneg & sum_one

    def set_all_entries_to(self, val):
        """
        Sets all entries of dmat_arr to val.

        Parameters
        ----------
        val : complex

        Returns
        -------
        None

        """
        self.dmat_arr = np.zeros(self.nd_sizes*2, dtype=complex) + val

    def set_to_diag_mat(self, diag_val):
        """
        Sets dmat_arr to diagonal matrix with diagonal entries equal to
        diag_val.

        Parameters
        ----------
        diag_val : complex

        Returns
        -------
        None

        """
        self.dmat_arr = np.zeros(self.nd_sizes*2, dtype=complex)

        ind_gen = ut.cartesian_product(self.nd_sizes)
        for index in ind_gen:
            self.dmat_arr[tuple(index*2)] = diag_val

    def set_to_random(self, max_int=None, normalize=False):
        """
        Sets dmat_arr to random Hermitian matrix with non-negative
        eigenvalues. If you want the eigenvalues to sum to one,
        use normalize=True.

        Parameters
        ----------
        max_int : int | None
            if max_int=None, then use random floats in [0, 1) for real and
            imaginary part of entries. If max_int is an int, then use random
            ints in [0, max_int) for real and imaginary part of entries.

        normalize : bool

        Returns
        -------
        None

        """
        dims = self.nd_sizes*2
        nrows = np.prod(np.array(self.nd_sizes))
        if max_int:
            arr = np.random.randint(0, max_int, size=(nrows, nrows)) + \
                  1j*np.random.randint(0, max_int, size=(nrows, nrows))
        else:
            arr = np.random.rand(nrows, nrows) + \
                  1j*np.random.rand(nrows, nrows)
        arr += arr.conj().T
        # matrix multiply arr times itself to make eigenvalues non-negative
        arr = np.dot(arr, arr)
        if normalize:
            arr /= np.trace(arr)
        self.dmat_arr = np.reshape(arr, dims)

    @staticmethod
    def new_from_tr_of_pure_st(pot, fin_nd_list):
        """
        Returns new DensityMatrix dmat obtained by tracing an input pure
        state pot over all nodes except those in the list fin_nd_list. For
        example, suppose pot is a Potential with is_quantum=True and
        ord_nodes = [a, b, c]. pot represents a pure state |psi_abc>. If
        dmat_abc represents |psi_abc><psi_abc| and fin_nd_list = [a, b],
        then this function returns tr_c dmat_abc = dmat_ab. This function
        works even if fin_node_list=[].

        Parameters
        ----------
        pot : Potential
        fin_nd_list : list[BayesNode]

        Returns
        -------
        DensityMatrix

        """
        assert pot.nodes >= set(fin_nd_list)
        fin_dmat = DensityMatrix(fin_nd_list)

        fin_nd_axes = [pot.ord_nodes.index(nd) for nd in fin_nd_list]
        ind_gen1 = ut.cartesian_product(fin_dmat.nd_sizes)
        for k1, indices1 in enumerate(ind_gen1):
            slicex1 = pot.slicex_from_axes(indices1, fin_nd_axes)
            ind_gen2 = ut.cartesian_product(fin_dmat.nd_sizes)
            for k2, indices2 in enumerate(ind_gen2):
                indices12 = tuple(list(indices1) + list(indices2))
                if k2 >= k1:
                    slicex2 = pot.slicex_from_axes(indices2, fin_nd_axes)
                    fin_dmat.dmat_arr[indices12] = (
                        pot.pot_arr[slicex1]*np.conj(pot.pot_arr[slicex2])
                        ).sum()
                else:
                    indices21 = tuple(list(indices2) + list(indices1))
                    fin_dmat.dmat_arr[indices12] = \
                        np.conj(fin_dmat.dmat_arr[indices21])
        return fin_dmat

    @staticmethod
    def new_from_tr_of_mixed_st(dmat, fin_nd_list):
        """
        Returns new DensityMatrix new_dmat obtained by tracing an input
        DensityMatrix dmat over all nodes except those in the list
        fin_nd_list. For example, suppose dmat=dmat_abc is a DensityMatrix
        with ord_nodes = [a, b, c]. If fin_nd_list = [a, b], then this
        function returns tr_c dmat_abc = dmat_ab. This function works even
        if fin_node_list=[].

        Parameters
        ----------
        dmat : DensityMatrix
        fin_nd_list : list[BayesNode]

        Returns
        -------
        DensityMatrix

        """
        assert dmat.nodes >= set(fin_nd_list)
        fin_dmat = DensityMatrix(fin_nd_list)

        traced_nd_list = [nd for nd in dmat.ord_nodes
                          if nd not in fin_nd_list]
        dmat.set_to_transpose(traced_nd_list + fin_nd_list)
        # print('dmat', dmat)

        arr = dmat.dmat_arr
        for k in range(len(traced_nd_list)):
            arr = np.trace(arr, axis1=0, axis2=dmat.num_nodes-k)
        fin_dmat.dmat_arr = arr
        return fin_dmat

    def trace(self):
        """
        Returns the result of tracing over all nodes of self.

        Returns
        -------
        complex

        """
        return DensityMatrix.new_from_tr_of_mixed_st(self, []).dmat_arr

    def tr_normalize_self(self):
        """
        Divides all entries of self.dmat_arr by self.trace(). Returns
        self.trace()

        Returns
        -------
        complex

        """
        tr = self.trace()
        self.dmat_arr /= tr
        return tr

    def get_sq_dataframe(self):
        """
        Returns a pandas Dataframe obtained by reshaping dmat_arr to a
        square matrix. The rows and columns of the dataframe are labeled by
        tuples of ints in C order (last entry varies faster), where the ints
        represent the states of the nodes in ord_nodes

        Returns
        -------
        pandas.DataFrame

        """
        indices = list(ut.cartesian_product(self.nd_sizes))
        num_rows = np.prod(self.nd_sizes)
        init_shape = self.dmat_arr.shape
        # print("init_shape", init_shape)
        new_shape = (num_rows, num_rows)
        # print("new_shape", new_shape)
        self.dmat_arr = np.reshape(self.dmat_arr, new_shape)
        df = pd.DataFrame(self.dmat_arr, index=indices, columns=indices)
        self.dmat_arr = np.reshape(self.dmat_arr, init_shape)
        return df

    def get_sq_array(self):
        """
        Returns a copy of self.dmat_arr reshaped as a square array.

        Returns
        -------
        numpy.ndarray

        """
        num_rows = np.prod(self.nd_sizes)
        new_shape = (num_rows, num_rows)
        arr = np.reshape(cp.copy(self.dmat_arr), new_shape)
        return arr

    def get_impurity(self):
        """
        Returns abs(trace(den_mat^2) -1). This is zero iff the density
        matrix den_mat represents a pure state. For example, for a pure
        state den_mat = |a><a|, den_mat^2 = den_mat = |a><a| so this
        quantity is indeed zero.

        Returns
        -------
        float

        """
        sq_arr = self.get_sq_array()
        return abs(np.trace(np.dot(sq_arr, sq_arr)) - 1)

    def get_eigen_pot(self, fin_nd_list):
        """
        This function first marginalizes self to fin_nd_list. Then it
        reshapes that marginal to a square array, and calculates the
        eigenvalues of that square array. Finally, it returns a Potential
        pot such that pot.pot_arr contains the eigenvalues as an array with
        shape = [x.size for x in fin_nd_list].

        Parameters
        ----------
        fin_nd_list : list[BayesNode]

        Returns
        -------
        Potential

        """
        assert self.nodes >= set(fin_nd_list)
        sub_dmat = DensityMatrix.new_from_tr_of_mixed_st(self, fin_nd_list)
        arr = sub_dmat.dmat_arr
        shp = sub_dmat.nd_sizes
        num_rows = np.prod(np.array(shp))
        arr = np.reshape(arr, (num_rows, num_rows))
        evals = np.linalg.eigvalsh(arr)
        evals = np.reshape(evals, shp)
        return Potential(False, fin_nd_list, pot_arr=evals)

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

    def slicex_from_axes(self, indices1, indices2, axes1):
        """
        slicex is a portmanteau that stands for slice index. This function
        works hand in hand with __getitem__ and __setitem__ which override
        getting and setting via [ ]. It takes 3 lists: indices1, indices2
        and axes1, all of the same length and in 1-1 correspondence. It
        returns a slicex generated by (1) padding the lists 'indices1' and
        'indices2' with extra slice(None) indices at the same positions and
        (2) adding the padded indices1 list to the padded indices2 list.

        Parameters
        ----------
        indices1 : list[int]
        indices2 : list[int]
        axes1 : list[int]

        Returns
        -------
        tuple

        """

        assert len(indices1) == len(axes1)
        assert len(indices2) == len(axes1)
        assert len(axes1) <= self.num_nodes

        padded_indices1 = [slice(None)]*self.num_nodes
        padded_indices2 = [slice(None)]*self.num_nodes

        for k in range(len(axes1)):
            padded_indices1[axes1[k]] = indices1[k]
            padded_indices2[axes1[k] + self.num_nodes] = indices2[k]
        padded_indices = padded_indices1 + padded_indices2

        # Will make slicex a tuple to avert advanced indexing
        # We want just a basic slice
        return tuple(padded_indices)

    def slicex_from_nds(self, indices1, indices2, node_list):
        """
        The _nds version of this function has node_list as argument,
        the _axes version has axes instead, but they return the same thing.

        Parameters
        ----------
        indices1 : list[int]
        indices2 : list[int]
        node_list : list[BayesNode]

        Returns
        -------
        tuple

        """

        return self.slicex_from_axes(indices1, indices2,
                                    self.get_axes(node_list))

    def set_to_transpose(self, node_list):
        """
        node_list should be a permutation of self.ord_nodes. Like numpy
        does, we will use the words "permutation" and "transpose"
        interchangeably. This function replaces ord_nodes by node_list and
        applies corresponding numpy transposition to the first half and the
        second half of the axes of self.dmat_arr.

        Parameters
        ----------
        node_list : list[BayesNode]

        Returns
        -------
        None

        """
        assert set(node_list) == self.nodes
        axes = self.get_axes(node_list)
        double_axes = axes + [ax + self.num_nodes for ax in axes]

        self.dmat_arr = np.transpose(self.dmat_arr, double_axes)
        self.ord_nodes = node_list
        self.nd_sizes = [node.size for node in self.ord_nodes]

    @staticmethod
    def mag(dmat):
        """
        mag= magnitude. Returns the norm of self.dmat_arr, where by norm we
        mean the usual norm used in Quantum Mechanics, called either
        Frobenius or 2-norm, \sqrt(\sum_index  abs(dmat_arr[index])^2)

        Parameters
        ----------
        dmat : DensityMatrix

        Returns
        -------
        float

        """
        return np.linalg.norm(dmat.dmat_arr)

    @staticmethod
    def distance(dmat1, dmat2):
        """
        Returns the mag of (dmat1 - dmat2).

        Parameters
        ----------
        dmat1 : DensityMatrix
        dmat2 : DensityMatrix

        Returns
        -------
        float

        """
        return DensityMatrix.mag(dmat1 - dmat2)

    def __getitem__(self, slicex):
        """
        Overrides the [] getter. This allows us to use a slicex as an index
        of a DensityMatrix object.


        Parameters
        ----------
        slicex : tuple

        Returns
        -------
        complex

        """
        return self.dmat_arr[slicex]

    def __setitem__(self, slicex, value):
        """
        Overrides the [] setter. This allows us to use a slicex as
        an index of a DensityMatrix object.

        Parameters
        ----------
        slicex : tuple
        value : complex

        Returns
        -------
        None

        """
        self.dmat_arr[slicex] = value

    def __eq__(self, other):
        """
        Overrides ==. This defines what it means for two dmats to be equal.
        The 2 dmats don't have to have the same dmat_arr. As long as one can
        be converted to the other using set_to_transpose(), they are
        considered equal.

        Parameters
        ----------
        other : DensityMatrix

        Returns
        -------
        bool

        """
        return DensityMatrix.distance(self, other) < 1e-6

    def __ne__(self, other):
        """
        Overrides !=

        Parameters
        ----------
        other : DensityMatrix

        Returns
        -------
        bool

        """
        return not self.__eq__(other)

    def dmat_op(self, right, arr_op):
        """
        This private method will be used to override binary operators
        __add__, __sub__, __mult__ and __truediv__ for dmats. A re-alignment
        of the dmat axes is required before applying the binary operator to
        two numpy dmat_arr's.

        Parameters
        ----------
        right : DensityMatrix
        arr_op : wrapper_descriptor
            This is going to be either
            np.ndarray.[__add__, __sub__, __mul__]

        Returns
        -------
        DensityMatrix

        """
        if isinstance(right, (int, float, complex)):
            if arr_op == np.ndarray.__mul__:
                new_dmat_arr = arr_op(self.dmat_arr, right)
            else:
                right1 = DensityMatrix(self.ord_nodes, diag_val=right)
                new_dmat_arr = arr_op(self.dmat_arr, right1.dmat_arr)
            new = DensityMatrix(self.ord_nodes, dmat_arr=new_dmat_arr)
        else:
            # nlist = node list

            self_only_nlist = list(self.nodes - right.nodes)
            overlap_nlist = list(self.nodes & right.nodes)
            right_only_nlist = list(right.nodes - self.nodes)

            la = len(self_only_nlist)
            lb = len(overlap_nlist)
            lc = len(right_only_nlist)

            new = DensityMatrix(
                self_only_nlist + overlap_nlist + right_only_nlist)

            self.set_to_transpose(
                self_only_nlist + overlap_nlist)
            right.set_to_transpose(
                overlap_nlist + right_only_nlist)

            self_slicex = [slice(None)]*(la+lb)
            self_slicex += [np.newaxis]*lc
            self_slicex = tuple(self_slicex*2)

            right_slicex = [np.newaxis]*la
            right_slicex += [slice(None)]*(lb+lc)
            right_slicex = tuple(right_slicex*2)

            # numpy array magic
            # print("self[self_slicex].shape", self[self_slicex].shape)
            # print("right[right_slicex].shape", right[right_slicex].shape)
            new.dmat_arr = arr_op(
                self[self_slicex], right[right_slicex])
        return new

    def dmat_iop(self, right, arr_iop):
        """
        This private method will be used to override the in place binary
        operators __iadd__, __isub__, __imult__ . for dmats. A re-alignment
        of the dmat axes is required before applying the binary operator to
        two numpy dmat_arr's.

        Parameters
        ----------
        right : DensityMatrix
        arr_iop : wrapper_descriptor
            This is going to be either
            np.ndarray.[__iadd__, __isub__]

        Returns
        -------
        DensityMatrix

        """
        if isinstance(right, (int, float, complex)):
            if arr_iop == np.ndarray.__imul__:
                arr_iop(self.dmat_arr, right)
            else:
                right1 = DensityMatrix(self.ord_nodes, diag_val=right)
                arr_iop(self.dmat_arr, right1.dmat_arr)
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
            self_slicex = tuple(self_slicex*2)

            right_slicex = [np.newaxis]*la
            right_slicex += [slice(None)]*lb
            right_slicex = tuple(right_slicex*2)

            # numpy array magic
            arr_iop(self[self_slicex], right[right_slicex])
        return self

    def __add__(self, right):
        """
        Entrywise addition (+) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """

        return self.dmat_op(right, np.ndarray.__add__)

    def __iadd__(self, right):
        """
        Entrywise in place addition (+=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """

        return self.dmat_iop(right, np.ndarray.__iadd__)

    def __sub__(self, right):
        """
        Entrywise subtraction (-) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """

        return self.dmat_op(right, np.ndarray.__sub__)

    def __isub__(self, right):
        """
        Entrywise in place subtraction (-=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """

        return self.dmat_iop(right, np.ndarray.__isub__)

    def __mul__(self, right):
        """
        Entrywise multiplication (*) of self and right. self and right can
        be defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """
        if isinstance(right, (int, float, complex)):
            return self.dmat_op(right, np.ndarray.__mul__)
        num_overlap_nds = len(set(self.nodes & right.nodes))
        num_all_nds = len(self.nodes | right.nodes)
        num_self_only_nds = len(self.nodes - right.nodes)
        self_axes = range(num_all_nds + num_self_only_nds,
            num_all_nds + num_self_only_nds + num_overlap_nds)
        right_axes = range(num_self_only_nds,
                      num_self_only_nds + num_overlap_nds)
        axes = (self_axes, right_axes)
        # print('axes', list(self_axes), list(right_axes))
        return self.dmat_op(right, lambda x, y: np.squeeze(
                np.tensordot(x, y, axes=axes)))

    def __imul__(self, right):
        """
        Entrywise in place multiplication (*=) of self and right. self node
        set must contain right node set.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """
        assert False

    def __truediv__(self, right):
        """
        Entrywise division (/) of self and right. self and right can be
        defined over different, perhaps overlapping node sets.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """
        assert False
        
    def __itruediv__(self, right):
        """
        Entrywise in place division (/=) of self and right. self node set
        must contain right node set.

        Parameters
        ----------
        right : DensityMatrix

        Returns
        -------
        DensityMatrix

        """

        assert False
        
    def __deepcopy__(self, memo):
        """
        We want deepcopy to produce a copy of dmat_arr but not of the nodes
        in self.nodes so need to override the usual deepcopy.

        Parameters
        ----------
        memo :

        Returns
        -------
        DensityMatrix

        """
        copy_dmat_arr = cp.deepcopy(self.dmat_arr)
        return DensityMatrix(
                    ord_nodes=self.ord_nodes, dmat_arr=copy_dmat_arr)

    def __str__(self):
        """
        What string is outputted by print(dmat) where dmat is an object of
        DensityMatrix? The Shadow and __str__ know.

        Returns
        -------
        str

        """
        return str([node.name for node in self.ord_nodes]) \
            + "\n" + str(self.get_sq_dataframe())
        
if __name__ == "__main__":
    # define some nodes
    a_node = BayesNode(0, name="A", size=2)
    b_node = BayesNode(1, name="B", size=3)
    c_node = BayesNode(2, name="C", size=2)
    d_node = BayesNode(3, name="D", size=3)
    e_node = BayesNode(4, name="E", size=2)

    print("\n-----------------define some density matrices")
    rho_ab = DensityMatrix([a_node, b_node])
    print("rho_ab:", rho_ab)

    rho_ab.set_to_random(max_int=10)
    print("rho_ab:", rho_ab)
    
    rho_ab2 = DensityMatrix([a_node, b_node])
    rho_bc = DensityMatrix([b_node, c_node])

    rho_abc = DensityMatrix([a_node, b_node, c_node])
    print("rho_abc:", rho_abc)

    rho_abc.set_all_entries_to(2 + 5j)
    print("rho_abc:", rho_abc)

    rho_abc.set_to_random()

    rho_dbc = DensityMatrix([d_node, b_node, c_node])
    rho_dbc.set_to_random()

    print("\n-----------------try transpose, distance and ==")

    new_abc = cp.deepcopy(rho_abc)
    new_abc.set_to_transpose([b_node, a_node, c_node])
    assert rho_abc == new_abc
    assert rho_abc != (new_abc + 5)

    print("distance(rho_abc, new_abc)=",
                    DensityMatrix.distance(new_abc, rho_abc))
    print("rho_abc == new_abc?", rho_abc == new_abc)

    print("distance(rho_abc, rho_bc)=",
        DensityMatrix.distance(rho_abc, rho_bc))
    print("rho_abc == rho_bc?", rho_abc == rho_bc)

    print("\n-----------------try add, sub, mult")

    print("rho_ab:", rho_ab)
    print("rho_ab2:", rho_ab2)
    print("rho_ab + 5:", rho_ab + 5)
    print("rho_ab - 5:", rho_ab - 5)
    print("rho_ab * 5:", rho_ab * 5)
    print("rho_ab + rho_ab2:", rho_ab + rho_ab2)
    print("rho_ab - rho_ab2:", rho_ab - rho_ab2)
    print("rho_ab * rho_ab2:", rho_ab * rho_ab2)
    print("rho_ab + rho_bc:", rho_ab + rho_bc)
    print("rho_ab - rho_bc:", rho_ab - rho_bc)
    print("rho_ab * rho_bc:", rho_ab * rho_bc)

    print("\n-----------------try iadd, isub")

    new_abc = cp.deepcopy(rho_abc)
    new_abc.set_to_transpose([b_node, a_node, c_node])
    new_abc += 5
    assert new_abc == rho_abc + 5

    new_abc = cp.deepcopy(rho_abc)
    new_abc.set_to_transpose([b_node, a_node, c_node])
    new_abc += rho_bc
    assert new_abc == rho_abc + rho_bc

    new_abc = cp.deepcopy(rho_abc)
    new_abc.set_to_transpose([b_node, a_node, c_node])
    new_abc -= rho_bc
    assert new_abc == rho_abc - rho_bc

    print("\n-----------------try tracing ops and normalize")

    new_abc = cp.deepcopy(rho_abc)
    new_abc.tr_normalize_self()
    # print('new_abc', new_abc)
    print('tr new_abc', new_abc.trace())

    fin_nd_list = [b_node, a_node]
    new_ba = DensityMatrix.new_from_tr_of_mixed_st(new_abc, fin_nd_list)
    print("new_ba", new_ba)

    new_a = DensityMatrix.new_from_tr_of_mixed_st(new_ba, [a_node])
    print('new_a', new_a)

    print("\n**************************")
    pot = Potential(True, [a_node, b_node, c_node])
    pot.set_to_random()
    new_bc = DensityMatrix.new_from_tr_of_pure_st(pot, [b_node, c_node])
    new_bc.tr_normalize_self()
    print("from this pot", pot)
    print("_________")
    print("new_bc", new_bc)
    new_c = DensityMatrix.new_from_tr_of_mixed_st(new_bc, [c_node])
    print('new_c', new_c)
