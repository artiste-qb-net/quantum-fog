# import numpy as np

from potentials.DiscreteCondPot import *
from nodes.BayesNode import *
import Utilities as ut


class Marginalizer(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a marginalizer node. Given a quantum state |a0,
    a1, a2,....> and asked to project along axis or component 2,
    a marginalizer node will give back |a2>. That is the gist of it.

    The following is expected:

    * the focus node has exactly one parent node,

    * the states of the parent node are of one of two forms: has_commas=True
    or False

    When 'has_commas' is True, the states of the parent node must be of the
    form (a_0, a_1, a_2,... a_{n-1}) where a_0 \in S_0, a_1 \in S_1,
    etc. The sets S_0, S_1, ... S_{n-1} need not be the same. If
    'projected_axis' equals k, then QFog will use S_k as the states of the
    focus node. The transition matrix will be assigned the value \delta(b_k,
    a_k), where \delta() is the Kronecker delta function, b_k is the state
    of the focus node and a_k is the kth component of the state of the
    parent node.

    The case 'has_commas' is False is virtually identical to the True case,
    except that now the states of the parent node are not tuples, and don't
    have commas. Instead they are all strings of the same length. In this
    case, projected axis k refers to the position k (0 based) on the string.

    More information about marginalizer nodes can be found in the documents
    entitled "Quantum Fog Manual", and "Quantum Fog Library Of Essays" that
    are included with the legacy QFog.

    Attributes
    ----------
    has_commas : bool
    potential : Potential
    projected_axis : int
    state_names : list[str]

    """

    def __init__(self, id_num, name,
            is_quantum, pa_nd, projected_axis, has_commas=True):
        """
        Constructor

        Parameters
        ----------
        id_num : int
            id number of self (focus node)
        name : str
            name of self (focus node)
        is_quantum : bool
        pa_nd : BayesNode
            parent node
        projected_axis : int
        has_commas : bool

        Returns
        -------

        """

        self.projected_axis = projected_axis
        self.has_commas = has_commas

        if has_commas:
            bad = '() '
            # rep = repetitive
            rep_name_list = [ut.fix(name, bad, '').split(',')[projected_axis]
                            for name in pa_nd.state_names]
        else:
            rep_name_list = [name[projected_axis]
                            for name in pa_nd.state_names]
        non_rep_name_list = sorted(list(set(rep_name_list)))

        size = len(non_rep_name_list)
        BayesNode.__init__(self, id_num, name, size=size)
        self.add_parent(pa_nd)

        self.state_names = non_rep_name_list

        pot = DiscreteCondPot(is_quantum, [pa_nd, self], bias=0)
        self.potential = pot

        for k, name in enumerate(self.state_names):
            for r, pa_name in enumerate(rep_name_list):
                if name == pa_name:
                    # remember, focus node is last topo_index
                    self.potential[r, k] = 1

if __name__ == "__main__":
    pa_nd = BayesNode(0, "pa_nd", size=24)
    # has_commas = True
    has_commas = False
    pa_nd.set_state_names_to_product(
        [range(2), range(3), range(4)], trim=not has_commas)
    marg = Marginalizer(1, "marg",
        False, pa_nd, projected_axis=1, has_commas=has_commas)

    print("pa state names: ", pa_nd.state_names)
    print("marg state names: ", marg.state_names)
    print(marg.potential)
    print(marg.potential.get_total_probs())



