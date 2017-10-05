from potentials.DiscreteCondPot import *
from nodes.BayesNode import *
import math
import cmath


class PhaseShifter(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a phase shifter.

    The following is expected:

    * the focus node has precisely one parent.

    * if 'occ_nums' is True (occ_num = occupation number), the states of the
    parent node are expected to be non-negative integers.

    Quantum Fog defines the set of states of the phase-shifter to be exactly
    the same as the set of states of its parent node.

    Suppose occ_nums is True. When the state of the parent node is N,
    Quantum Fog assigns amplitude exp(1j*N*theta_degs*pi/180) to state N of
    the phase-shifter, and it assigns zero amplitude to all other states of
    the phase-shifter. Note that N must be an integer.

    Suppose occ_nums is False. When the state of the parent node is STR,
    Quantum Fog assigns amplitude exp(1j*theta_degs*pi/180) to state STR of
    the phase-shifter, and it assigns zero amplitude to all other states of
    the phase-shifter. Note that STR need not be an integer.

    More information about phase shifter nodes can be found in the documents
    entitled "Quantum Fog Manual", and "Quantum Fog Library Of Essays" that
    are included with the legacy QFog.

    Attributes
    ----------
    occ_nums : bool
    potential : Potential
    state_names : list[str]
    theta_degs : float

    """

    def __init__(self, id_num, name,
            pa_nd, theta_degs, occ_nums=False):
        """
        Constructor

        Parameters
        ----------
        id_num : int
            id number of self (focus node)
        name : str
            name of self (focus node)
        pa_nd : BayesNode
            parent node
        theta_degs : float
        occ_nums : bool
            True (False) if the states of the parent node are (are not)
            occupation numbers.

        Returns
        -------

        """

        self.theta_degs = theta_degs
        self.occ_nums = occ_nums

        BayesNode.__init__(self, id_num, name, size=pa_nd.size)
        self.add_parent(pa_nd)

        self.state_names = pa_nd.state_names

        pot = DiscreteCondPot(True, [pa_nd, self], bias=0)
        self.potential = pot

        theta_rads = theta_degs*math.pi/180
        for k in range(pa_nd.size):
            if not occ_nums:
                phase = theta_rads
            else:
                num = int(pa_nd.state_names[k])
                phase = num*theta_rads
            self.potential[k, k] = cmath.exp(1j*phase)

if __name__ == "__main__":
    pa_nd = BayesNode(0, "pa_nd", size=4)
    pa_nd.state_names = [str(k) for k in range(4)]

    p_sh = PhaseShifter(1, "pshifter", pa_nd, 30, occ_nums=True)

    print("pa_nd state names: ", pa_nd.state_names)
    print("phase shifter state names: ", p_sh.state_names)
    print(p_sh.potential)
    print(p_sh.potential.get_total_probs())
