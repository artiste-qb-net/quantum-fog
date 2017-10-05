from potentials.DiscreteCondPot import *
from nodes.BayesNode import *


class CNot(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a CNot (Controlled Not)

    The following is expected:

    * the focus node has exactly two parent nodes,

    * the parent nodes each has 2 states named, 0 and 1, in that order.

    Suppose M1, M2, N1, N2 are elements of {0,1}. Say M1 is the state of the
    first parent P1, and M2 is the state of the second parent P2. Define (
    N1, N2) in terms of M1 and M2 as follows.

    M1 	M2 	(N1,N2)
    0 	0 	(0,0)
    0 	1 	(0,1)
    1 	0 	(1,1)
    1 	1 	(1,0)

    This table may be described by saying that

    * N1 = M1  and M2 is transformed into N2 ,

    * P1 (the "control" bit) causes P2 (the "target" bit) to flip (i.e.,
    to change from 0 to 1 or vice versa) whenever M1 is 1, and

    * P1 has no effect on P2 whenever M1 is 0.

    Quantum Fog gives names N1N2 (i.e., 00, 01, 10 and 11) to the states of
    the Controlled-Not. When the states M1 and M2 are given by a particular
    row of the above table, Quantum Fog assigns unit amplitude to the state
    N1N2 shown in that row, and it assigns zero amplitude to all the other
    states.

    More information about CNot nodes can be found in the documents
    entitled "Quantum Fog Manual", and "Quantum Fog Library Of Essays" that
    are included with the legacy QFog.

    Attributes
    ----------
    flipped_by_0 : bool
        True (False) if target flips when control is in state 0 (state 1).
        In the above table example, flipped_by_0=False
    pa1_is_control : bool
        True (False) if parent 1 (parent 2) is the control. In the above
        table example, pa1_is_control=True
    potential : Potential

    """

    def __init__(self, id_num, name,
            is_quantum, pa1, pa2, pa1_is_control, flipped_by_0):
        """
        Constructor

        Parameters
        ----------
        id_num : int
            id number of self (focus node)
        name : str
            name of self (focus node)
        is_quantum : bool
        pa1 : BayesNode
            parent 1
        pa2 : BayesNode
            parent 2
        pa1_is_control : bool
            True (False) when parent 1 (parent 2) is control
        flipped_by_0 : bool
            True (False) when target flips when state of control is 0 (1)

        Returns
        -------

        """
        self.pa1_is_control = pa1_is_control
        self.flipped_by_0 = flipped_by_0

        assert pa1.size == 2 & pa2.size == 2, \
            "The parent nodes of the CNot don't both have size 2"
        assert pa1.state_names == ['0', '1'], "parent1 states not 0,1"
        assert pa2.state_names == ['0', '1'], "parent2 states not 0,1"

        BayesNode.__init__(self, id_num, name, size=4)
        self.add_parent(pa1)
        self.add_parent(pa2)

        self.set_state_names_to_product(['01'], repeat=2, trim=True)

        pot = DiscreteCondPot(is_quantum, [pa1, pa2, self], bias=0)
        self.potential = pot

        for st1 in range(2):
            for st2 in range(2):
                self.potential[st1, st2, self.final_st(st1, st2)] = 1

    def final_st(self, pa1_st, pa2_st):
        """
        Give final state (an element of 0=00, 1=01, 2=10, 3=11) of focus
        node, given the states of its 2 parents.

        Parameters
        ----------
        pa1_st : int
            parent 1 state
        pa2_st : int
            parent 2 state

        Returns
        -------
        int

        """
        if self.flipped_by_0:
            x = 1 - pa1_st
        else:  # flipped by 1
            x = pa1_st
        if self.pa1_is_control:
            bit0 = pa1_st
            bit1 = (x + pa2_st)//2
        else:  # pa2 is control
            bit0 = (x + pa2_st)//2
            bit1 = pa2_st
        foc_st = bit0 + 2*bit1
        return foc_st


if __name__ == "__main__":
    pa1 = BayesNode(0, "parent1", size=2)
    pa2 = BayesNode(1, "parent2", size=2)

    pa1.state_names = ['0', '1']
    pa2.state_names = ['0', '1']

    cn = CNot(2, "a_cnot",
        False, pa1, pa2, True, True)

    print("pa1 state names: ", pa1.state_names)
    print("pa2 state names: ", pa2.state_names)
    print("cnot state names: ", cn.state_names)
    print(cn.potential)
    print(cn.potential.get_total_probs())
