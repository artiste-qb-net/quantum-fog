from DiscreteCondPot import *
from BayesNode import *


class CNot(BayesNode):
    """


    Attributes
    ----------
    pa1_is_control : bool
    flipped_by_0 : bool

    """

    def __init__(self, id_num, name,
            is_quantum, pa1, pa2, pa1_is_control, flipped_by_0):
        """

        Parameters
        ----------
        is_quantum : bool
        pa1 : BayesNode
        pa2 : BayesNode
        pa1_is_control : bool
        flipped_by_0 : bool

        Returns
        -------

        """
        self.pa1_is_control = pa1_is_control
        self.flipped_by_0 = flipped_by_0

        assert pa1.size == 2 & pa2.size == 2, "The parent nodes of the CNot" \
            "don't both have size 2"
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

        Parameters
        ----------
        pa1_st : int
        pa2_st : int

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