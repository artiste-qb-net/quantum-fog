# import numpy as np

from DiscreteCondPot import *
from BayesNode import *
import Utilities as ut


class Marginalizer(BayesNode):

    def __init__(self, id_num, name,
            is_quantum, pa_nd, projected_axis, has_commas=True):

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
                    # remember, focus node is last index
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



