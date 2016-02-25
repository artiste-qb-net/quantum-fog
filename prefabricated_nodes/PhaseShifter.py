from DiscreteCondPot import *
from BayesNode import *
import math
import cmath

class PhaseShifter(BayesNode):

    def __init__(self, id_num, name,
            pa_nd, theta_degs, occ_nums=False):

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

    ps = PhaseShifter(1, "pshifter", pa_nd, 30, occ_nums=True)

    print("pa_nd state names: ", pa_nd.state_names)
    print("phase shifter state names: ", ps.state_names)
    print(ps.potential)
    print(ps.potential.get_total_probs())