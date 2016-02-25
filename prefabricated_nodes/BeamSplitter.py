from DiscreteCondPot import *
from BayesNode import *
import math
import cmath
import Utilities as ut


class BeamSplitter(BayesNode):

    def __init__(self, id_num, name, in_nd1, in_nd2,
            tau_mag, tau_degs, rho_degs, num_of_comps, max_n_sum=10000):

        self.tau_mag = tau_mag
        self.tau_degs = tau_degs
        self.rho_degs = rho_degs
        self.num_of_comps = num_of_comps
        # self.max_n_sum  and self.true_max_n_sum defined later

        assert 0 <= tau_mag <= 1, "tau_mag must be between 0 and 1"
        assert num_of_comps == 1 or num_of_comps == 2, \
            "number of components must be 1 or 2"

        if self.num_of_comps == 1:
            m1 = [int(name) for name in in_nd1.state_names]
            m2 = [int(name) for name in in_nd2.state_names]
            m1x = m1
            m2x = m2
            m1y = None
            m2y = None
            self.true_max_n_sum = max(m1x) + max(m2x)
        else:
            m1 = [map(int, ut.fix(name, '() ', '').split(','))
                    for name in in_nd1.state_names]
            m2 = [map(int, ut.fix(name, '() ', '').split(','))
                    for name in in_nd2.state_names]
            # x = [(1,2), (3,4), (8,9)]
            # y = zip(*x)
            # y
            # [(1, 3, 8), (2, 4, 9)]
            m1x, m1y = zip(*m1)
            m2x, m2y = zip(*m2)
            self.true_max_n_sum = max(m1x) + max(m2x) + max(m1y) + max(m2y)

        if max_n_sum > self.true_max_n_sum:
            max_n_sum = self.true_max_n_sum
        self.max_n_sum = max_n_sum

        expected_degen = self.get_expected_degen(m1x, m2x, m1y, m2y)
        assert expected_degen > 0, \
            "expected degen of beam splitter node is zero"

        BayesNode.__init__(self, id_num, name, size=expected_degen)
        self.add_parent(in_nd1)
        self.add_parent(in_nd2)

        pot = DiscreteCondPot(True, [in_nd1, in_nd2, self], bias=0)
        self.potential = pot

        self.fill_trans_mat_and_st_names_of_nd(m1x, m2x, m1y, m2y)

    @staticmethod
    def get_bs_amp(n1, n2, m1, m2, tau_mag, tau_degs, rho_degs):
        # from TWO_MODE_FUN::get_bs_amp()
        # calculates beam splitter amp
        
        tau_rads = tau_degs*math.pi/180
        rho_rads = rho_degs*math.pi/180
        rho_mag = math.sqrt(1 - tau_mag**2)
        tau = tau_mag*cmath.exp(1j*tau_rads)
        rho = rho_mag*cmath.exp(1j*rho_rads)

        # no incomming photons
        if n1+n2+m1+m2 == 0:
            return 1+0j
    
        # zero amp cases
        if n1 <= m1:
            up_lim = n1
        else:
            up_lim = m1

        if m1 <= n2:
            lo_lim = 0
        else:
            lo_lim = m1-n2

        if (n1+n2 != m1+m2) or (lo_lim > up_lim):
            return 0+0j

        # tau_mag=1 case
        n_dif = n1 - n2
        if abs(tau_mag-1) < TOL:
            if n1 == m1 and n2 == m2:
                return cmath.exp(1j*tau_rads*n_dif)
            else:
                return 0+0j

        # tau_mag=0 case
        if tau_mag < TOL:
            if n1 == m2 and n2 == m1:
                return cmath.exp(1j*(rho_degs/180*n_dif + n2)*math.pi)
            else:
                return 0+0j

        sum = 0+0j

        for j1 in range(lo_lim, up_lim+1):
            term = np.power(tau, j1)/math.factorial(j1)
            j = n2 - m1 + j1
            term = term*np.power(np.conj(tau), j)/math.factorial(j)
            j = n1 - j1
            term = term*np.power(rho, j)/math.factorial(j)
            j = m1 - j1
            term = term*np.power(np.conj(-rho), j)/math.factorial(j)
            sum += term
        
        return math.sqrt(
            math.factorial(n1)*math.factorial(
                n2)*math.factorial(m1)*math.factorial(m2))*sum

    def get_bs_amp_self(self, n1, n2, m1, m2):
        return BeamSplitter.get_bs_amp(n1, n2, m1, m2,
                        self.tau_mag, self.tau_degs, self.rho_degs)

    def fill_trans_mat_and_st_names_of_nd(
            self, m1x, m2x, m1y, m2y, dry_run=False):

        """
            notation: transition element < n1, n2 || m1, m2>
            where n1 scalar or n1 = (n1x, n1y), etc.

            m1->/--\->n2
                |  |
            m2->\--/->n1

            counter-clockwise: n1, n2, m1, m2

            mnemonic: 	n1 and n2 are the new beams, hence the n.
                 0     1
                 nw    ne
                  m2  m1
                  |   |
                  V   V
                 /-----\
                |   X   |
                 \-----/
                  |   |
                  V   V
                 n1   n2
                 sw   se
                 3     2
        """
        # this combines the following functions from legacy:
        # C_BEAM_SPL_AMP_GEN::get_expected_degen()
        # C_BEAM_SPL_AMP_GEN::fill_trans_mat_and_st_names_of_nd()
        # BEAM_SPL::obey_amp_gen()

        in_shape = [len(m1x), len(m2x)]

        row = -1
        degen = 0

        if self.num_of_comps == 1:  # scalar field case
            for n1x in range(self.max_n_sum+1):
                for n2x in range(self.max_n_sum - n1x + 1):
                    tm_row_starting = True
                    for in_st1, in_st2 in ut.cartesian_product(in_shape):
                        zx = self.get_bs_amp_self(
                            n1x, n2x, m1x[in_st1], m2x[in_st2])
                        if abs(zx) >= TOL:
                            if dry_run:
                                degen += 1
                                break  # goto next_n1_n2_pair
                            else:
                                if tm_row_starting:
                                    row += 1
                                    self.state_names[row] = \
                                        '(' + str(n1x) + ',' + str(n2x) + ')'
                                    tm_row_starting = False
                                self.potential[in_st1, in_st2, row] = zx
        else:  # vector field case
            for n1x in range(self.max_n_sum+1):
                for n1y in range(self.max_n_sum - n1x + 1):
                    for n2x in range(self.max_n_sum - n1y - n1x + 1):
                        for n2y in range(
                                self.max_n_sum - n2x - n1y - n1x + 1):
                            tm_row_starting = True
                            for in_st1, in_st2 in \
                                    ut.cartesian_product(in_shape):
                                zx = self.get_bs_amp_self(
                                    n1x, n2x, m1x[in_st1], m2x[in_st2])
                                if abs(zx) >= TOL:
                                    zy = self.get_bs_amp_self(
                                        n1y, n2y, m1y[in_st1], m2y[in_st2])
                                    if abs(zy) >= TOL:
                                        if dry_run:
                                            degen += 1
                                            break  # goto next_n_set
                                        else:
                                            if tm_row_starting:
                                                row += 1
                                                self.state_names[
                                                    row] = \
                                                    "((" + str(n1x) + "," + \
                                                    str(n1y) + "),(" \
                                                    + str(n2x) \
                                                    + "," + str(n2y) + "))"
                                                tm_row_starting = False
                                            self.potential[
                                                in_st1, in_st2, row] = zx*zy
        return degen

    def get_expected_degen(self, m1x, m2x, m1y, m2y):
        return self.fill_trans_mat_and_st_names_of_nd(
            m1x, m2x, m1y, m2y, dry_run=True)

if __name__ == "__main__":

    num_of_comps = 1

    tau_mag = .5
    tau_degs = 35
    rho_degs = 45

    if num_of_comps == 1:
        size1 = 3
        size2 = 4
        max_n_sum = 5
        pa1 = BayesNode(0, "parent1", size=size1)
        pa2 = BayesNode(1, "parent2", size=size2)

        pa1.state_names = [str(k) for k in range(size1)]
        pa2.state_names = [str(k) for k in range(size2)]

        bs = BeamSplitter(2, "a_bs", pa1, pa2,
                tau_mag, tau_degs, rho_degs, num_of_comps, max_n_sum)

        print("pa1 state names: ", pa1.state_names)
        print("pa2 state names: ", pa2.state_names)
        print("bs state names: ", bs.state_names)
        print(bs.potential)
        print("full dict of total probs: ",
              bs.potential.get_total_probs())
        print("brief dict of total probs: ",
              bs.potential.get_total_probs(brief=True))
    elif num_of_comps == 2:
        size1 = 6
        size2 = 8
        max_n_sum = 7
        pa1 = BayesNode(0, "parent1", size=size1)
        pa2 = BayesNode(1, "parent2", size=size2)

        pa1.set_state_names_to_product(
            [range(2), range(3)], trim=False)
        pa2.set_state_names_to_product(
            [range(2), range(4)], trim=False)

        bs = BeamSplitter(2, "a_bs", pa1, pa2,
                tau_mag, tau_degs, rho_degs, num_of_comps, max_n_sum)

        print("pa1 state names: ", pa1.state_names)
        print("pa2 state names: ", pa2.state_names)
        print("bs state names: ", bs.state_names)
        print(bs.potential)
        print("full dict of total probs: ",
              bs.potential.get_total_probs())
        print("brief dict of total probs: ",
              bs.potential.get_total_probs(brief=True))

