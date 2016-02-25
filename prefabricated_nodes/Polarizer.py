from DiscreteCondPot import *
from BayesNode import *
# from BeamSplitter import *
import math
import cmath
import Utilities as ut

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





class Polarizer(BayesNode):
    
    def __init__(self, id_num, name, in_nd, theta_degs, max_n_sum=10000):

        self.theta_degs = theta_degs
        # self.max_n_sum  and self.true_max_n_sum defined later

        m = [map(int, ut.fix(name, '() ', '').split(','))
                for name in in_nd.state_names]
        mx, my = zip(*m)

        self.true_max_n_sum = max([mx[k] + my[k] for k in range(len(mx))])
        if max_n_sum > self.true_max_n_sum:
            max_n_sum = self.true_max_n_sum
        self.max_n_sum = max_n_sum

        expected_degen = self.get_expected_degen(mx, my)
        assert expected_degen > 0, \
            "expected degen of polarization rot node is zero"

        BayesNode.__init__(self, id_num, name, size=expected_degen)
        self.add_parent(in_nd)

        pot = DiscreteCondPot(True, [in_nd, self], bias=0)
        self.potential = pot

        self.fill_trans_mat_and_st_names_of_nd(mx, my)

    def fill_trans_mat_and_st_names_of_nd(
            self, mx, my, dry_run=False):

        # This combines the following functions from legacy:
        # C_PHASOR_YZER_AMP_GEN::get_expected_degen()
        # POLARIZER::obey_amp_gen()

        num_of_in_sts = len(mx)
        row = -1
        degen = 0

        for nx in range(self.max_n_sum+1):
            for ny in range(self.max_n_sum - nx + 1):
                for nloss in range(self.max_n_sum - nx - ny +1):
                    tm_row_starting = True
                    for in_st in range(num_of_in_sts):
                        z = self.get_pol_amp(
                                    nx, ny, nloss, mx[in_st], my[in_st])
                        if abs(z) >= TOL:
                            if dry_run:
                                degen += 1
                                break  # goto next nx, ny, nloss triple
                            else:  
                                if tm_row_starting:
                                    row += 1
                                    self.state_names[row] = \
                                        "(" + str(nx)  + "," + \
                                        str(ny) + ")" + str(nloss)
                                    tm_row_starting=False
                                self.potential[in_st, row] = z
        return degen

    def get_expected_degen(self, mx, my):
        return self.fill_trans_mat_and_st_names_of_nd(
            mx, my, dry_run=True)

    def get_pol_amp(self, nx, ny, nloss, mx, my):

        # calculates polarizer amplitude
        
        theta_rads = self.theta_degs*math.pi/180
        coseno = math.cos(theta_rads)
        seno = math.sin(theta_rads)
        tau_mag = abs(coseno)
        tau_degs = 0 if coseno > 0 else 180
        rho_degs = 0 if seno > 0 else 180

        # seno = 0 case
        if abs(seno) < TOL:
            if nx == mx and ny == 0 and nloss == my:
                return pow(coseno, 2*nx + nloss)
            else:
                return 0+0j


        # coseno = 0 case
        if abs(coseno) < TOL:
            if nx == 0 and ny == my and nloss == mx:
                return (1 if nloss%2 == 0 else -1)*pow(seno, 2*ny + nloss)
            else:
                return 0+0j

        z = get_bs_amp(nx + ny, nloss, mx, my,
                tau_mag, tau_degs, rho_degs)
        if abs(z)< TOL:
            return 0+0j
    
        return z*pow(coseno, nx)*pow(seno, ny)*\
            math.sqrt(math.factorial(nx+ny)/
                  (math.factorial(nx)*math.factorial(ny)))

if __name__ == "__main__":

    theta_degs = 35
    max_n_sum = 3

    in_nd = BayesNode(0, "parent1", size=6)

    in_nd.set_state_names_to_product(
        [range(2), range(3)], trim=False)

    pol = Polarizer(1, "pol_rot", in_nd, theta_degs, max_n_sum)

    print("in_nd state names: ", in_nd.state_names)
    print("bs state names: ", pol.state_names)
    print(pol.potential)
    print("full dict of total probs: ",
          pol.potential.get_total_probs())
    print("brief dict of total probs: ",
          pol.potential.get_total_probs(brief=True))

