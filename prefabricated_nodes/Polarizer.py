from potentials.DiscreteCondPot import *
# from nodes.BayesNode import *
from prefabricated_nodes.BeamSplitter import *
import math
import cmath
import Utilities as ut


class Polarizer(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a polarizer that projects an electric field E to
    to an electric field E'. 'theta_degs' is the angle in degrees between
    the X axis and the direction of the polarization axis, which is the same
    as the direction of E'.

    The following is expected:

    * the focus node has exactly one parent node,

    * the parent node is a vector-field node, meaning it has states labelled
    (Nx, Ny), where Nx and Ny are integers.

    Quantum Fog gives names of the form (Nx, Ny)Nloss to the states of the
    Polarizer. Nx, Ny and Nloss are non-negative integers. Nx refers to the
    number of outgoing photons polarized in the X direction, Ny to the number of
    outgoing photons polarized in the Y direction, and Nloss to the number of
    photons absorbed by the polarizer.

    See BeamSplitter class for explanation of parameter 'max_n_sum'

    More information about polarizer nodes can be found in the
    documents entitled "Quantum Fog Manual", and "Quantum Fog Library Of
    Essays" that are included with the legacy QFog.

    Attributes
    ----------
    max_n_sum : int
    potential : Potential
    theta_degs : float
    true_max_n_sum : bool

    """
    
    def __init__(self, id_num, name, in_nd, theta_degs, max_n_sum=10000):
        """
        Constructor

        Parameters
        ----------
        id_num : int
            id number of self (focus node)
        name : str
            name of self (focus node)
        in_nd : BayesNode
            input node
        theta_degs : float
        max_n_sum : float

        Returns
        -------

        """

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
        """
        When dry_run=False, this method fills the transition matrix and
        state names of the focus node. For dry_run=True, this method doesn't
        change any of the attributes of the self object; it just calculates
        the expected size=degeneracy of the focus node.

        Parameters
        ----------
        mx : list[int]
        my : list[int]
        dry_run : bool

        Returns
        -------
        None | int

        """

        # This combines the following functions from legacy:
        # C_PHASOR_YZER_AMP_GEN::get_expected_degen()
        # POLARIZER::obey_amp_gen()

        num_of_in_sts = len(mx)
        row = -1
        degen = 0

        for nx in range(self.max_n_sum+1):
            for ny in range(self.max_n_sum - nx + 1):
                for nloss in range(self.max_n_sum - nx - ny + 1):
                    tm_row_starting = True
                    for in_st in range(num_of_in_sts):
                        z = self.get_pol_amp(
                                    nx, ny, nloss, mx[in_st], my[in_st])
                        if abs(z) >= 1e-6:
                            if dry_run:
                                degen += 1
                                break  # goto next nx, ny, nloss triple
                            else:  
                                if tm_row_starting:
                                    row += 1
                                    self.state_names[row] = \
                                        "(" + str(nx) + "," + \
                                        str(ny) + ")" + str(nloss)
                                    tm_row_starting = False
                                self.potential[in_st, row] = z
        return degen

    def get_expected_degen(self, mx, my):
        """
        Get expected degeneracy=size of focus node.

        Parameters
        ----------
        mx : list[int]
        my : list[int]

        Returns
        -------
        int

        """
        return self.fill_trans_mat_and_st_names_of_nd(
            mx, my, dry_run=True)

    def get_pol_amp(self, nx, ny, nloss, mx, my):
        """
        Calculate the polarizer amplitude.

        notation <nx, ny, nloss | operator | mx, my>

        Parameters
        ----------
        nx : int
        ny : int
        nloss : int
        mx : int
        my : int

        Returns
        -------
        complex

        """

        theta_rads = self.theta_degs*math.pi/180
        coseno = math.cos(theta_rads)
        seno = math.sin(theta_rads)
        tau_mag = abs(coseno)
        tau_degs = 0 if coseno > 0 else 180
        rho_degs = 0 if seno > 0 else 180

        # seno = 0 case
        if abs(seno) < 1e-6:
            if nx == mx and ny == 0 and nloss == my:
                return pow(coseno, 2*nx + nloss)
            else:
                return 0+0j

        # coseno = 0 case
        if abs(coseno) < 1e-6:
            if nx == 0 and ny == my and nloss == mx:
                return (1 if nloss % 2 == 0 else -1)*pow(seno, 2*ny + nloss)
            else:
                return 0+0j

        z = BeamSplitter.get_bs_amp(nx + ny, nloss, mx, my,
                tau_mag, tau_degs, rho_degs)
        if abs(z) < 1e-6:
            return 0+0j
    
        return z*pow(coseno, nx)*pow(seno, ny) * \
            math.sqrt(math.factorial(nx+ny) /
                  (math.factorial(nx)*math.factorial(ny)))

if __name__ == "__main__":

    theta_degs = 35
    max_n_sum = 3

    in_nd = BayesNode(0, "parent1", size=6)

    in_nd.set_state_names_to_product(
        [range(2), range(3)], trim=False)

    pol = Polarizer(1, "pol_rot", in_nd, theta_degs, max_n_sum)

    print("in_nd state names: ", in_nd.state_names)
    print("pol state names: ", pol.state_names)
    print(pol.potential)
    print("full dict of total probs: ",
          pol.potential.get_total_probs())
    print("brief dict of total probs: ",
          pol.potential.get_total_probs(brief=True))

