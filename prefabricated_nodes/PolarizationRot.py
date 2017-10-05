# from DiscreteCondPot import *
from nodes.BayesNode import *
from prefabricated_nodes.BeamSplitter import *
import math
import cmath
import Utilities as ut


class PolarizationRot(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a polarization rotation of an electric field E to
    another E'. 'theta_degs' is the angle in degrees from E to E'.

    The following is expected:

    * the focus node has exactly one parent node,

    * the parent node is a vector-field node, meaning it has states labelled
    (Nx, Ny), where Nx and Ny are integers.

    Quantum Fog gives names of the vector-field form (Nx, Ny) to the states
    of the Polarization Rotator.

    See BeamSplitter class for explanation of parameter 'max_n_sum'

    More information about polarization rotation nodes can be found in the
    documents entitled "Quantum Fog Manual", and "Quantum Fog Library Of
    Essays" that are included with the legacy QFog.

    Attributes
    ----------
    max_n_sum : int
    potential : Potential
    theta_degs : float
    true_max_n_sum : int

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
        max_n_sum : int

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
        # POL_ROTATOR::obey_amp_gen()

        theta_rads = self.theta_degs*math.pi/180
        coseno = math.cos(theta_rads)
        seno = math.sin(theta_rads)
        tau_mag = abs(coseno)
        tau_degs = 0 if coseno > 0 else 180
        rho_degs = 0 if seno > 0 else 180

        num_of_in_sts = len(mx)
        row = -1
        degen = 0

        for nx in range(self.max_n_sum+1):
            for ny in range(self.max_n_sum - nx+1):
                tm_row_starting = True
                for in_st in range(num_of_in_sts):
                    z = BeamSplitter.get_bs_amp(
                            nx, ny, mx[in_st], my[in_st],
                            tau_mag, tau_degs, rho_degs)
                    if abs(z) >= 1e-6:
                        if dry_run:
                            degen += 1
                            break  # goto next nx,ny pair
                        else:
                            if tm_row_starting:
                                row += 1
                                self.state_names[row] = \
                                    '(' + str(nx) + ',' + str(ny) + ')'
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

if __name__ == "__main__":

    theta_degs = 35
    max_n_sum = 3

    in_nd = BayesNode(0, "parent1", size=6)

    in_nd.set_state_names_to_product(
        [range(2), range(3)], trim=False)

    pr = PolarizationRot(1, "pol_rot", in_nd, theta_degs, max_n_sum)

    print("in_nd state names: ", in_nd.state_names)
    print("pol rot state names: ", pr.state_names)
    print(pr.potential)
    print("full dict of total probs: ",
          pr.potential.get_total_probs())
    print("brief dict of total probs: ",
          pr.potential.get_total_probs(brief=True))
