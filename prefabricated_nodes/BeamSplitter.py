from potentials.DiscreteCondPot import *
from nodes.BayesNode import *
import math
import cmath
import Utilities as ut


class BeamSplitter(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a beam splitter.

    The following is expected:

    * the focus node has exactly two parent nodes,

    * Both parent nodes are scalar-field nodes OR both parent nodes are
    vector-field nodes.

    M1   M2
      \|/
       |
      /|\
    N2   N1
    all arrows pointing downward
    (mnemonic: N for New beams=modes)

    In the scalar-field case, M2, M1, N1 and N2 are the numbers of photons
    that pass through the two incoming and the two outgoing ports. In this
    case, Quantum Fog gives names of the form (N1,N2) to the states of the
    beam-splitter.

    In the vector-field case, M2, M1, N1 and N2  also correspond in a 1-1
    fashion to the incoming and outgoing ports, but instead of being
    non-negative integers, they are 2-component vectors. For example,
    M1 = (M1x, M1y), where M1x and M1y are non-negative integers. In this
    case, Quantum Fog gives names of the form ((N1x, N1y), (N2x, N2y)) to
    the states of the beam-splitter.

    tau and rho, satisfying |tau|^2 + |rho|^2 = 1, are the complex
    transmission and reflection coefficients of the beam-splitter. You must
    enter tau_mag = the magnitude tau, tau_degs = phase of tau in degrees,
    and rho_degs = phase of rho in degrees. These 3 parameters completely
    specify the complex numbers tau and rho.

    Consider the scalar-field case, for example. Frequently, nets which
    contain a beam-splitter node are such that we know what is the maximum
    number of photons that will ever enter the beam-splitter. For example,
    suppose that a net starts with 2 photons in its root nodes, and that for
    one of the input states (M1, M2) of the beam-splitter, M1 + M2 = 3. Then
    the list of states of the beam-splitter node would be forced to include
    all states with N1 + N2 = 3. Or would it? Clearly, such states would
    never occur in any of the possible stories of the net. So if we were to
    exclude such states from the list of states of the beam-splitter node,
    the physical predictions of the net (that is, the stories with non-zero
    amplitude and their amplitudes) would still be the same. That's where
    the input parameter 'max_n_sum' comes in. In the scalar case, Quantum
    Fog lists those and only those states (N1, N2) for which N1 + N2 <=
    max_n_sum. In our example, we could set 'max_n_sum' to 2 and thus
    exclude states with N1 + N2 = 3. Of course, excluding some states would
    cause the Total Probability sum_x P(x|input states) for some input
    states to be different from 1. But the physical predictions of the net
    would not change, and we would save memory by excluding unused baggage
    from the transition matrix.

    In the vector-field case, Quantum Fog lists those and only those states
    ((N1x, N1y), (N2x, N2y)) for which N1x + N1y+ N2x+ N2y <= max_n_sum

    More information about beam splitter nodes can be found in the documents
    entitled "Quantum Fog Manual", and "Quantum Fog Library Of Essays" that
    are included with the legacy QFog.

    Attributes
    ----------
    max_n_sum : int
    num_of_comps : int
        number of components, equals 1 for scalar case and 2 for vector case.
    potential : Potential
    rho_degs : float
    tau_degs : float
    tau_mag : float
    true_max_n_sum : int

    """

    def __init__(self, id_num, name, in_nd1, in_nd2,
            tau_mag, tau_degs, rho_degs, num_of_comps, max_n_sum=10000):
        """
        Constructor

        Parameters
        ----------
        id_num : int
            id number of self (focus node)
        name : str
            name of self (focus node)
        in_nd1 : BayesNode
            input node (parent) 1
        in_nd2 : BayesNode
            input node (parent) 2
        tau_mag : float
        tau_degs : float
        rho_degs : float
        num_of_comps : int
            number of components, 1 for scalar fields and 2 for vector ones
        max_n_sum : int

        Returns
        -------

        """
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
        """
        Get beam splitter amplitude for arbitrary tau and rho, not self.tau,
        self.rho.

        Parameters
        ----------
        n1 : int
        n2 : int
        m1 : int
        m2 : int
        tau_mag : float
        tau_degs : float
        rho_degs : float

        Returns
        -------
        complex

        """
        # from TWO_MODE_FUN::get_bs_amp()
        
        tau_rads = tau_degs*math.pi/180
        rho_rads = rho_degs*math.pi/180
        rho_mag = math.sqrt(1 - tau_mag**2)
        tau = tau_mag*cmath.exp(1j*tau_rads)
        rho = rho_mag*cmath.exp(1j*rho_rads)

        # no incoming photons
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
        if abs(tau_mag-1) < 1e-6:
            if n1 == m1 and n2 == m2:
                return cmath.exp(1j*tau_rads*n_dif)
            else:
                return 0+0j

        # tau_mag=0 case
        if tau_mag < 1e-6:
            if n1 == m2 and n2 == m1:
                return cmath.exp(1j*(rho_degs/180*n_dif + n2)*math.pi)
            else:
                return 0+0j

        tot_sum = 0+0j

        for j1 in range(lo_lim, up_lim+1):
            term = np.power(tau, j1)/math.factorial(j1)
            j = n2 - m1 + j1
            term = term*np.power(np.conj(tau), j)/math.factorial(j)
            j = n1 - j1
            term = term*np.power(rho, j)/math.factorial(j)
            j = m1 - j1
            term = term*np.power(np.conj(-rho), j)/math.factorial(j)
            tot_sum += term
        
        return math.sqrt(
            math.factorial(n1)*math.factorial(
                n2)*math.factorial(m1)*math.factorial(m2))*tot_sum

    def get_bs_amp_self(self, n1, n2, m1, m2):
        """
        Get beam splitter amplitude using the focus node values of tau and
        rho.

        Parameters
        ----------
        n1 : int
        n2 : int
        m1 : int
        m2 : int

        Returns
        -------
        complex

        """
        return BeamSplitter.get_bs_amp(n1, n2, m1, m2,
                        self.tau_mag, self.tau_degs, self.rho_degs)

    def fill_trans_mat_and_st_names_of_nd(
            self, m1x, m2x, m1y, m2y, dry_run=False):
        """
        When dry_run=False, this method fills the transition matrix and
        state names of the focus node. For dry_run=True, this method doesn't
        change any of the attributes of the self object; it just calculates
        the expected size=degeneracy of the focus node.

        notation: transition matrix = <n1, n2 |operator| m1, m2>
        where n1 scalar or n1 = (n1x, n1y), etc.

        Parameters
        ----------
        m1x : list[int]
        m2x : list[int]
        m1y : list[int]
        m2y : list[int]
        dry_run : bool

        Returns
        -------
        None | int

        """
        # This combines the following functions from legacy:
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
                        if abs(zx) >= 1e-6:
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
                                if abs(zx) >= 1e-6:
                                    zy = self.get_bs_amp_self(
                                        n1y, n2y, m1y[in_st1], m2y[in_st2])
                                    if abs(zy) >= 1e-6:
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
        """
        Get expected degeneracy=size of focus node.

        Parameters
        ----------
        m1x : list[int]
        m2x : list[int]
        m1y : list[int]
        m2y : list[int]

        Returns
        -------
        int

        """
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

