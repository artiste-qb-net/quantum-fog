from potentials.DiscreteCondPot import *
from nodes.BayesNode import *
import math


class QubitRot(BayesNode):
    """
    The Constructor of this class builds a BayesNode that has a transition
    matrix appropriate for a qubit rotation.

    The following is expected:

    * the focus node has exactly one parent node,

    * the parent node has 2 states named 0 and 1, in that order.

    Quantum Fog gives names 0 and 1 to the states of the Qbit Rotator.

    The Constructor allows you to enter the angles theta_degs[0:3] (in
    degrees) such that the amplitudes of the node are given by the entries
    of the following 2 × 2 matrix: (t =theta_degs)

    exp(i*(pi/180)*{t[0] + t[1]*sigmaX + t[2]*sigmaY + t[3]*sigmaY}),

    where sigmaX, sigmaY and sigmaZ are the Pauli Matrices.

    More information about Qbit Rotation nodes can be found in the documents
    entitled "Quantum Fog Manual", and "Quantum Fog Library Of Essays" that
    are included with the legacy QFog.

    Attributes
    ----------
    potential : Potential
    state_names : list[str]
    thetas_degs : list[float]

    """

    def __init__(self, id_num, name, pa_nd, thetas_degs):
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
        thetas_degs : list[float]

        Returns
        -------

        """

        self.thetas_degs = thetas_degs

        assert pa_nd.size == 2, "pa_nd of qubit rot does not have size 2"
        assert pa_nd.state_names == ['0', '1'], \
            "parent node state names are not 0,1"

        BayesNode.__init__(self, id_num, name, size=2)
        self.add_parent(pa_nd)

        self.state_names = ['0', "1"]

        pot = DiscreteCondPot(True, [pa_nd, self], bias=0)
        self.potential = pot

        for n in range(2):
            for m in range(2):
                self.potential[m, n] = self.qbit_rot_amp(n, m)

    def qbit_rot_amp(self, n, m):
        """
        Returns <n |rot| m> where m and n are in {0, 1}

        Parameters
        ----------
        n : int
        m : int

        Returns
        -------
        complex

        """

        rads = [self.thetas_degs[k]*math.pi/180 for k in range(4)]

        theta_mag = 0
        for k in range(1, 4):  # note: k=0 not included!!
            theta_mag += rads[k]**2
        theta_mag = math.sqrt(theta_mag)

        theta_hat = [0]*4
        if theta_mag > 1e-6:
            theta_hat = [rads[k]/theta_mag for k in range(4)]
        # theta_hat[0] = 0  will never be used

        s = math.sin(theta_mag)
        c = math.cos(theta_mag)
        if n == 0 and m == 0:
            x = c
            y = theta_hat[3]*s
        elif n == 1 and m == 1:
            x = c
            y = -theta_hat[3]*s
        elif n == 0 and m == 1:
            x = theta_hat[2]*s
            y = theta_hat[1]*s
        elif n == 1 and m == 0:
            x = -theta_hat[2]*s
            y = theta_hat[1]*s
        else:
            x = 0
            y = 0

        s = math.sin(rads[0])
        c = math.cos(rads[0])

        return (c*x - s*y) + 1j*(c*y + s*x)


if __name__ == "__main__":
    pa_nd = BayesNode(0, "pa_nd", size=2)
    pa_nd.state_names = ['0', '1']

    thetas_degs = [20, 30, 40, 60]
    qr = QubitRot(1, "rot", pa_nd, thetas_degs)

    print("pa_nd state names: ", pa_nd.state_names)
    print("qubit rot state names: ", qr.state_names)
    print(qr.potential)
    print(qr.potential.get_total_probs())
