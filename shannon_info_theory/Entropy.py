import numpy as np
from potentials.Potential import *
from potentials.DensityMatrix import *


class Entropy:
    """
    This class calculates classical and quantum versions of entropy,
    conditional information (CI), mutual information (MI) and conditional
    mutual info ( CMI).

    In the classical case, it uses a PD carried by a Potential pot. You can
    check that pot is legal using pot.is_joint_prob_dist()

    In the quantum case, it uses a density matrix carried by a DensityMatrix
    dmat. You can check that dmat is legal using dmat.is_legal_dmat()

    log's in entropies are base e, natural logs

    """

    @staticmethod
    def ent_c(pot, x_nds, verbose=False):
        """
        Returns the classical entropy H(x) for the PD P_x = (pot
        marginalized to x). Here x is the list of nodes x_nds which is a
        subset of pot.ord_nodes
        
        Parameters
        ----------
        pot : Potential
        x_nds : list[BayesNode]
        verbose : bool

        Returns
        -------
        float

        """
        sub_pot = pot.get_new_marginal(x_nds)
        # Replace very small probabilities
        # by 1 (because eps*ln(eps) = 0 = 1*ln(1) for 0< eps <<1)
        sub_pot.pot_arr[sub_pot.pot_arr < 1e-6] = 1.
        slicex = tuple([slice(None)]*sub_pot.num_nodes)
        ent = -(sub_pot[slicex]*np.log(sub_pot[slicex])).sum()
        if verbose:
            print('\nentropy for', [z.name for z in x_nds])
            print(ent)
        return ent

    @staticmethod
    def ent_q(dmat, x_nds, verbose=False):
        """
        Returns the quantum entropy S(x) for the density matrix dmat_x = (
        dmat marginalized to x). Here x is the list of nodes x_nds which is
        a subset of dmat.ord_nodes

        Parameters
        ----------
        dmat : DensityMatrix 
        x_nds : 
        verbose : bool 

        Returns
        -------
        float

        """
        pot = dmat.get_eigen_pot(x_nds)
        return Entropy.ent_c(pot, x_nds, verbose)

    @staticmethod
    def cond_info_c(pot, x_nds, y_nds, verbose=False):
        """
        Returns the classical conditional information (CI) H(x|y) for the PD
        P_xy = (pot marginalized to [x, y]). Here x (y, resp.) is the list
        of nodes x_nds (y_nds, resp.) which is a subset of pot.ord_nodes. x,
        y are assumed to be disjoint.

        Parameters
        ----------
        pot : Potential 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        if x_nds is None:
            x_nds = []
        if y_nds is None:
            y_nds = []

        xy_nds = x_nds + y_nds
        pot_xy = pot.get_new_marginal(xy_nds)
        hxy = Entropy.ent_c(pot_xy, xy_nds)
        hy = Entropy.ent_c(pot_xy, y_nds)
        info = hxy - hy
        if verbose:
            print('\ncond_info for', [z.name for z in x_nds], '|',
                  [z.name for z in y_nds])
            print(info)
        return info

    @staticmethod
    def cond_info_q(dmat, x_nds, y_nds, verbose=False):
        """
        Returns the quantum conditional information (CI) S(x|y) for the
        density matrix dmat_xy = (dmat marginalized to [x, y]). Here x (y,
        resp.) is the list of nodes x_nds (y_nds, resp.) which is a subset
        of dmat.ord_nodes. x, y are assumed to be disjoint.

        Parameters
        ----------
        dmat : DensityMatrix 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        xy_nds = x_nds + y_nds
        pot = dmat.get_eigen_pot(xy_nds)
        return Entropy.cond_info_c(pot, x_nds, y_nds, verbose)

    @staticmethod
    def mut_info_c(pot, x_nds, y_nds, verbose=False):
        """
        Returns the classical mutual information (CM) H(x:y) for the PD P_xy
        = (pot marginalized to [x, y]). Here x (y, resp.) is the list of
        nodes x_nds (y_nds, resp.) which is a subset of pot.ord_nodes. x,
        y are assumed to be disjoint.

        Parameters
        ----------
        pot : Potential 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        if x_nds is None:
            x_nds = []
        if y_nds is None:
            y_nds = []

        xy_nds = x_nds + y_nds
        pot_xy = pot.get_new_marginal(xy_nds)
        hxy = Entropy.ent_c(pot_xy, xy_nds)
        hx = Entropy.ent_c(pot_xy, x_nds)
        hy = Entropy.ent_c(pot_xy, y_nds)
        info = hx + hy - hxy
        if verbose:
            print('\nmut_info for', [z.name for z in x_nds], ':',
                  [z.name for z in y_nds])
            print(info)
        return info

    @staticmethod
    def mut_info_q(dmat, x_nds, y_nds, verbose=False):
        """
        Returns the quantum mutual information (CM) S(x:y) for the density
        matrix dmat_xy = (dmat marginalized to [x, y]). Here x (y, resp.) is
        the list of nodes x_nds (y_nds, resp.) which is a subset of
        dmat.ord_nodes. x, y are assumed to be disjoint.

        Parameters
        ----------
        dmat : DensityMatrix 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        xy_nds = x_nds + y_nds
        pot = dmat.get_eigen_pot(xy_nds)
        return Entropy.mut_info_c(pot, x_nds, y_nds, verbose)

    @staticmethod
    def cond_mut_info_c(pot, x_nds, y_nds, z_nds, verbose=False):
        """
        Returns the classical conditional mutual information (CMI) H(x:y|z)
        for the PD P_xyz = (pot marginalized to [x, y, z]). Here x ( y, z,
        resp.) is the list of nodes x_nds (y_nds, z_nds, resp.) which is a
        subset of pot.ord_nodes. x, y, z are assumed to be disjoint.

        Parameters
        ----------
        pot : Potential 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        z_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        if x_nds is None:
            x_nds = []
        if y_nds is None:
            y_nds = []
        if z_nds is None:
            z_nds = []

        xz_nds = x_nds + z_nds
        yz_nds = y_nds + z_nds
        xyz_nds = x_nds + y_nds + z_nds
        pot_xyz = pot.get_new_marginal(xyz_nds)
        pot_yz = pot_xyz.get_new_marginal(yz_nds)
        hxyz = Entropy.ent_c(pot_xyz, xyz_nds)
        hz = Entropy.ent_c(pot_yz, z_nds)
        hxz = Entropy.ent_c(pot_xyz, xz_nds)
        hyz = Entropy.ent_c(pot_yz, yz_nds)
        info = hxz + hyz - hxyz - hz
        if verbose:
            print('\ncond mut_info for', [z.name for z in x_nds], ':',
                  [z.name for z in y_nds], '|', [z.name for z in z_nds])
            print(info)
        return info

    @staticmethod
    def cond_mut_info_q(dmat, x_nds, y_nds, z_nds, verbose=False):
        """
        Returns the quantum conditional mutual information (CMI) S(x:y|z)
        for the density matrix dmat_xyz = (dmat marginalized to [x, y,
        z]). Here x ( y, z, resp.) is the list of nodes x_nds (y_nds, z_nds,
        resp.) which is a subset of dmat.ord_nodes. x, y, z are assumed to
        be disjoint.

        Parameters
        ----------
        dmat : DensityMatrix 
        x_nds : list[BayesNode] 
        y_nds : list[BayesNode] 
        z_nds : list[BayesNode] 
        verbose : bool 

        Returns
        -------
        float

        """
        xyz_nds = x_nds + y_nds + z_nds
        pot = dmat.get_eigen_pot(xyz_nds)
        return Entropy.cond_mut_info_c(pot, x_nds, y_nds, z_nds, verbose)

if __name__ == "__main__":
    # define some nodes
    a_nd = BayesNode(0, name="A", size=2)
    b_nd = BayesNode(1, name="B", size=3)
    c_nd = BayesNode(2, name="C", size=2)
    d_nd = BayesNode(3, name="D", size=3)

    print('----------------------classical case')
    pot = Potential(False, [a_nd, b_nd, c_nd, d_nd])
    pot.set_to_random()
    pot /= pot.get_new_marginal([])  # this normalizes so all entries sum to 1
    assert pot.is_joint_prob_dist()
    Entropy.ent_c(pot, [a_nd], verbose=True)
    Entropy.ent_c(pot, [a_nd, c_nd], verbose=True)
    Entropy.ent_c(pot, [c_nd, a_nd], verbose=True)
    Entropy.cond_info_c(pot, [a_nd, b_nd], [c_nd], verbose=True)
    Entropy.mut_info_c(pot, [a_nd, b_nd], [c_nd], verbose=True)
    Entropy.cond_mut_info_c(pot, [a_nd], [b_nd, d_nd], [c_nd], verbose=True)

    print('----------------------quantum case')
    dmat = DensityMatrix([a_nd, b_nd, c_nd, d_nd])
    # dmat = DensityMatrix([a_nd, b_nd])
    dmat.set_to_random(normalize=True)
    # print(dmat)
    assert dmat.is_legal_dmat()
    Entropy.ent_q(dmat, [a_nd], verbose=True)
    Entropy.ent_q(dmat, [a_nd, c_nd], verbose=True)
    Entropy.ent_q(dmat, [c_nd, a_nd], verbose=True)
    Entropy.cond_info_q(dmat, [a_nd, b_nd], [c_nd], verbose=True)
    Entropy.mut_info_q(dmat, [a_nd, b_nd], [c_nd], verbose=True)
    Entropy.cond_mut_info_q(dmat, [a_nd], [b_nd, d_nd], [c_nd], verbose=True)




