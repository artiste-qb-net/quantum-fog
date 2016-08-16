import numpy as np
import csv
from potentials.Potential import *


class RandGen_NetParams:
    """
    RandGen_NetParams (Random Generator of Net Parameters). This class
    generates random parameters (i.e. either a single pot or all pots) for a
    given dag structure.

    The states are sampled the same way in the classical and quantum cases.
    In the quantum case, if a node C with parents pa(C) has C=x and pa(C)=y,
    where x and y are the sampled states, then we set the phase of node C
    equal to the phase of the amplitude A( C=x | pa(C)=y ).

    Attributes
    ----------
    bnet : BayesNet
        The pots of this bnet are sampled to generate a states_df and
        also a degs_df in the quantum case.
    do_int_df : bool
        If False, the states_df generated has state names as entries. If
        True, states_df has int entries. The int entries are the index in
        the states_names list of the node for that column.
    is_quantum : bool
        True if quantum bnets and False if classical ones
    num_samples : int
        The number of samples = len(states_df.index) = len(degs_df.index)
    topo_nd_list : list[BayesNode]
        List of the nodes of the bnet in topological (=chronological) order,
        root node first
    """

    def __init__(self, is_quantum, bnet, num_samples, do_int_df=False):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        bnet : BayesNet
        num_samples : int
        do_int_df : bool

        Returns
        -------

        """

        self.is_quantum = is_quantum
        self.bnet = bnet
        self.num_samples = num_samples
        self.do_int_df = do_int_df
        self.topo_nd_list = \
            [bnet.get_node_with_topo_index(k) for k in range(bnet.num_nodes)]

    def sam_generator(self):
        """
        A generator of samples. The generator yields two dictionaries,
        nd_to_int_st and nd_to_degs. nd_to_int_st maps each node to its
        sampled state given as an integer (the integer being the index of
        the state in the node_states list of the node). In the quantum case,
        nd_to_degs maps each node to its sampled phase in degrees. In the
        classical case, nd_to_degs = None

        Returns
        -------
        (dict[int], dict[float])

        """
        # In this function, it is essential that we sample the nodes
        # in topo order with root node first so that the state of
        # each parent is fixed before the state of their children are
        # sampled
        nd_to_int_st = {}
        if self.is_quantum:
            nd_to_degs = {}
        else:
            nd_to_degs = None
        sam_ctr = 0
        while sam_ctr < self.num_samples:
            for nd in self.topo_nd_list:
                pot = nd.potential
                nd_sts = list(range(nd.size))
                if nd.parents:
                    ord_pa = nd.potential.ord_nodes[:-1]
                    ord_pa_sts = [nd_to_int_st[x] for x in ord_pa]
                    nd_pot_vals = [pot[tuple(ord_pa_sts+[k])] for k in nd_sts]
                else:
                    nd_pot_vals = [pot[k] for k in nd_sts]

                if self.is_quantum:
                    nd_probs = np.real(nd_pot_vals*np.conj(nd_pot_vals))
                else:
                    nd_probs = nd_pot_vals

                nd_int_st = np.random.choice(nd_sts, p=nd_probs)
                nd_to_int_st[nd] = nd_int_st

                if self.is_quantum:
                    nd_to_degs[nd] = np.angle(
                        nd_pot_vals[nd_int_st], deg=True)

            yield nd_to_int_st, nd_to_degs

            sam_ctr += 1

    def write_csv(self, sts_file_path, degs_file_path=None):
        """
        Writes a cvs file (comma separated values) for states_df at the path
        sts_file_path and for degs_df at the path degs_file_path

        Parameters
        ----------
        sts_file_path : str
        degs_file_path : str or NoneType

        Returns
        -------

        """

        header = [self.topo_nd_list[k].name for
                  k in range(self.bnet.num_nodes)]
        fi_sts = open(sts_file_path, 'w', newline='')
        wr_sts = csv.writer(fi_sts)
        wr_sts.writerows([header])
        if self.is_quantum:
            fi_degs = open(degs_file_path, 'w', newline='')
            wr_degs = csv.writer(fi_degs)
            wr_degs.writerows([header])
        for nd_to_int_st, nd_to_degs in self.sam_generator():
            if self.do_int_df:
                row = [str(nd_to_int_st[nd])
                       for nd in self.topo_nd_list]
            else:
                row = [nd.state_names[nd_to_int_st[nd]]
                        for nd in self.topo_nd_list]
            wr_sts.writerows([row])
            if self.is_quantum:
                row = [str(nd_to_degs[nd])
                       for nd in self.topo_nd_list]
                wr_degs.writerows([row])
        fi_sts.close()
        if self.is_quantum:
            fi_degs.close()


if __name__ == "__main__":

    from examples_cbnets.WetGrass import *
    from examples_qbnets.QuWetGrass import *
    num_samples = 2000
    do_int_df = True

    is_quantum = False
    bnet = WetGrass.build_bnet()
    gen = RandGen_NetParams(is_quantum, bnet, num_samples, do_int_df)
    gen.write_csv('training_data_c/wetgrass.csv')

    is_quantum = True
    bnet = QuWetGrass.build_bnet()
    gen = RandGen_NetParams(is_quantum, bnet, num_samples, do_int_df)
    gen.write_csv('training_data_q/wetgrass_sts.csv',
                  'training_data_q/wetgrass_degs.csv')

    is_quantum = False
    b_net = BayesNet.read_bif(
        '../examples_cbnets/earthquake.bif', is_quantum)
    num_samples = 5000
    gen = RandGen_NetParams(is_quantum, b_net, num_samples)
    csv_path = 'training_data_c/earthquake.csv'
    gen.write_csv(csv_path)

