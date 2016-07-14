import pandas as pd
import numpy as np
from learning.DataBinner import *
from learning.NetLner import *
from potentials.Potential import *
from potentials.DiscreteCondPot import *


class NetParamsLner(NetLner):
    """
    NetParamsLner (Net parameters Learner) is a subclass of NetLner. All net
    parameter learner classes have this class as parent. This class learns
    the parameters (i.e., the pots) of a bnet (either a cbnet or qbnet)
    given a dag (directed acyclic graph) structure.

    The input data from which the parameters are learned consists of one
    dataframe states_df in the classical case, and two dataframes states_df
    and degs_df in the quantum case. Both dataframes have the same column
    labels, one column for each node of the bnet whose pots are being learned.

    Each row of degs_df gives an angle Ang in degrees (a float) for the
    column node C for that sample. If z = A( C | parents(C) ) is the
    amplitude for node C, then z = |z| exp(i Ang*pi/180).

    In order to use degs_df, its degree entries (floats) are first binned,
    and then the frequencies of those bins yield a probability for each bin.
    Each bin is then mapped to the mean value of the angles that went into
    that bin. No a priori info is used in this class so its approach is
    purely frequentist.

    In the quantum case, states_df must be measured by an incoherent
    measurement of all the nodes, whereas degs_df must be measured
    by a coherent measurement of node bunches, one bunch for each node and
    its parents.

    Attributes
    ----------

    is_quantum : bool
        True for quantum bnets amd False for classical bnets
    dag : Dag
        a Dag (Directed Acyclic Graph) into which we load what is learned
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.
    ord_nodes : list[DirectedNode]
        a list of DirectedNode's named and in the same order as the column
        labels of self.states_df.

    degs_df : pandas.DataFrame
        Only used in the quantum case. None in classical case. A Pandas
        DataFrame with training data. column=node and row=sample. Each
        row/sample gives an angle Ang in degrees for the column/node.
    do_qtls : bool
        If True (False, resp.), will use quantiles (equal size bins,
        resp) to bin ALL columns of degs_df.
    nd_to_num_deg_bins: dict[DirectedNode, int]
        node to number of degree bins. Use this to specify how many bins you
        want to use for each node when binning degs_df.

    """

    def __init__(self, is_quantum, bnet, states_df, degs_df=None,
            nd_to_num_deg_bins=None, do_qtls=True):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        bnet : BayesNet
        states_df : pandas.DataFrame
        degs_df : pandas.DataFrame
        nd_to_num_deg_bins : dict[DirectedNode, int]
        do_qtls : bool

        Returns
        -------

        """
        NetLner.__init__(self, is_quantum, states_df, bnet)
        nd_names = states_df.columns
        self.ord_nodes = [bnet.get_node_named(name) for name in nd_names]

        self.degs_df = degs_df
        if nd_to_num_deg_bins:
            self.nd_to_num_deg_bins = nd_to_num_deg_bins
        else:
            self.nd_to_num_deg_bins = {nd: 2 for nd in bnet.nodes}
        self.do_qtls = do_qtls

    @staticmethod
    def learn_slice(is_quantum, states_cols, degs_col=None,
                    num_deg_bins=2, do_qtls=True):
        """
        Returns a dataframe called pot_df in which the rows are labelled by
        a multi-index giving the state of each node of the net. pot_df has a
        single column with either complex probability amplitudes A for the
        quantum case or probabilities |A|^2 for the classical case

        Parameters
        ----------
        is_quantum : bool
            True for quantum bnets, False for classical bnets. In the
            classical case, degs_col will be ignored.
        states_cols : pandas.DataFrame
            usually a dataframe containing a subset of the columns of
            self.states_df
        degs_col : pandas.DataFrame
            usually a dataframe or series containing a single column of
            self.degs_df
        num_deg_bins : int
            how many bins will be used to bin degs_col
        do_qtls : bool
            If True, will use quantile bins when binning degs_col. If False,
            will use equal length bins.

        Returns
        -------
        pandas.DataFrame

        """
        if is_quantum:
            # make sure degs_df is a one column dataframe, not a series
            if isinstance(degs_col, pd.Series):
                degs_col = degs_col.to_frame()
            degs_col.columns = ['degs_new_col']
            bin_edges, bin_to_mean = DataBinner.bin_col(
                degs_col, 'degs_new_col', num_deg_bins, do_qtls)
            # print('---states_df', states_df)
            # states_df.loc[:, 'degs_new_col'] = degs_col
            # this works despite warning
            states_cols['degs_new_col'] = degs_col
            # print('---states_df', states_df)

            # doesn't work unless cast to list
            groups = states_cols.groupby(list(states_cols.columns))
            # for key, gp in groups:
            #     print('key=\n', key)
            #     print('gp=\n', gp)

            amp_df = groups.size()
            col_sum = amp_df.sum()
            # print('col_sum', col_sum)
            # print('amp_df\n', amp_df)
            amp_df = amp_df.apply(lambda x: np.sqrt(x/col_sum))
            # print('amp_df\n', amp_df)

            ph_df = groups['degs_new_col'].max()
            # print('ph_df\n', ph_df)
            ph_df = ph_df.apply(lambda x: np.exp(1j*bin_to_mean[x]*np.pi/180))
            # print('ph_df\n', ph_df)

            pot_df = amp_df*ph_df
            del states_cols['degs_new_col']

        else:
            groups = states_cols.groupby(list(states_cols.columns))
            pot_df = groups.size()
            col_sum = pot_df.sum()
            pot_df = pot_df.apply(lambda x: x/col_sum)
        return pot_df

    def learn_one_pot(self, ord_nodes, num_deg_bins):
        """
        Learns from the data in states_df and degs_df a single Potential
        called pot. The nodes in pot are ordered according to the list of
        ordered nodes ord_nodes. pot is a DiscreteCondPot and it is
        normalized assuming as usual that the last node in ord_nodes,
        ord_nodes[-1], is the focus node. In the quantum case, the degs_col
        used is given by the column of the focus node in self.degs_df.

        Parameters
        ----------
        ord_nodes : list[BayesNode]
            list of ordered nodes used in the potential that is returned
        num_deg_bins : int
            number of bins used in binning the degs_col, which is the column
            in self.degs_df corresponding to the focus node = ord_nodes[-1].

        Returns
        -------
        Potential

        """

        col_names = [x.name for x in ord_nodes]
        if self.is_quantum:
            degs_df = self.degs_df[col_names[-1]]
        else:
            degs_df = None
        # print(self.states_df[col_names])
        pot_df = self.learn_slice(self.is_quantum,
                                self.states_df[col_names],
                                degs_df,
                                num_deg_bins,
                                self.do_qtls)
        # this sets multi-index to column headers
        pot_df = pot_df.reset_index(name='pot_values')
        # print(pot_df)

        pot = DiscreteCondPot(self.is_quantum, ord_nodes, bias=0)
        for row in range(len(pot_df.index)):
            index = [
                ord_nodes[col].st_name_index(str(pot_df.iloc[row, col]))
                for col in range(len(ord_nodes))
                ]
            # print(index, [pot_df.iloc[row, col]
            #                              for col in range(len(ord_nodes))
            #                              ])
            pot.pot_arr[tuple(index)] = pot_df.loc[row, 'pot_values']
        pot.normalize_self()
        # print(pot)

        return pot

    def learn_all_bnet_pots(self):
        """
        Learns all the pots of a bnet given a structure (dag) and empirical
        data in the form of dataframes states_df and degs_df.

        Returns
        -------
        BayesNet

        """
        for nd in self.dag.nodes:
            ord_nodes = list(nd.parents) + [nd]
            pot = self.learn_one_pot(ord_nodes, self.nd_to_num_deg_bins[nd])
            # print(nd.name, type(nd))
            # print('pot', pot, type(pot))
            nd.set_potential(pot)

        return self.dag

if __name__ == "__main__":
    print('first test learn slice ---------------')
    df = pd.DataFrame({
        'A': [3, 1, 3, 1, 1, 4, 5, 6],
        'B': [1, 4, 1, 4, 4, 7, 8, 3],
        'C': [2, 3, 2, 3, 3, 5, 6, 7],
        'D': [3, 1, 3, 1, 1, 2, 9, 5],
        'E': [5, 2, 5, 2, 2, 9, 1, 3]
    })
    print('input df=\n', df)

    degs_df = df['B']*10 + .7
    print('input degs_df=\n', degs_df)

    for is_quantum in [False, True]:
        print('\nis_quantum=', is_quantum)
        pot_df = NetParamsLner.learn_slice(
                is_quantum, df, degs_df, num_deg_bins=3)
        print('pot_df=\n', pot_df)
        if is_quantum:
            print('should be one=',
                  pot_df.to_frame().apply(lambda x:
                                          x*np.conjugate(x)).sum()[0])

        else:
            print('should be one=', pot_df.sum())

    from examples_cbnets.WetGrass import *
    from examples_qbnets.QuWetGrass import *

    print('next test learn bnet ---------------')
    for is_quantum in [False, True]:
        print('------is_quantum=', is_quantum)
        if is_quantum:
            bnet = QuWetGrass.build_bnet()
            bnet_emp = QuWetGrass.build_bnet()
            states_df = pd.read_csv(
                'training_data_q\\wetgrass_sts.csv', dtype=str)
            degs_df = pd.read_csv(
                'training_data_q\\wetgrass_degs.csv', dtype=str)
        else:
            bnet = WetGrass.build_bnet()
            bnet_emp = WetGrass.build_bnet()
            states_df = pd.read_csv(
                'training_data_c\\wetgrass.csv', dtype=str)
            degs_df = None

        # forget pots of emp=empirical bnet because we want to learn them
        for nd in bnet_emp.nodes:
            nd.potential = None

        # print('states_df=\n', states_df)
        # if is_quantum:
        #     print('degs_df=\n', degs_df)

        lnr = NetParamsLner(is_quantum, bnet_emp, states_df, degs_df)
        lnr.learn_all_bnet_pots()
        for nd in bnet.nodes:
            print('\nnode=', nd.name)
            print('true:')
            print(nd.potential)
            print('empirical:')
            print(bnet_emp.get_node_named(nd.name).potential)
