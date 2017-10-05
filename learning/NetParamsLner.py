import pandas as pd
import numpy as np
from learning.DataBinner import *
from learning.NetStrucLner import *
from potentials.Potential import *
from potentials.DiscreteCondPot import *


class NetParamsLner:
    """
    NetParamsLner (Net Parameters Learner) learns the parameters ( i.e.,
    the pots) of a bnet (either a cbnet or qbnet) given the bnet's structure.

    The input data from which the parameters are learned consists of one
    dataframe states_df in the classical case, and two dataframes states_df
    and degs_df in the quantum case. Both dataframes have the same column
    labels, one column for each node of the bnet whose pots are being learned.

    Each row of degs_df gives an angle Ang in degrees (a float) for the
    column node C for that sample. If z = A( C | parents(C) ) is the
    amplitude for node C, then z = |z| exp(i Ang*pi/180).

    In order to use degs_df, its degree entries (floats) are first binned.
    Each bin is then mapped to the mean value of the angles that went into
    that bin. No a priori info is used in this class so its approach is
    purely frequentist.

    In the quantum case, states_df must be measured by an incoherent
    measurement of all the nodes, whereas degs_df must be measured
    by a coherent measurement of node bunches, one bunch for each node and
    its parents.

    IMPORTANT: We will use the word 'vtx' = vertex to denote a node name and
    the word 'node' to denote a Node object.

    See constructor of class NetStrucLner. All structure learners will
    either (a) learn the state names from states_df or (b) import them if a
    vtx_to_states is submitted as input to the constructor.

    This class assumes that its attribute self.bnet knows the net's (1)
    structure and (2) state names. If self.bnet knows (1) but not (2) as
    would happen if self.bnet comes from reading a dot file, then before
    using this class its self.bnet should  learn (2) via
    bnet.import_nd_state_names() or bnet.learn_nd_state_names().

    Attributes
    ----------

    bnet : BayesNet
        a BayesNet in which we store what is learned
    degs_df : pandas.DataFrame
        Only used in the quantum case. None in classical case. A Pandas
        DataFrame with training data. column=node and row=sample. Each
        row/sample gives an angle Ang in degrees for the column/node.
    do_qtls : bool
        If True (False, resp.), will use quantiles (equal size bins,
        resp) to bin ALL columns of degs_df.
    is_quantum : bool
        True for quantum bnets amd False for classical bnets
    nd_to_num_deg_bins: dict[DirectedNode, int]
        node to number of degree bins. Use this to specify how many bins you
        want to use for each node when binning degs_df.
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.
    use_int_sts : bool
        If False, the states_df has state names as entries. If True,
        states_df has int entries. The int entries are the index in the
        states_names list of the node for that column.

    """

    def __init__(self, is_quantum, bnet, states_df, 
            degs_df=None, nd_to_num_deg_bins=None, do_qtls=True):
        """
        Constructor

        This constructor assumes that the parameter bnet is a BayesNet
        which already contains the correct net structure and the desired
        state names for each node.

        If bnet does not contain the desired state names for each node,
        you will have to do some pre-processing of bnet before you pass it
        in to this constructor. The two static methods
        NetStrucLner:learn_nd_state_names() and
        NetStrucLner:import_nd_state_names() can be used for this purpose.

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
        self.is_quantum = is_quantum
        self.bnet = bnet
        self.states_df = states_df
        self.use_int_sts = NetParamsLner.int_sts_detector(states_df)

        self.degs_df = degs_df
        if nd_to_num_deg_bins:
            self.nd_to_num_deg_bins = nd_to_num_deg_bins
        else:
            self.nd_to_num_deg_bins = {nd: 2 for nd in bnet.nodes}
        self.do_qtls = do_qtls

    @staticmethod
    def int_sts_detector(sub_states_df):
        """
        This function returns True iff the first row of sub_states_df has
        only int entries. We will assume that if the first row does,
        then all rows do.

        Parameters
        ----------
        sub_states_df : pandas.DataFrame

        Returns
        -------
        bool

        """
        # print('inside detector\n', sub_states_df.head())
        for k in range(len(sub_states_df.columns)):
            if not str(sub_states_df.iloc[0, k]).isdigit():
                # print('returns false')
                return False
        # print('returns true')
        return True

    @staticmethod
    def learn_pot_df(is_quantum, states_cols, degs_col=None,
                     num_deg_bins=2, do_qtls=True):
        """
        Returns a dataframe called pot_df containing one more column (the
        last one) than states_cols. That column contains either complex
        probability amplitudes A for the quantum case or probabilities |A|^2
        for the classical case. The non-last columns of pot_df are a list of
        distinct states of the nodes whose names are given by the column
        labels of pot_df.

        Besides returning a pot_df, this function also returns two
        s_d_pair's, max_s_d_pair and min_s_d_pair. An s_d_pair is a tuple of
        a state entry from states_df and a degs. The state is a state of the
        focus node (the last column of states_cols corresponds to the focus
        node). The degs is an angle in degrees. max_s_d_pair (resp.,
        min_s_d_pair) is a pair with maximum (resp., minimum) likelihood.

        Parameters
        ----------
        is_quantum : bool
            True for quantum bnets amd False for classical bnets. In the
            classical case, degs_col will be ignored.
        states_cols : pandas.DataFrame
            usually a dataframe containing a subset of the columns of
            self.states_df
        degs_col : pandas.DataFrame
            usually a dataframe or series containing a single column of
            self.degs_df
        num_deg_bins : int
            how many bins will be used to bin degs_col.
        do_qtls : bool
            If True, will use quantile bins when binning degs_col. If False,
            will use equal length bins.


        Returns
        -------
        pandas.DataFrame, tuple[int|str, float], tuple[int|str, float]

        """
        focus_vtx = states_cols.columns[-1]

        if is_quantum:
            # make sure degs_df is a one column dataframe, not a series
            if isinstance(degs_col, pd.Series):
                degs_col = degs_col.to_frame()
            degs_col.columns = ['degs_as_bins']
            bin_edges, bin_to_mean = DataBinner.bin_col(
                degs_col, 'degs_as_bins', num_deg_bins, do_qtls)
            # print('bin_edges, bin_to_mean', bin_edges, bin_to_mean)
            # print('---states_cols\n', states_cols)
            # states_df.loc[:, 'degs_as_bins'] = degs_col
            # this works despite warning
            states_cols['degs_as_bins'] = degs_col
            # print('---states_cols\n', states_cols)

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

            ph_df = groups['degs_as_bins'].max()
            # print('ph_df\n', ph_df)
            ph_df = ph_df.apply(lambda x: np.exp(1j*bin_to_mean[x]*np.pi/180))
            # print('ph_df\n', ph_df)

            pot_df = amp_df*ph_df
            del states_cols['degs_as_bins']
            
            # this sets multi-index to column headers           
            amp_df = amp_df.reset_index(name='last_col_with_vals')
            ph_df = ph_df.reset_index(name='last_col_with_vals')
            pot_df = pot_df.reset_index(name='last_col_with_vals')
            del pot_df['degs_as_bins']
            # print('ph_df\n', pot_df)
            
            max_idx = amp_df['last_col_with_vals'].idxmax()
            max_st = amp_df.loc[max_idx, focus_vtx]
            max_degs = bin_to_mean[ph_df.loc[max_idx, 'degs_as_bins']]
            max_s_d_pair = (max_st, max_degs)

            min_idx = amp_df['last_col_with_vals'].idxmin()
            min_st = amp_df.loc[min_idx, focus_vtx]
            min_degs = bin_to_mean[ph_df.loc[min_idx, 'degs_as_bins']]
            min_s_d_pair = (min_st, min_degs)

        else:
            groups = states_cols.groupby(list(states_cols.columns))
            pot_df = groups.size()
            col_sum = pot_df.sum()
            pot_df = pot_df.apply(lambda x: x/col_sum)
            
            pot_df = pot_df.reset_index(name='last_col_with_vals')
            
            max_idx = pot_df['last_col_with_vals'].idxmax()
            max_st = pot_df.loc[max_idx, focus_vtx]
            max_s_d_pair = (max_st, 0)
            
            min_idx = pot_df['last_col_with_vals'].idxmin()
            min_st = pot_df.loc[min_idx, focus_vtx]
            min_s_d_pair = (min_st, 0)

        return pot_df, max_s_d_pair, min_s_d_pair

    @staticmethod
    def convert_pot_df_to_pot(is_quantum, pot_df, ord_nodes,
                s_d_pair, use_int_sts=None, normalize=True):
        """
        Returns a DiscreteCondPot pot with ordered nodes ord_nodes.

        Parameters
        ----------
        is_quantum : bool
        pot_df : pandas.DataFrame
            a dataframe returned by learn_pot_df(). pot_df must have one
            column for each node in ord_nodes plus additioanl final column
            with pot values
        ord_nodes : list[BayesNode]
            ordered nodes of Potential pot that is returned. Nodes must be
            ordered so that pot_df[:-1].columns = [nd.name for nd in
            ord_nodes]
        s_d_pair : tuple[int|str, float]
            a pair consisting of a state and a degs. The state is taken
            directly from states_df, so it might be a digit or an actual
            state name, depending on whether use_int_sts was True or False
            when states_df was generated. The state is a state of the focus
            node (the last column of states_cols corresponds to the focus
            node). The degs is an angle in degrees. For an s_d_pair equal to
            (x0,ang), whenever pot(C=x| pa(C)=y) = 0 for all x, we will set
            pot(C=x|pa(C)=y) = exp( 1j*ang*pi/180)*delta(x, x0) in the
            quantum case and pot( C=x| pa(C)=y) = delta(x, x0) in the
            classical case.
        use_int_sts : bool
            True if states in states_df are integers, False if they are the
            actual state names. If the parameter use_int_sts is set to None,
            this function will attempt to find its bool value using the
            function NetStrucLner:int_sts_detector()
        normalize : bool
            True if want pot to be normalized as a cond probability
            or cond amplitude, P(x|y) or A(x|y), where x is ord_nodes[-1]

        Returns
        -------
        DiscreteCondPot

        """
        for k, vtx in enumerate(pot_df.columns[:-1]):
                assert ord_nodes[k].name == vtx
        focus_vtx = pot_df.columns[-2]
        focus_nd = ord_nodes[-1]
        if use_int_sts is None:
            use_int_sts = NetParamsLner.int_sts_detector(pot_df[:-1])

        def int_state(st):
            if use_int_sts:
                return int(st)
            else:
                return focus_nd.pos_of_st_name(str(st))

        pot = DiscreteCondPot(is_quantum, ord_nodes, bias=0)
        for row in range(len(pot_df.index)):
            # print([pot_df.iloc[row, col]
            #                              for col in range(len(ord_nodes))
            #                              ])
            if not use_int_sts:
                arr_index = [
                    ord_nodes[col].pos_of_st_name(str(pot_df.iloc[row, col]))
                    for col in range(len(ord_nodes))]
                
            else:
                arr_index = [int(pot_df.iloc[row, col])
                            for col in range(len(ord_nodes))]
            pot.pot_arr[tuple(arr_index)] = \
                    pot_df.loc[row, 'last_col_with_vals']

        if normalize:
            try:
                pot.normalize_self()
            except UnNormalizablePot as xce:
                # print('caught exception')
                focus_index = int_state(s_d_pair[0])
                arr_index = tuple(list(xce.pa_indices) + [focus_index])
                # print('********************arr_index', arr_index)
                if is_quantum:
                    pot.pot_arr[arr_index] = \
                        np.exp(1j * s_d_pair[1] * np.pi / 180)
                else:
                    pot.pot_arr[arr_index] = 1.0
                print('mended pot:\n', pot)
                # try normalizing pot again now that it is mended
                pot.normalize_self()
        return pot

    def learn_pot(self, ord_nodes):
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

        Returns
        -------
        Potential

        """

        ord_nd_names = [x.name for x in ord_nodes]
        focus_vtx = ord_nd_names[-1]
        focus_nd = ord_nodes[-1]
        if self.is_quantum:
            degs_col = self.degs_df[focus_vtx]
            num_deg_bins = self.nd_to_num_deg_bins[focus_nd]
        else:
            degs_col = None
            num_deg_bins = 2
        # print(self.states_df[col_names])
        pot_df, max_s_d_pair, min_s_d_pair = \
            NetParamsLner.learn_pot_df(
                self.is_quantum,
                self.states_df[ord_nd_names],
                degs_col=degs_col,
                num_deg_bins=num_deg_bins,
                do_qtls=self.do_qtls)

        pot = NetParamsLner.convert_pot_df_to_pot(
                self.is_quantum,
                pot_df,
                ord_nodes,
                max_s_d_pair,
                self.use_int_sts,
                normalize=True)
        return pot

    def learn_all_bnet_pots(self):
        """
        Learns all the pots of a bnet given its structure and given
        empirical data in the form of dataframes states_df and degs_df.

        Returns
        -------
        BayesNet

        """
        for nd in self.bnet.nodes:
            ord_nodes = list(nd.parents) + [nd]
            pot = self.learn_pot(ord_nodes)
            # print(nd.name, type(nd))
            # print('pot', pot, type(pot))
            nd.set_potential(pot)

        return self.bnet

    @staticmethod
    def compare_true_and_emp_pots(bnet, bnet_emp):
        """
        Prints a comparison of the potentials of true and empirical bnets

        Parameters
        ----------
        bnet : BayesNet
            true BayesNet
        bnet_emp : BayesNet
            empirical BayesNet

        Returns
        -------

        """
        for nd in bnet.nodes:
            print('\nnode=', nd.name)
            true_pot = nd.potential
            # must permute ord_nodes of emp_pot so that they are
            # in same order as those of true_pot or else they
            # may not be
            emp_pot = bnet_emp.get_node_named(nd.name).potential
            new_ord_nds = [bnet_emp.get_node_named(nd.name) for nd
                                   in true_pot.ord_nodes]
            emp_pot.set_to_transpose(new_ord_nds)
            print('true:')
            print(true_pot)
            print('empirical:')
            print(emp_pot)

if __name__ == "__main__":
    print('first test learn pot_df ---------------')
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
        pot_df, max_s_d_pair, min_s_d_pair = NetParamsLner.learn_pot_df(
                is_quantum, df, degs_col=degs_df, num_deg_bins=3)
        print('pot_df=\n', pot_df)
        print('max_s_d_pair', max_s_d_pair)
        print('min_s_d_pair', min_s_d_pair)
        if is_quantum:
            print('should be one=',
                  pot_df['last_col_with_vals'].apply(lambda x:
                                          x*np.conjugate(x)).sum())

        else:
            print('should be one=', pot_df['last_col_with_vals'].sum())

    from examples_cbnets.WetGrass import *
    from examples_qbnets.QuWetGrass import *

    print('\nnext test learn bnet, wetgrass---------------')
    for is_quantum in [False, True]:
        print('------is_quantum=', is_quantum)
        if is_quantum:
            bnet = QuWetGrass.build_bnet()
            bnet_emp = QuWetGrass.build_bnet()
            states_df = pd.read_csv(
                'training_data_q/WetGrass_sts.csv', dtype=str)
            degs_df = pd.read_csv(
                'training_data_q/WetGrass_degs.csv', dtype=str)
        else:
            bnet = WetGrass.build_bnet()
            bnet_emp = WetGrass.build_bnet()
            states_df = pd.read_csv(
                'training_data_c/WetGrass.csv', dtype=str)
            degs_df = None

        # forget pots of emp=empirical bnet because we want to learn them
        for nd in bnet_emp.nodes:
            nd.potential = None

        # print('states_df=\n', states_df)
        # if is_quantum:
        #     print('degs_df=\n', degs_df)

        bnet_emp.learn_nd_state_names(states_df)
        lnr = NetParamsLner(is_quantum, bnet_emp, states_df, degs_df)
        lnr.learn_all_bnet_pots()
        lnr.compare_true_and_emp_pots(bnet, bnet_emp)

    print('\nnext test learn bnet, earthquake ---------------')
    is_quantum = False
    bnet = BayesNet.read_bif(
        '../examples_cbnets/earthquake.bif', is_quantum)

    # for bnet_emp, read a dot file (no a priori pots) instead of a bif file
    bnet_emp = BayesNet.read_dot('../examples_cbnets/earthquake.dot')
    vtx_to_states = bnet.get_vtx_to_state_names()
    bnet_emp.import_nd_state_names(vtx_to_states)

    states_df = pd.read_csv(
        'training_data_c/earthquake.csv', dtype=str)

    lnr = NetParamsLner(is_quantum, bnet_emp, states_df)
    lnr.learn_all_bnet_pots()
    lnr.compare_true_and_emp_pots(bnet, bnet_emp)

    # The above example didn't match correctly the entries of
    # the true and empirical pots because the state names are
    # alphabetically ordered in bnet_emp but they aren't in
    # bnet. This time we input the state names into bnet_emp
    # instead of learning them from states_df

    print('\nnext test learn bnet, earthquake2 ---------------')

    vtx_to_states = bnet.get_vtx_to_state_names()
    bnet_emp.import_nd_state_names(vtx_to_states)
    lnr = NetParamsLner(is_quantum, bnet_emp, states_df)
    lnr.learn_all_bnet_pots()
    lnr.compare_true_and_emp_pots(bnet, bnet_emp)

    # Sometimes the earthquake training data does not contain
    # samples for Burglary=False, Earthquake=False, Alarm=True or False.
    # because P(A=True|E=False, B=False)=.001.
    # In such cases, one gets an UnNormalizablePot exception.
    # Generating 5000 or more samples usually avoids this exception.

