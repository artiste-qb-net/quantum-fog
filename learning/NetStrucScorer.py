import scipy.special as sp

from learning.NetParamsLner import *
from shannon_info_theory.DataEntropy import *


class NetStrucScorer:
    """
    NetStrucScorer (Net Structure Scorer) is a class that can be used to
    give a score (i.e., measuring agreement with the data in states_df),
    to a graph structure (the plain graph, not the pots aka parameters)
    using various scoring functions, both frequentist and Bayesian. But
    NetStrucScorer does more than score individual graphs. It also keeps a
    running record of the scores of the moves performed by objects of class
    HillClimbingLner and its subclasses.

    frequentist scoring functions (from info theory):
    "LL" log likelihood
    "BIC" Bayesian Information Criterion
    "AIC" Akaike Information Criterion

    Bayesian scoring functions:
    "BDEU"
    "K2"

    IMPORTANT: We will use the word 'vtx' = vertex to denote a node name and
    the word 'node' to denote a Node object.

    References
    ----------
    1. "Scoring functions for learning Bayesian networks", by
    Alexandra M. Carvalho (powerpoint presentation on web)

    Attributes
    ----------
    ess : float
        Equivalent Sample Size, a parameter in BDEU scorer. Fudge factor
        that is supposed to grow as the amount of prior knowledge grows.
    score_type : str
        score type, either 'LL', 'BIC, 'AIC', 'BDEU' or 'K2'
    size_penalty_fun : function
        This is the name of a size penalty function (either
        BIC_size_penalty_fun or AIC_size_penalty_fun without quotes). Some
        of the scoring methods specified by the score_type parameter deduct
        some points to the sum of scores of all vertices to penalize the use
        of too many parameters for the given sample size (overfitting).
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.
    tot_score : float
        The total score for the whole graph. It equals the sum over all
        vertices of each vertex's score, minus, for some scoring methods,
        a size penalty.
    vertices : list[str]
        a list of the vertices of the graph. Equal to self.states_df.columns.
    vtx_score_fun : function
        the name of a function that takes as input a vertex and returns a
        score for it.
    vtx_to_parents : dict[str, list[str]]
        a dictionary that maps each vertex to a list of its parents.
    vtx_to_score : dict[str, float]
        a dictionary that maps each vertex to its score
    vtx_to_size : dict[str, int]
        a dictionary that maps each vertex to its number of states (aka its
        size)
    vtx_to_states : dict[str, list[str]]
        A dictionary mapping each node name to a list of its state names.
        This information will be stored in self.bnet. If
        vtx_to_states=None, constructor will learn vtx_to_states
        from states_df

    """

    def __init__(self, states_df, vtx_to_parents,
                 vtx_to_states, score_type, ess=1):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.core.frame.DataFrame
        vtx_to_parents : dict[str, list[str]]
        vtx_to_states : dict[str, list[str]]
        score_type : str
        ess : float

        Returns
        -------
        None

        """
        self.states_df = states_df
        self.vertices = states_df.columns

        self.vtx_to_parents = vtx_to_parents
        self.vtx_to_states = vtx_to_states
        self.vtx_to_size = {vtx: len(vtx_to_states[vtx])
                            for vtx in self.vertices}
        self.score_type = score_type
        self.ess = ess  # BDEU equivalent sample size

        ty = score_type
        size_penalty_fun = None
        # frequentist scores (from information theory)
        if ty == 'LL':
            vtx_score_fun = self.LL_vtx_score
        elif ty == 'BIC':
            vtx_score_fun = self.BIC_vtx_score
            size_penalty_fun = self.BIC_size_penalty
        elif ty == 'AIC':
            vtx_score_fun = self.AIC_vtx_score
            size_penalty_fun = self.AIC_size_penalty

        # bayesian scores
        elif ty == 'BDEU':
            vtx_score_fun = self.BDEU_vtx_score
        elif ty == 'K2':
            vtx_score_fun = self.K2_vtx_score

        else:
            assert False, 'Unsupported scoring function'

        self.vtx_score_fun = vtx_score_fun
        self.size_penalty_fun = size_penalty_fun
        self.vtx_to_score = {vtx: [] for vtx in self.vertices}
        self.tot_score = None
        self.refresh_scores()

    def get_vtx_num_params(self, vtx, new_parents=None):
        """
        This function returns the number of degrees of freedom of a vertex (
        so it subtracts one from vtx size, and multiplies that by the
        product of the sizes of the parents of vtx). If new_parents=None,
        it assumes the parents of vtx are those in vtx_to_parents[vtx].
        Otherwise, it assumes the parents of vtx are given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        int

        """

        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]
        pa_sizes = [self.vtx_to_size[pa] for pa in new_parents]
        if len(pa_sizes) == 0:  # root nodes
            pa_size_prod = 1
        else:
            pa_size_prod = np.prod(np.array(pa_sizes))
        return (self.vtx_to_size[vtx] - 1)*pa_size_prod

    def get_tot_num_params(self, new_vtx_to_parents=None):
        """
        This function returns the sum of the number of parameters for all
        the vertices. If new_vtx_to_parents[vtx] is nonempty, it uses that
        for the parents of vtx. Otherwise, it uses self.vtx_to_parents[vtx]
        for its parents.


        Parameters
        ----------
        new_vtx_to_parents : dict[str, list[str]]

        Returns
        -------
        int

        """
        tot = 0
        for v in self.vertices:
            if new_vtx_to_parents is not None and new_vtx_to_parents[v]:
                tot += self.get_vtx_num_params(v, new_vtx_to_parents[v])
            else:
                tot += self.get_vtx_num_params(v)
        return tot

    def refresh_scores(self):
        """
        This function recalculates from scratch vtx_to_score[vtx] for all
        vtx. It also calculates self.tot_score for the graph.

        Returns
        -------
        None

        """
        tot = 0
        for vtx in self.vertices:
            score = self.vtx_score_fun(vtx)
            self.vtx_to_score[vtx] = score
            tot += score
        if self.size_penalty_fun is not None:
            tot += self.size_penalty_fun(self.get_tot_num_params())
        self.tot_score = tot

    def get_vtx_score_ch(self, vtx, new_parents, for_end_vtx):
        """
        Returns score change for either beg_vtx or end_vtx

        Parameters
        ----------
        new_parents : list[str]
        vtx : str
            either beg_vtx or edn_vtx
        for_end_vtx : bool
            True if end vtx score is changing. False if beg vtx score is
            changing.

        Returns
        -------
        list[str, str, str]

        """
        score_ch = [0, 0, 0]
        # score_ch = beg_vtx_score_ch, end_vtx_score_ch, tot_score_ch
        if for_end_vtx:
            score_ch[1] += self.vtx_score_fun(vtx, new_parents) -\
                    self.vtx_to_score[vtx]
            score_ch[2] += score_ch[1]
        else:  # for beg vtx
            score_ch[0] += self.vtx_score_fun(vtx, new_parents) -\
                    self.vtx_to_score[vtx]
            score_ch[2] += score_ch[0]
        return score_ch

    def score_move(self, move):
        """
        This function takes as input a move = (beg_vtx, end_vtx, action) and
        returns the move's score_change = (beg_vtx_score_ch,
        end_vtx_score_ch, tot_score_ch ) where beg=beginning, vtx=vertex,
        tot=total, ch=change


        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        tuple[float, float, float]

        """
        (beg_vtx, end_vtx, action) = move

        new_vtx_to_parents = {vtx: [] for vtx in self.vertices}
        if action == 'add':
            # score change when end_vtx gains parent
            new_parents = self.vtx_to_parents[end_vtx] + [beg_vtx]
            score_ch = self.get_vtx_score_ch(
                    end_vtx, new_parents, for_end_vtx=True)
            new_vtx_to_parents[end_vtx] = new_parents

        elif action == 'del':
            # score change when end_vtx loses parent
            new_parents = list(set(self.vtx_to_parents[end_vtx])-{beg_vtx})
            score_ch = self.get_vtx_score_ch(
                    end_vtx, new_parents, for_end_vtx=True)
            new_vtx_to_parents[end_vtx] = new_parents

        elif action == 'rev':
            # score change when end_vtx loses parent
            new_parents = list(set(self.vtx_to_parents[end_vtx]) - {beg_vtx})
            score_ch1 = self.get_vtx_score_ch(
                    end_vtx, new_parents, for_end_vtx=True)
            new_vtx_to_parents[end_vtx] = new_parents

            # score change when beg_vtx gains parent
            new_parents = self.vtx_to_parents[beg_vtx] + [end_vtx]
            score_ch2 = self.get_vtx_score_ch(
                    beg_vtx, new_parents, for_end_vtx=False)
            new_vtx_to_parents[beg_vtx] = new_parents

            score_ch = [score_ch1[k] + score_ch2[k] for k in range(3)]
        else:
            assert False

        if self.size_penalty_fun is not None:
            init_num_params = self.get_tot_num_params()
            new_num_params = self.get_tot_num_params(new_vtx_to_parents)
            score_ch[2] += self.size_penalty_fun(
                    new_num_params - init_num_params)

        return score_ch

    def do_move(self, move, score_change):
        """
        This function takes as input a move and its score change and changes
        all attributes of the class object to realize that move. Only one
        move of a try makes it to this function.

        Parameters
        ----------
        move : tuple[str, str, str]
        score_change : tuple[float, float, float]

        Returns
        -------
        None

        """
        (beg_vtx, end_vtx, action) = move
        (beg_vtx_score_ch, end_vtx_score_ch, tot_score_ch) = score_change
        # this is not necessary because vtx_to_parents same
        # for lner and scorer
        # if action == 'add':
        #     self.vtx_to_parents[end_vtx].append(beg_vtx)
        # elif action == 'del':
        #     self.vtx_to_parents[end_vtx].remove(beg_vtx)
        # elif action == 'rev':
        #     self.vtx_to_parents[end_vtx].remove(beg_vtx)
        #     self.vtx_to_parents[beg_vtx].append(end_vtx)
        # else:
        #     assert False
        self.tot_score += tot_score_ch
        self.vtx_to_score[beg_vtx] += beg_vtx_score_ch
        self.vtx_to_score[end_vtx] += end_vtx_score_ch

    def LL_vtx_score(self, vtx, new_parents=None):
        """
        LL = Log Likelihood. This function returns the LL score for vtx. If
        new_parents=None, it assumes the parents of vtx are those in
        vtx_to_parents[vtx]. Otherwise, it assumes the parents of vtx are
        given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        float

        """

        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]

        sam_size = len(self.states_df.index)

        return -sam_size*DataEntropy.cond_info(
            self.states_df, [vtx], new_parents)

    def BIC_vtx_score(self, vtx, new_parents=None):
        """
        BIC = Bayesian Information Criterion. This function returns the BIC
        score for vtx. If new_parents=None, it assumes the parents of vtx
        are those in vtx_to_parents[vtx]. Otherwise, it assumes the parents
        of vtx are given by new_parents.


        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        float

        """

        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]
        return self.LL_vtx_score(vtx, new_parents)

    def BIC_size_penalty(self, num_params):
        """
        This function returns the size penalty for BIC.

        Parameters
        ----------
        num_params : int

        Returns
        -------
        float

        """
        sam_size = len(self.states_df.index)
        return -0.5*np.log(sam_size)*num_params

    def AIC_vtx_score(self, vtx, new_parents=None):
        """
        AIC = Akaike Information Criterion. This function returns the AIC
        score for vtx. If new_parents=None, it assumes the parents of vtx
        are those in vtx_to_parents[vtx]. Otherwise, it assumes the parents
        of vtx are given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        float

        """

        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]
        return self.LL_vtx_score(vtx, new_parents)

    def AIC_size_penalty(self, num_params):
        """
        This function returns the size penalty for AIC.

        Parameters
        ----------
        num_params : int

        Returns
        -------
        int

        """
        return -num_params

    def get_pot(self, vtx, new_parents=None):
        """
        This function estimates and returns a Potential based on the data in
        the dataframe states_df. If new_parents=None, it assumes the parents
        of vtx are those in vtx_to_parents[vtx]. Otherwise, it assumes the
        parents of vtx are given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        Potential

        """
        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]
        states_cols = self.states_df[new_parents + [vtx]]

        # the following is same as in NetParamsLnr.learn_pot_df
        # but without dividing by col_sum
        groups = states_cols.groupby(list(states_cols.columns))
        pot_df = groups.size()
        # this sets multi-index to column headers
        pot_df = pot_df.reset_index(name='pot_values')

        num_pa = len(new_parents)

        pa_nds = [BayesNode(k, new_parents[k]) for k in range(num_pa)]
        for k in range(num_pa):
            pa_nds[k].state_names = self.vtx_to_states[new_parents[k]]
        focus_nd = BayesNode(num_pa, vtx)
        focus_nd.state_names = self.vtx_to_states[vtx]
        ord_nodes = pa_nds + [focus_nd]

        pot = NetParamsLner.convert_pot_df_to_pot(
            False,
            pot_df,
            ord_nodes,
            s_d_pair=(0, 0),  # won't be used since we are not normalizing
            normalize=False)
        return pot

    @staticmethod
    def BD_family_vtx_score(n_ijk, a_ijk):
        """
        This function returns the value of a special log Gamma function of
        n_ijk, a_ijk that occurs in both (BDEU and K2) bayesian scoring
        functions.

        Parameters
        ----------
        n_ijk : numpy.array
        a_ijk : numpy.array

        Returns
        -------
        float

        """

        n_ij = n_ijk.sum(axis=-1)
        a_ij = a_ijk.sum(axis=-1)
        # print('n_ijk', n_ijk)
        # print('n_ij', n_ij)

        part1 = sp.gammaln(a_ijk + n_ijk) - sp.gammaln(a_ijk)
        part1 = part1.sum()

        part2 = sp.gammaln(a_ij) - sp.gammaln(a_ij + n_ij)
        part2 = part2.sum()
        return part1 + part2

    def BDEU_vtx_score(self, vtx, new_parents=None):
        """
        This function returns the contribution or 'score' of a single vertex
        vtx to log(P(G|D)) according to the BDEU method. If
        new_parents=None, it assumes the parents of vtx are those in
        vtx_to_parents[vtx]. Otherwise, it assumes the parents of vtx are
        given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : int

        Returns
        -------
        float

        """
        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]

        pot = self.get_pot(vtx, new_parents)
        n_ijk = pot.pot_arr
        shape = n_ijk.shape

        # size of a numpy array is prod of shape elements
        fudge_factor = self.ess / n_ijk.size
        a_ijk = np.ones(shape)*fudge_factor
        # print('BDEU fudge_factor (=1 for K2)=', fudge_factor)

        return NetStrucScorer.BD_family_vtx_score(n_ijk, a_ijk)

    def K2_vtx_score(self, vtx, new_parents=None):
        """
        This function returns the contribution or 'score' of a single vertex
        vtx to log(P(G|D)) according to the K2 method. If new_parents=None,
        it assumes the parents of vtx are those in vtx_to_parents[vtx].
        Otherwise, it assumes the parents of vtx are given by new_parents.

        Parameters
        ----------
        vtx : str
        new_parents : list[str]

        Returns
        -------
        float

        """
        if new_parents is None:
            new_parents = self.vtx_to_parents[vtx]

        pot = self.get_pot(vtx, new_parents)
        n_ijk = pot.pot_arr
        shape = n_ijk.shape

        a_ijk = np.ones(shape)

        return NetStrucScorer.BD_family_vtx_score(n_ijk, a_ijk)

if __name__ == "__main__":
    print(5)

