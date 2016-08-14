from learning.HillClimbingLner import *
from learning.MB_MMPC_Lner import *
from learning.HC_TabuLner import *
from learning.HC_RandRestartLner import *


class HC_MMHC_rr_Lner(HC_RandRestartLner):
    """
    The class HC_MMHC_rr_Lner (Hill Climbing Min-Max Hill Climbing Random
    Restart Learner) is a child of HC_RandomRestartLner. It adds to the
    latter a search at the beginning of the learning process of the PC (
    parents children, aka neighbors) set of each node. This knowledge is
    then used in the move_allowed() function to forbid any 'add' moves
    unless they add arrows compatible with the PC list of each node.

    References
    ----------

    1. Tsamardinos I, Brown LE, Aliferis CF (2006). "The Max-Min
    Hill-Climbing Bayesian Network Structure Learning Algorithm". Machine
    Learning, 65(1), 31-78.

    Attributes
    ----------
    is_quantum : bool
        True for quantum bnets amd False for classical bnets
    dag : Dag
        a Dag (Directed Acyclic Graph) in which we store what is learned
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.
    ord_nodes : list[DirectedNode]
        a list of DirectedNode's named and in the same order as the column
        labels of self.states_df.

    max_num_mtries : int
        maximum number of move tries
    nx_graph : networkx.DiGraph
        a networkx directed graph used to store arrows
    score_type : str
        score type, either 'LL', 'BIC, 'AIC', 'BDEU' or 'K2'
    scorer : NetStrucScorer
        object of NetStrucScorer class that keeps a running record of scores
    verbose : bool
        True for this prints a running commentary to console
    vertices : list[str]
        list of vertices (node names). Same as states_df.columns
    vtx_to_parents : dict[str, list[str]]
        dictionary mapping each vertex to a list of its parents's names

    mcache : list[tuple[str, str, str]]
        a list that stores all moves of a try. This list is cleared at
        beginning of each try.
    score_ch_cache : list[tuple[float, float, float]]
        A list that stores the score changes of each move of a try. Score
        changes here are given as a 3-tuple of 3 floats, (beg_score_ch,
        end_score_ch, tot_score_ch).
    best_start_graph : networkx.DiGraph
    best_start_score : float
        Each time restart() function is called, the current tot score is
        compared with best_start_score, and if it is higher, then both the
        current nx_graph and its score are stored in best_start_graph and
        best_start_score.
    cur_start : int
        current start. Increases by one every time restart() is called.
    num_starts : int
        number of starts. Number of times minus 1 that restart function is
        called.
    vtx_to_nbors : dict[str, list[str]]
        a dictionary mapping each vertex to a list of its neighbors. The
        literature also calls the set of neighbors of a vertex its PC (
        parents-children) set.
    alpha : float
        threshold used for deciding whether a conditional or unconditional
        mutual info is said to be close to zero (independence) or not (
        dependence). The error in a data entropy is on the order of ln(n+1)
        - ln(n) \approx 1/n where n is the number of samples so 5/n is a
        good default value for alpha.

    """

    def __init__(self, states_df, score_type, max_num_mtries,
            alpha=0, num_starts=10,
            ess=1.0, verbose=False, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        alpha : float
        states_df : pandas.DataFrame
        score_type : str
        max_num_mtries : int
        num_starts : int
        ess : float
            Equivalent Sample Size, a parameter in BDEU scorer. Fudge factor
            that is supposed to grow as the amount of prior knowledge grows.
        verbose : bool
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.dag. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df


        Returns
        -------
        None

        """
        # this is a good default value
        self.alpha = 5/len(states_df.index)

        lner = MB_MMPC_Lner(states_df, alpha, verbose,
                vtx_to_states, learn_later=True)
        lner.find_PC()
        self.vtx_to_nbors = lner.vtx_to_nbors

        HC_RandRestartLner.__init__(
            self, states_df, score_type, max_num_mtries,
            num_starts,
            ess, verbose, vtx_to_states)

    def move_approved(self, move):
        """
        Returns bool indicating whether move is approved. All 'del' and
        'rev' moves are approved but 'add' moves approved only if they add
        an edge from a vertex to one of its neighbors. Neighbors are given
        by the dictionary vtx_to_nbors which is calculated before the hill
        climbing starts.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        bool

        """
        m_approved = True
        if move[2] == 'add':
            if move[0] not in self.vtx_to_nbors[move[1]] or\
                    move[1] not in self.vtx_to_nbors[move[0]]:
                m_approved = False
        return m_approved

if __name__ == "__main__":
    HillClimbingLner.HC_lner_test(HC_MMHC_rr_Lner)
