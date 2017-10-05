from learning.HillClimbingLner import *
import networkx as nx
import pprint as pp


class HC_RandRestartLner(HillClimbingLner):
    """

    HC_RandRestartLner (Hill Climbing Random Restart Learner) adds to its
    parent class HillClimbinmgLner a restart() function for random restarts.
    Once a local maximum is reached, a random move ('restart') is made and
    then from that new starting point the system is again driven towards a
    possibly new local maximum. One can specify the number of random starts
    desired. The local maximum with the highest score is selected at the end.

    References
    ----------
    1. Nicholas Cullen neuroBN at github


    Attributes
    ----------
    best_start_graph : networkx.DiGraph
    best_start_score : float
        Each time restart() function is called, the current tot score is
        compared with best_start_score, and if it is higher, then both the
        current nx_graph and its score are stored in best_start_graph and
        best_start_score.
    cur_start : int
        current start. Increases by one every time restart() is called.
    mcache : list[tuple[str, str, str]]
        a list that stores all moves of a try. This list is cleared at
        beginning of each try.
    num_starts : int
        number of starts. Number of times minus 1 that restart function is
        called.
    nx_graph : networkx.DiGraph
        a networkx directed graph used to store arrows
    score_ch_cache : list[tuple[float, float, float]]
        A list that stores the score changes of each move of a try. Score
        changes here are given as a 3-tuple of 3 floats, (beg_score_ch,
        end_score_ch, tot_score_ch).

    """

    def __init__(self, states_df, score_type, max_num_mtries,
            num_starts=10,
            ess=1.0, verbose=False, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.core.frame.DataFrame
        score_type : str
        max_num_mtries : int
        num_starts : int
        ess : float
            Equivalent Sample Size, a parameter in BDEU scorer. Fudge factor
            that is supposed to grow as the amount of prior knowledge grows.
        verbose : bool
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.bnet. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df


        Returns
        -------
        None

        """
        self.num_starts = num_starts
        self.cur_start = 1
        self.mcache = []
        self.score_ch_cache = []
        self.best_start_graph = None
        self.best_start_score = None

        HillClimbingLner.__init__(
            self, states_df, score_type, max_num_mtries,
            ess, verbose, vtx_to_states)

    def restart(self, mtry_num):
        """
        This function takes in mtry_num and returns (restart_approved,
        0). Zero is intended to be the next mtry_num, so mtry_num is set to
        zero. Also the cur_start num is increased by one. All restarts are
        approved as long as they don't exceed num_starts in number.

        The cur total score is compared with best_start_score and it is
        higher, then current nx_graph and its score are stored as best so far.

        The function selects uniformly at random a move from mcache (mcache
        is a list of all non-optimal moves of last try, and score_ch_cache
        is a list of the score changes of those non-optimal moves). This
        random move is submitted to do_move(), so it is performed regardless
        of its non-optimality. The hope is that this random restart will
        carry the system from the attractor basin of the current local max,
        to the attractor basin of another local max.

        Parameters
        ---------
        mtry_num : int

        Returns
        -------
        bool, int

        """
        restart_approved = False
        if self.cur_start <= self.num_starts:
            if self.cur_start == 1:
                self.best_start_graph = cp.deepcopy(self.nx_graph)
                self.best_start_score = self.scorer.tot_score

            if self.verbose:
                print('\n***********************start=', self.cur_start)
            tot_score = self.scorer.tot_score
            if self.cur_start > 1 and tot_score > self.best_start_score:
                self.best_start_graph = cp.deepcopy(self.nx_graph)
                self.best_start_score = tot_score

            rand_int = np.random.randint(low=0, high=len(self.mcache))
            rand_move = self.mcache[rand_int]
            rand_score_change = self.score_ch_cache[rand_int]
            if self.verbose:
                print('before rand move, ',
                      self.score_type + ' cur tot score=',
                      self.scorer.tot_score)
            self.do_move(rand_move, rand_score_change, do_finish=False)
            if self.verbose:
                print("random move", rand_move)
                print('after rand move, ',
                      self.score_type + ' cur tot score=',
                      self.scorer.tot_score)

            restart_approved = True
            self.cur_start += 1
        else:
            for vtx in self.vertices:
                self.vtx_to_parents[vtx] = \
                    self.best_start_graph.predecessors(vtx)
            self.nx_graph = self.best_start_graph
        # reset mtry number to 0
        return restart_approved, 0

    def cache_this(self, move, score_change):
        """
        Stores every move of a try in the list mcache and its score change
        in the list score_ch_cache.

        Parameters
        ----------
        move : tuple[str, str, str]
        score_change : tuple[float, float, float]

        Returns
        -------
        None

        """
        self.mcache.append(move)
        self.score_ch_cache.append(score_change)

    def empty_cache(self):
        """
        Clears mcache and score_ch_cache. Called at the beginning of each try.

        Returns
        -------
        None

        """
        self.mcache.clear()
        self.score_ch_cache.clear()

if __name__ == "__main__":
    HillClimbingLner.HC_lner_test(HC_RandRestartLner, verbose=True)
