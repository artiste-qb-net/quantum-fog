from learning.HillClimbingLner import *
import copy as cp


class HC_TabuLner(HillClimbingLner):
    """
    The class HC_TabuLner (Hill Climbing Tabu Learner) is a child of
    HiilClimbingLner. It adds to the latter a Tabu list, which is a memory
    of places that are forbidden to revisit, at least temporarily.

    The idea is to keep a list of the last n moves. Then disallow any next
    move that is in that list, BUT allow next moves that go downhill if no
    uphill moves are currently possible and allowed. Each time a downhill
    move is necessary, the starting point of that downhill move is a local
    maximum. Of all local maxs encountered, the one with the highest score
    is selected in the end.

    Attributes
    ----------
    best_loc_max_graph : networkx.DiGraph
    best_loc_max_score : float
        Every time the restart() function is called because a try yields no
        moves with positive score change, we infer that a new local max has
        been reached. If total score of the current loc max is higher than
        best_loc_mac_score, we replace both best_loc_max_score and
        best_loc_max_graph by those of the better local max.
    loc_max_ctr : int
        local maximum counter, counts the number of local maxs encountered.
    nx_graph : networkx.DiGraph
        a networkx directed graph used to store arrows
    tabu_list : list[tuple[str, str, str]]
        a list of the previous moves. The list's length is specified in the
        constructor by means of tabu_len parameter. Every time a new move is
        added to end of the tabu list, the first item of the list is removed.

    """

    def __init__(self, states_df, score_type, max_num_mtries,
            tabu_len=10,
            ess=1.0, verbose=False, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        tabu_len : int
            length of tabu list
        states_df : pandas.DataFrame
        score_type : str
        max_num_mtries : int
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

        self.tabu_list = [None]*tabu_len
        self.best_loc_max_graph = None
        self.best_loc_max_score = None
        self.loc_max_ctr = 0
        HillClimbingLner.__init__(
            self, states_df, score_type, max_num_mtries,
            ess, verbose, vtx_to_states)

    @staticmethod
    def equal_dags(vtx_to_parents1, vtx_to_parents2, num_nodes):
        """
        Returns True iff its two input dicts denote the same labelled dag.

        Parameters
        ----------
        vtx_to_parents1 : dict[str, list[str]]
            dictionary mapping each vertex to a list of its parents's names
        vtx_to_parents2 : dict[str, list[str]]
            dictionary mapping each vertex to a list of its parents's names
        num_nodes : int
            number of nodes

        Returns
        -------
        bool

        """

        assert len(vtx_to_parents1) == num_nodes
        assert len(vtx_to_parents2) == num_nodes
        for vtx in vtx_to_parents1:
            if set(vtx_to_parents1[vtx]) != set(vtx_to_parents2[vtx]):
                return False
        return True

    def move_approved(self, move):
        """
        Returns True indicating approval of input move iff the move would
        NOT take the current dag to a dag which has been visited before,
        as far as one can infer by backtracking from current vtx_to_parents
        using tabu list of moves.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        bool

        """
        # initial self.tabu_list has all None entries,
        # gradually they get replaced by moves.
        # Remove None's from self.tabu_list
        mini_tabu_list = [x for x in self.tabu_list if x is not None]
        vtx_to_parents_next = cp.deepcopy(self.vtx_to_parents)
        try:
            HillClimbingLner.do_move_vtx_to_parents(
                move, vtx_to_parents_next)
        except:
            return False
        vtx_to_parents_past = cp.deepcopy(self.vtx_to_parents)
        for past_move in reversed(mini_tabu_list):
            HillClimbingLner.do_move_vtx_to_parents(
                past_move, vtx_to_parents_past, reversal=True)
            if HC_TabuLner.equal_dags(vtx_to_parents_next,
                    vtx_to_parents_past, len(self.vtx_to_parents)):
                return False
        return True

    def finish_do_move(self, move):
        """
        Adds move to end of tabu list and removes first element of tabu list.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        None

        """
        self.tabu_list.pop(0)
        self.tabu_list.append(move)

    def restart(self, mtry_num):
        """
        Takes in mtry_num, the number of the current move try, and returns (
        restart_approved, mtry_num_fin) where mtry_num_fin equals the input
        mtry_num. So this function does nothing to mtry_num, although it
        could have. For example, in the class HC_RandomStartLner,
        this function sets mtry_num to zero. Restarts are always approved
        until they reach their limit given by max_num_mtries. Current
        nx_graph and score is stored if its score is better than previous
        best score.

        Parameters
        ----------
        mtry_num : int

        Returns
        -------
        bool, int

        """
        restart_approved = True
        self.loc_max_ctr += 1
        # if self.verbose:
        #     print('\n****number and score of local max=',
        #           self.loc_max_ctr, self.scorer.tot_score)
        if self.best_loc_max_score is None or\
                self.best_loc_max_score < self.scorer.tot_score:
            self.best_loc_max_score = self.scorer.tot_score
            self.best_loc_max_graph = cp.deepcopy(self.nx_graph)
        # print('mtry inside restart', mtry_num)
        if mtry_num == self.max_num_mtries:
            restart_approved = False
            for vtx in self.vertices:
                self.vtx_to_parents[vtx] = \
                    self.best_loc_max_graph.predecessors(vtx)
            self.nx_graph = self.best_loc_max_graph
            if self.verbose:
                print('*************saved best')
        # don't reset mtry_num to zero like random starts does
        return restart_approved, mtry_num

if __name__ == "__main__":
    HillClimbingLner.HC_lner_test(HC_TabuLner, verbose=True)
