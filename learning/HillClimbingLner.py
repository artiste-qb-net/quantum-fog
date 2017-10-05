from learning.NetStrucLner import *
from learning.NetStrucScorer import *
import networkx as nx
import pprint as pp


class HillClimbingLner(NetStrucLner):
    """
    The HiilClimbingLner( Hill Climbing Learner ) class learns the structure
    of a bnet using a greedy strategy, meaning it goes for the highest short
    term gain without caring that that may not be in its long term interest,
    as it may lead it to a local rather than the global maximum.

    Each 'move' consists of either adding, deleting or reversing the 
    direction of an arrow. Each move is given a score. Score keeping is done 
    by an object of a separate class called NetStrucScorer. A 'try' or 
    'mtry' (move try) is a set of candidate moves. Only the highest scoring 
    move of a try is actually performed. 

    Classes that inherit from this one wil have the prefix HC_ for easy
    identification and so that they stay together in an alphabetical listing.

    References
    ----------
    1. Nicholas Cullen neuroBN at github

    Attributes
    ----------
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

    """

    def __init__(self, states_df, score_type, max_num_mtries,
            ess=1.0, verbose=False, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.core.frame.DataFrame
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

        NetStrucLner.__init__(self, False, states_df, vtx_to_states)

        self.max_num_mtries = max_num_mtries
        self.score_type = score_type
        self.verbose = verbose

        self.vertices = states_df.columns
        self.vtx_to_parents = {vtx: [] for vtx in self.vertices}

        # get vtx_to_states info from self.bnet
        vtx_to_states1 = {nd.name: nd.state_names for nd in self.bnet.nodes}
        self.scorer = NetStrucScorer(
            self.states_df,
            self.vtx_to_parents,
            vtx_to_states1,
            score_type,
            ess)

        self.nx_graph = nx.DiGraph()

        self.climb()

    def climb(self):
        """
        This is the main engine of the whole class. Most of the other
        functions of this class are called inside this function.

        Returns
        -------
        None

        """
        try_again = True
        mtry_num = 0
        best_move = None

        while try_again and (mtry_num < self.max_num_mtries):
            max_change = None
            try_again = False
            mtry_num += 1
            self.empty_cache()
            if self.verbose:
                print('\n-----------mtry =', mtry_num)
                print(self.score_type + ' cur tot score=',
                      self.scorer.tot_score)
            for action in ['add', 'del', 'rev']:
                for end_vtx in self.vertices:
                    beg_vtx_look_list = []
                    if action == 'add':
                        beg_vtx_look_list = [x for x in self.vertices
                                if x not in self.vtx_to_parents[end_vtx]]
                    else:  # if action is del or rev
                        beg_vtx_look_list = self.vtx_to_parents[end_vtx]
                    for beg_vtx in beg_vtx_look_list:
                        move = (beg_vtx, end_vtx, action)
                        if not self.would_create_cycle(move) \
                                and self.move_approved(move):
                            score_change = self.scorer.score_move(move)
                            self.cache_this(move, score_change)
                            # print('score change', score_change)
                            # score change is a 3 component tuple
                            # (beg_vtx_score_ch, end_vtx_score_ch,
                            # tot_score_ch)
                            if max_change is None or \
                                    score_change[2] > max_change[2]:
                                best_move = move
                                max_change = score_change
                                if self.verbose:
                                    print("\nbest move so far", best_move)
                                    print('score change', max_change)
            if max_change is not None and max_change[2] > -1e-7:
                try_again = True
                self.do_move(best_move, max_change)
                if self.verbose:
                    print('\nsuccessful mtry')
                    print("move", best_move)
                    print('score change', max_change)
                    print(self.score_type + ' cur tot score=',
                          self.scorer.tot_score)
                    print('vtx_to_parents:')
                    pp.pprint(self.vtx_to_parents, width=1)
            else:
                try_again, mtry_num = self.restart(mtry_num)

        self.fill_bnet_with_parents(self.vtx_to_parents)

    @staticmethod
    def do_move_vtx_to_parents(move, vtx_to_parents, reversal=False):
        """
        Applies a move or its reversal to a vtx_to_parents

        Parameters
        ----------
        move : tuple[str, str, str]
        vtx_to_parents : dict[str, list[str]]
            dictionary mapping each vertex to a list of its parents's names
        reversal : bool
            If True, the function applies reversal of move to vtx_to_parents

        Returns
        -------
        None

        """
        (beg_vtx, end_vtx, action) = move
        if action == 'add':
            if not reversal:
                vtx_to_parents[end_vtx].append(beg_vtx)
            else:
                vtx_to_parents[end_vtx].remove(beg_vtx)
        elif action == 'del':
            if not reversal:
                vtx_to_parents[end_vtx].remove(beg_vtx)
            else:
                vtx_to_parents[end_vtx].append(beg_vtx)
        elif action == 'rev':
            if not reversal:
                vtx_to_parents[end_vtx].remove(beg_vtx)
                vtx_to_parents[beg_vtx].append(end_vtx)
            else:
                vtx_to_parents[beg_vtx].remove(end_vtx)
                vtx_to_parents[end_vtx].append(beg_vtx)
        else:
            assert False

    @staticmethod
    def do_move_nx_graph(move, nx_graph, reversal=False):
        """
        Applies move or its reversal to a networkx dag.

        Parameters
        ----------
        move : tuple[str, str, str]
        nx_graph : networkx.DiGraph
        reversal : bool
            If True, the function applies reversal of move to nx_graph

        Returns
        -------
        None

        """
        (beg_vtx, end_vtx, action) = move
        if action == 'add':
            if not reversal:
                nx_graph.add_edge(beg_vtx, end_vtx)
            else:
                nx_graph.remove_edge(beg_vtx, end_vtx)
        elif action == 'del':
            if not reversal:
                nx_graph.remove_edge(beg_vtx, end_vtx)
            else:
               nx_graph.add_edge(beg_vtx, end_vtx)
        elif action == 'rev':
            if not reversal:
                nx_graph.remove_edge(beg_vtx, end_vtx)
                nx_graph.add_edge(end_vtx, beg_vtx)
            else:
                nx_graph.remove_edge(end_vtx, beg_vtx)
                nx_graph.add_edge(beg_vtx, end_vtx)
        else:
            assert False

    def do_move(self, move, score_change, do_finish=True):
        """
        Once the move has been approved/vetted, it is performed by this
        method.

        Parameters
        ----------
        move : tuple[str, str, str]
            a move is a 3-tuple (beg_vtx, end_vtx, action) describing an
            arrow beg_vtx->end_vtx that will be either added, deleted or
            reversed, depending on whether action = 'add', 'del', or 'rev'.
        score_change : tuple[float, float, float]
            score_change is a 3-tuple (beg_vtx_score_ch, end_vtx_score_ch,
            tot_score_ch). beg_score_ch is the score change of the beg_vtx
            of move, end_score_ch is the score change of the end_vtx of
            move. This is possible because all score functions can be
            evaluated for a single vtx. tot_score_ch is the sum of
            beg_score_ch and end_score_ch minus a positive penalty that
            increases with the size of the network.
        do_finish : bool
            True if you want to finish off this function with a call to
            finish_do_move().

        Returns
        -------
        None

        """
        HillClimbingLner.do_move_vtx_to_parents(move, self.vtx_to_parents)
        HillClimbingLner.do_move_nx_graph(move, self.nx_graph)

        self.scorer.do_move(move, score_change)

        if do_finish:
            self.finish_do_move(move)

    def refresh_nx_graph(self):
        """
        This function clears self.nx_graph and refills it with info in
        self.vtx_to_parents.

        Returns
        -------
        None

        """
        self.nx_graph.clear()
        for vtx in self.vertices:
            self.nx_graph.add_edges_from([(pa_vtx, vtx)
                    for pa_vtx in self.vtx_to_parents[vtx]])

    def would_create_cycle(self, move):
        """
        This function performs the move 'move' on self.nx_graph and then
        tests to see if the new graph has cycles. It communicates the result
        of the tests in the bool output. It restores nx_graph to its
        original state after the testing for cycles is concluded.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        bool

        """

        (beg_vtx, end_vtx, action) = move

        if action == 'del':
            return False

        if action == 'add':
            self.nx_graph.add_edge(beg_vtx, end_vtx)
        elif action == 'rev':
            self.nx_graph.remove_edge(beg_vtx, end_vtx)
            self.nx_graph.add_edge(end_vtx, beg_vtx)
        else:
            assert False

        try:
            cycle_edge_list = nx.find_cycle(self.nx_graph, source=beg_vtx)
        except nx.exception.NetworkXNoCycle:
            cycle_edge_list = []
        # restore self.nx_graph to original state
        if action == 'add':
            self.nx_graph.remove_edge(beg_vtx, end_vtx)
        elif action == 'rev':
            self.nx_graph.add_edge(beg_vtx, end_vtx)
            self.nx_graph.remove_edge(end_vtx, beg_vtx)

        return len(cycle_edge_list) > 0

    def move_approved(self, move):
        """
        This is a hook function that allows subclasses of this class to
        impose more stringent requirements on a move before it is approved,
        beyond the usual requirement that the move not create a cycle. The
        function returns a bool, its decision.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        bool

        """
        return True

    def finish_do_move(self, move):
        """
        This is a hook function that allows subclasses of this class to do
        some additional processing before the do_move() function is concluded.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        None

        """

        pass

    def restart(self, mtry_num):
        """
        This is a hook function that allows subclasses of this class to
        restart the trying process when the last try yielded no moves with
        positive score change. mtry_num is passed in. (restart_approved,
        mtry_num) are returned, where restart_approved is a boolean
        signaling approval and mtry_num is an int, usually either the
        inputted mtry_num or zero.

        Parameters
        ----------

        mtry_num : int
            move try num is the number of the current try

        Returns
        -------
        bool, int

        """
        return False, 0

    def cache_this(self, move, score_change):
        """
        This is a hook function that allows subclasses of this class to
        store in a list every move and its score change that was considered
        (whether it was the best move of the try or not) for the current try.

        Parameters
        ----------
        move : tuple[str, str, str]
        score_change : tuple[float, float, float]

        Returns
        -------
        None

        """
        pass

    def empty_cache(self):
        """
        This is a hook function that allows subclasses of this class to
        clear its cached list of moves before starting the next try.

        Returns
        -------
        None

        """
        pass

    @staticmethod
    def HC_lner_test(LnerClass, verbose=False):
        """
        This static method gives a simple example that we use to test
        HillClimbingLner and its subclasses (those starting with HC_). The
        method takes in as input training data generated from 2 graphs (the
        classical versions of wetgrass and earthquake), and it outputs a
        drawing of the learned structure for 2 scoring functions (
        BIC-frequentist and BDEU-bayesian)

        Parameters
        ----------
        LnerClass : HillClimbingLner or subclass
            This is either HillClimbingLner without quotes or the name of a
            subclass of that class.

        verbose : bool

        Returns
        -------

        """
        path1 = 'training_data_c/WetGrass.csv'
        # true:
        # All arrows pointing down
        #    Cloudy
        #    /    \
        # Rain    Sprinkler
        #   \      /
        #   WetGrass

        path2 = 'training_data_c/earthquake.csv'
        # true:
        # All arrows pointing down
        # burglary   earthquake
        #   \         /
        #      alarm
        #   /         \
        # johnCalls  maryCalls

        # score types LL, BIC, AIC, BDEU, K2

        for score_type in ['BIC', 'BDEU']:
            for path in [path1, path2]:

                print('\n######### new path=', path)
                max_num_mtries = 30
                states_df = pd.read_csv(path, dtype=str)
                lner = LnerClass(
                    states_df, score_type, max_num_mtries,
                    ess=2, verbose=verbose)
                plt.title(score_type)
                lner.bnet.draw(algo_num=1)
                # nx.draw_networkx(lner.nx_graph)
                # plt.axis('off')
                # plt.show()

if __name__ == "__main__":
    HillClimbingLner.HC_lner_test(HillClimbingLner, verbose=True)

