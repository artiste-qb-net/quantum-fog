import pprint as pp

from learning.NetStrucLner import *
from shannon_info_theory.DataEntropy import *


class MB_BasedLner(NetStrucLner):
    """
    MB_BasedLner (Markov Blanket Based Learner) is an abstract class for
    learning the structure of a bnet by first finding the markov blanket of
    each node, then using that MB info to find the neighbors of each node,
    then orienting the edges of each node with its neighbors. The procedure
    for orienting the edges is not 100% effective and must be patched up (
    it might introduce some cycles and leave some edges undecided. A
    heuristic is introduced to patch things up.

    The first MB based learner was Grow Shrink (referring to growing and
    shrinking of the MB) by Margaritis. See Refs. 1 and 2 for his original
    paper and his 2003 Thesis at Carnegie Mellon.

    Many variations of Grow Shrink were introduced after it. In Quantum Fog,
    Grow Shrink and all its variants are subclasses of this class,
    MB_BasedLner, and their names all start with 'MB_' for easy
    identification and so they all stay together in an alphabetical listing
    of files.

    Ref. 3, the PhD thesis of Shunkai Fu, was very helpful in writing the
    MB_ classes, because it contains pseudo code for most of the MB_
    algorithms. However, note that in that pseudo code, whenever it says I <
    epsilon, it means that the conditional mutual info I > epsilon.

    See Shunkai Fu Thesis if you want to know who invented each MB_
    algorithm and in which papers they proposed it for the first time. The
    References given below are not necessarily the first papers, but rather
    papers wih good pseudo code


    References
    ----------

    1. D. Margaritis and S. Thrun, Bayesian Network Induction via Local
    Neighborhoods Adv. in Neural Info. Proc. Sys. 12 (MIT Press, 2000)

    2. D. Margaritis, Learning Bayesian Network Model Structure from Data,
    Thesis 2003 (Carnegie Mellon Univ.)

    3. Shunkai Fu, Efficient Learning of Markov Blanket and Markov Blanket
    Classifier, Thesis 2010, UNIVERSITÉ DE MONTRÉAL

    4. Jean-Philippe Pellet, Andre´ Elisseeff, Using Markov Blankets for
    Causal Structure Learning  (Journal of Machine Learning Research 9, 2008)

    5. Nicholas Cullen, NeuroBN at Github


    Attributes
    ----------
    alpha : float
        threshold used for deciding whether a conditional or unconditional
        mutual info is said to be close to zero (independence) or not (
        dependence). The error in a data entropy is on the order of ln(n+1)
        - ln(n) \approx 1/n where n is the number of samples so 5/n is a
        good default value for alpha.
    verbose : bool
        True for this prints a running commentary to console
    vtx_to_MB : dict[str, list[str]]
        A dictionary mapping each vertex to a list of the vertices in its
        Markov Blanket. (The MB of a node consists of its parents, children
        and children's parents, aka spouses).
    vtx_to_nbors : dict[str, list[str]]
        a dictionary mapping each vertex to a list of its neighbors. The
        literature also calls the set of neighbors of a vertex its PC (
        parents-children) set.
    vtx_to_parents : dict[str, list[str]]
        dictionary mapping each vertex to a list of its parents's names

    """

    def __init__(self, states_df, alpha, verbose=False,
                 vtx_to_states=None, learn_later=False):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame
        alpha : float
        verbose : bool
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.bnet. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df
        learn_later : bool
            False if you want to call the function learn_struc() inside the
            constructor. True if not.

        Returns
        -------
        None


        """
        NetStrucLner.__init__(self, False, states_df, vtx_to_states)
        self.alpha = alpha
        self.verbose = verbose

        self.vtx_to_MB = None
        self.vtx_to_parents = None
        self.vtx_to_nbors = None

        if not learn_later:
            self.learn_struc()

    def learn_struc(self):
        """
        This is the orchestra conductor of the symphony. Each of the
        functions it calls does a lot. By the end, a whole bnet structure
        has been learned from the data and has been stored in self.bnet.


        Returns
        -------
        None

        """
        
        self.find_MB()
        self.find_nbors()
        self.orient_edges()
        self.undo_cycles()
        self.orient_undecided_edges()
        self.fill_bnet_with_parents(self.vtx_to_parents)

    def find_MB(self, vtx=None):
        """
        This function finds the MB of vtx and stores it inside vtx_to_MB[
        vtx]. If vtx=None, then it will find the MB of all the vertices of
        the graph.

        This function is overridden by all the subclasses of this class (the
        ones with names starting with MB_). All the other functions called
        by learn_struc() are the same for most of the subclasses of this
        class.

        Parameters
        ----------
        vtx : str


        Returns
        -------
        bool

        """
        assert False

    def find_nbors(self):
        """
        Finds for each vtx of the graph, a list of all its neighbors and
        puts that info into vtx_to_nbors.

        Returns
        -------
        None

        """

        if self.verbose:
            print('\nbegin find_nbors')

        vertices = self.states_df.columns
        # list all vertices in case some have no neighbors
        self.vtx_to_nbors = {vtx: [] for vtx in vertices}

        for x in vertices:
            for y in self.vtx_to_MB[x]:
                # x and y are direct neighbors if
                # H(x:y|sub_list) >> 0 for all sub_list in super_list
                # where super_list is the smaller of the two lists
                # MB(x)-y and MB(y)-x
                set1 = set(self.vtx_to_MB[x]) - {y}
                set2 = set(self.vtx_to_MB[y]) - {x}
                if len(set1) < len(set2):
                    min_set = set1
                else:
                    min_set = set2
                # min_set = set1 & set2
                super_list = list(min_set)
                x_y_are_dep = True
                for combi_len in range(len(super_list)):
                    for sub_list in it.combinations(super_list, combi_len):
                        mi = DataEntropy.cond_mut_info(self.states_df,
                                    [x], [y], list(sub_list))
                        if mi < self.alpha:
                            x_y_are_dep = False
                            break
                    if not x_y_are_dep:
                        break
                if x_y_are_dep:
                    if y not in self.vtx_to_nbors[x]:
                        self.vtx_to_nbors[x].append(y)
                        self.vtx_to_nbors[y].append(x)
        if self.verbose:
            print('vtx_to_nbors=')
            pp.pprint(self.vtx_to_nbors, width=1)
            print('end find_nbors')

    def orient_edges(self):
        """
        This function gives an orientation to some (not necessarily all) the
        undirected edges implied by vtx_to_nbors. The edge orientation info
        found by this function is stored by it in vtx_to_parents.

        Returns
        -------
        None

        """
        if self.verbose:
            print('\nbegin orient_MB_edges')

        vertices = self.states_df.columns

        self.vtx_to_parents = {vtx: [] for vtx in vertices}
        for x in vertices:
            for y in self.vtx_to_nbors[x]:
                # set x->y if there exists z in sub_list
                # such that H(y:z| sub_list union x) >> 0
                # for all sub_list in super_list,
                # where super_list is the smaller of the two lists
                # MB(y)-{x, z} and MB(z)-{x, y}
                z_set = set(self.vtx_to_nbors[x]) - set(self.vtx_to_nbors[y])
                z_set = z_set - {y}
                y_to_x = False
                for z in z_set:
                    set1 = set(self.vtx_to_MB[y]) - {z}
                    set2 = set(self.vtx_to_MB[z]) - {y}
                    if len(set1) < len(set2):
                        min_set = set1
                    else:
                        min_set = set2
                    # min_set = set1 & set2
                    super_list = list(min_set)
                    y_to_x = True
                    for combi_len in range(len(super_list)):
                        for sub_list in it.combinations(super_list, combi_len):
                            mi = DataEntropy.cond_mut_info(self.states_df,
                                    [y], [z], list(set(sub_list) | {x}))
                            if mi < self.alpha:
                                y_to_x = False
                                break
                        if not y_to_x:
                            break
                    if y_to_x:
                        break
                if y_to_x:
                    if y not in self.vtx_to_parents[x] and \
                            x not in self.vtx_to_parents[y]:
                        self.vtx_to_parents[x].append(y)
        if self.verbose:
            print('vtx_to_parents=')
            pp.pprint(self.vtx_to_parents, width=1)
            print('end orient_MB_edges')

    def new_filled_nx_graph(self):
        """
        This function fills nx_graph with the info found in vtx_to_parents.

        Returns
        -------
        networkx.DiGraph

        """
        vertices = self.states_df.columns
        nx_graph = nx.DiGraph()
        for vtx in vertices:
            nx_graph.add_node(vtx)
            nx_graph.add_edges_from([(pa_vtx, vtx)
                    for pa_vtx in self.vtx_to_parents[vtx]])
        return nx_graph

    def undo_cycles(self):
        """
        When this function is called in learn_str(), the vtx_to_parents that
        has been leaned so far may imply (directed) cycles. This function
        uses a reasonable but not rigorous heuristic to reverse the
        direction of at least one arrow in each cycle and make it a non-cycle.

        Returns
        -------
        None

        """
        if self.verbose:
            print('\nbegin undo_cycles')
        # vertices = self.states_df.columns
        nx_graph = self.new_filled_nx_graph()
        dir_edge_to_freq = {}

        bad_dir_edges = []
        cycles = list(nx.simple_cycles(nx_graph))
        num_cyc = len(cycles)
        # print('cycles=', cycles)
        while num_cyc > 0:
            for cyc in cycles:
                for dir_edge in cyc:
                    if dir_edge not in dir_edge_to_freq.keys():
                        dir_edge_to_freq[dir_edge] = 1
                    else:
                        dir_edge_to_freq[dir_edge] += 1
            # xx = {'a':100, 'b':300, 'c':5}
            # print(max(xx, key=xx.get))
            max_freq_edge = max(dir_edge_to_freq,
                                key=dir_edge_to_freq.get)
            print('dir_edge_to_freq=', dir_edge_to_freq)
            bad_dir_edges.append(max_freq_edge)
            (beg_vtx, end_vtx) = max_freq_edge
            self.vtx_to_parents[end_vtx].remove(beg_vtx)
            nx_graph.remove_edge(beg_vtx, end_vtx)
            cycles = list(nx.simple_cycles(nx_graph))
            num_cyc = len(cycles)
        for (beg_vtx, end_vtx) in reversed(bad_dir_edges):
            self.vtx_to_parents[beg_vtx].append(end_vtx)

        if self.verbose:
            print('vtx_to_parents=')
            pp.pprint(self.vtx_to_parents, width=1)
            print('end undo_cycles')

    def orient_undecided_edges(self):
        """
        When this function is called in learn_str(), the vtx_to_parents that
        has been learned so far may not include all of the edges implied by
        vtx_to_nbors. Hence, there might still be some undirected edges.
        This function uses a reasonable but not rigorous heuristic to orient
        those undecided edges.

        Returns
        -------
        None

        """
        if self.verbose:
            print('\nbegin orient_undecided_edges')

        vertices = self.states_df.columns
        nx_graph = self.new_filled_nx_graph()
        undecided_edges = []
        for vtx in vertices:
            nbor_set = set(self.vtx_to_nbors[vtx]) - \
                       set(nx.all_neighbors(nx_graph, vtx))
            for nbor in nbor_set:
                if (nbor, vtx) not in undecided_edges:
                    undecided_edges.append((vtx, nbor))
        
        for beg_vtx, end_vtx in undecided_edges:
            # add dir_edge to nx_graph in one direction
            # and see if it causes cycle
            # If it doesn't, then
            # add dir edge to vtx_to_parents in same direction
            # and if it does add to vtx_to_parents in opposite direction
            nx_graph.add_edge(beg_vtx, end_vtx)
            try:
                cycle_edge_list = nx.find_cycle(nx_graph, source=beg_vtx)
            except nx.exception.NetworkXNoCycle:
                cycle_edge_list = []  
            if len(cycle_edge_list) == 0:
                self.vtx_to_parents[end_vtx].append(beg_vtx)
            else:
                self.vtx_to_parents[beg_vtx].append(end_vtx)
            # restore nx_graph to original state
            nx_graph.remove_edge(beg_vtx, end_vtx)

        if self.verbose:
            print('undecided edges=', undecided_edges)
            print('vtx_to_parents=')
            pp.pprint(self.vtx_to_parents, width=1)
            print('end orient_undecided_edges')

    @staticmethod
    def MB_lner_test(LnerClass, verbose=False):
        """
        This static method gives a simple example that we use to test
        MB_BasedLner and its subclasses (those starting with MB_). The
        method takes as input training data generated from 2 graphs (the
        classical versions of wetgrass and earthquake) and it outputs a
        drawing of the learned structure.

        Parameters
        ----------
        LnerClass : MB_BasedLner or subclass
            This is either MB_BasedLner without quotes or the name of a
            subclass of that class.


        Returns
        -------
        None

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

        for path in [path1, path2]:
            print('\n######### new path=', path)
            states_df = pd.read_csv(path, dtype=str)
            num_sam = len(states_df.index)
            alpha = None
            if path == path1:
                alpha = 4/num_sam
            elif path == path2:
                alpha = 4/num_sam
            lner = LnerClass(states_df, alpha, verbose=verbose)
            lner.bnet.draw(algo_num=1)
            # nx.draw_networkx(lner.nx_graph)
            # plt.axis('off')
            # plt.show()

if __name__ == "__main__":
    print(5)
