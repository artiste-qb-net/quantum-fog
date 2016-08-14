from learning.MB_BasedLner import *


class MB_inter_IAMB_Lner(MB_BasedLner):
    """

    The MB_inter_IAMB_Lner (Interleaved Incremental Association Markov
    Blanket Learner) is a subclass of MB_BasedLner. See docstring for
    MB_BasedLner for more info about this type of algo.

    Interleaved refers to the fact that the growing and shrinking phases (
    growing = adding true positives, shrinking = removing false positives)
    are intermingled and performed at the same time rather than one after
    the other as in the original MB_ algo, Grow Shrink.

    See Shunkai Fu Thesis for pseudo code on which this class is based.


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
            This information will be stored in self.dag. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df
        learn_later : bool
            False if you want to call the function learn_struc() inside the
            constructor. True if not.

        Returns
        -------
        None

        """
        MB_BasedLner.__init__(self, states_df, alpha, verbose,
                              vtx_to_states, learn_later)

    def find_MB(self, vtx=None):
        """
        This function finds the MB of vtx and stores it inside vtx_to_MB[
        vtx]. If vtx=None, then it will find the MB of all the vertices of
        the graph.

        Parameters
        ----------
        vtx : str

        Returns
        -------
        None

        """
        if self.verbose:
            print('alpha=', self.alpha)

        vertices = self.states_df.columns
        if vtx is None:
            tar_list = vertices
        else:
            tar_list = [vtx]

        self.vtx_to_MB = {}
        for tar in tar_list:
            self.vtx_to_MB[tar] = []
            changing = True
            while changing:
                changing = False
                mi_max = None
                y_max = None
                for y in vertices:
                    if y != tar and y not in self.vtx_to_MB[tar]:
                        mi = DataEntropy.cond_mut_info(self.states_df,
                                    [tar], [y], self.vtx_to_MB[tar])
                        if mi_max is None or mi > mi_max:
                            mi_max = mi
                            y_max = y
                            if self.verbose:
                                print('tar, y_max, mi_max=',
                                      tar, y_max, mi_max)
                if y_max is not None and mi_max > self.alpha:
                    self.vtx_to_MB[tar].append(y_max)
                    changing = True

                for y in self.vtx_to_MB[tar]:
                    mi = DataEntropy.cond_mut_info(self.states_df,
                             [tar], [y], list(set(self.vtx_to_MB[tar]) - {y}))
                    if mi < self.alpha:
                        self.vtx_to_MB[tar].remove(y)
                        changing = True

if __name__ == "__main__":
    MB_BasedLner.MB_lner_test(MB_inter_IAMB_Lner, verbose=True)
