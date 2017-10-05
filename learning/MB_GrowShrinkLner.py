from learning.MB_BasedLner import *


class MB_GrowShrinkLner(MB_BasedLner):
    """
    The MB_GrowShrinkLner (Grow Shrink Learner) is a subclass of
    MB_BasedLner. See docstring for MB_BasedLner for more info about this
    type of algo.

    See Shunkai Fu Thesis for pseudo code on which this class is based.

    Attributes
    ----------
    vtx_to_MB : dict[str, list[str]]
        A dictionary mapping each vertex to a list of the vertices in its
        Markov Blanket. (The MB of a node consists of its parents, children
        and children's parents, aka spouses).

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

            # growing phase
            if self.verbose:
                print('\n****begin growing phase')
            growing = True
            while growing:
                growing = False
                for y in vertices:
                    # y must be in vertices - self.vtx_to_MB[tar] - tar
                    if y != tar and y not in self.vtx_to_MB[tar]:
                        # if H(tar:y|self.vtx_to_MB[tar]) >> 0,
                        # add y to self.vtx_to_MB[tar]
                        mi = DataEntropy.cond_mut_info(self.states_df,
                                [tar], [y], self.vtx_to_MB[tar])
                        # if self.verbose:
                        #     print('tar, y, mi=', tar, y, mi)
                        if mi > self.alpha:
                            self.vtx_to_MB[tar].append(y)
                            growing = True
            if self.verbose:
                print('target, MB(tar) aft-growing, bef-shrinking:')
                print(tar, self.vtx_to_MB[tar])
                print('end growing phase')
                print('****begin shrinking phase')

            # shrinking phase
            shrinking = True
            while shrinking:
                shrinking = False
                for y in self.vtx_to_MB[tar]:
                    # if H(tar:y|self.vtx_to_MB[tar]-y) \approx 0,
                    # remove y from self.vtx_to_MB[tar]
                    mi = DataEntropy.cond_mut_info(self.states_df,
                        [tar], [y], list(set(self.vtx_to_MB[tar])-{y}))
                    # if self.verbose:
                    #     print('tar, y, mi=', tar, y, mi)
                    if mi < self.alpha:
                        self.vtx_to_MB[tar].remove(y)
                        shrinking = True
            if self.verbose:
                print('target, MB(tar) aft-shrinking:')
                print(tar, self.vtx_to_MB[tar])
                print('end shrinking phase')

if __name__ == "__main__":
    MB_BasedLner.MB_lner_test(MB_GrowShrinkLner, verbose=True)
