from learning.MB_BasedLner import *


class MB_lambda_IAMB_Lner(MB_BasedLner):
    """
    The MB_lambda_IAMB_Lner (lambda Incremental Association Markov Blanket
    Learner) is a subclass of MB_BasedLner. See docstring for MB_BasedLner
    for more info about this type of algo.

    lambda refers to the fact tha it uses an extra parameter lambda between
    zero and one.

    See Ref. 1 below for pseudo code on which this class is based.

    References
    ----------

    1. An Improved IAMB Algorithm for Markov Blanket Discovery, by Yishi
    Zhang, Zigang Zhang, Kaijun Liu, and Gangyi Qian (JCP 2010 Vol.5(11))


    Attributes
    ----------
    lam : float
        extra parameter between 0 and 1. The closer it is to 1, the fewer
        elements are added to MB
    vtx_to_MB : dict[str, list[str]]
        A dictionary mapping each vertex to a list of the vertices in its
        Markov Blanket. (The MB of a node consists of its parents, children
        and children's parents, aka spouses).

    """

    def __init__(self, states_df, alpha, verbose=False,
                 vtx_to_states=None, lam=.5, learn_later=False):
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
        lam : float
        learn_later : bool
            False if you want to call the function learn_struc() inside the
            constructor. True if not.

        Returns
        -------
        None

        """
        self.lam = lam
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
            print('lambda=', self.lam)

        vertices = self.states_df.columns
        if vtx is None:
            tar_list = vertices
        else:
            tar_list = [vtx]

        self.vtx_to_MB = {}

        def ci__(a, b):  # H(a | b)
            return DataEntropy.cond_info(self.states_df, a, b)

        def MB(a):
            return self.vtx_to_MB[a]

        for tar in tar_list:
            self.vtx_to_MB[tar] = []

            # growing phase
            if self.verbose:
                print('\n****begin growing phase')
            growing = True
            while growing:
                growing = False
                ht_y1_min, hty1_min, y1_min = None, None, None
                ht_y2_min, hty2_min, y2_min = None, None, None
                ht = ci__([tar], MB(tar))   # H(tar | MB(tar))
                y1_set = (set(vertices) - {tar}) - set(MB(tar))

                for y1 in y1_set:

                    # H(tar | MB(tar), y1)
                    ht_y1 = ci__([tar], list(set(MB(tar)) | {y1}))

                    # H( tar: y1 |MB(tar))
                    hty1 = ht - ht_y1

                    if ht_y1_min is None or ht_y1 < ht_y1_min:
                        ht_y1_min = ht_y1
                        hty1_min = hty1
                        y1_min = y1
                y2_set = y1_set - {y1_min}

                for y2 in y2_set:
                    # H(tar | MB(tar), y2)
                    ht_y2 = ci__([tar], list(set(MB(tar)) | {y2}))

                    # H( tar: y2 |MB(tar))
                    hty2 = ht - ht_y2

                    if ht_y2_min is None or ht_y2 < ht_y2_min:
                        ht_y2_min = ht_y2
                        hty2_min = hty2
                        y2_min = y2
                if y1_min is not None and hty1_min > self.alpha:
                    if y2_min is not None and hty2_min > self.alpha and\
                            ht_y2_min - self.lam*ht_y1_min < (1-self.lam)*ht:
                        self.vtx_to_MB[tar].append(y1_min)
                        self.vtx_to_MB[tar].append(y2_min)
                        growing = True
                elif y1_min is not None:
                    self.vtx_to_MB[tar].append(y1_min)
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
                for y in MB(tar):
                    cmi = DataEntropy.cond_mut_info(self.states_df,
                                [y], [tar], list(set(MB(tar)) - {y}))
                    if cmi < self.alpha:
                        self.vtx_to_MB[tar].remove(y)
                        shrinking = True
            if self.verbose:
                print('target, MB(tar) aft-shrinking:')
                print(tar, self.vtx_to_MB[tar])
                print('end shrinking phase')

if __name__ == "__main__":
    MB_BasedLner.MB_lner_test(MB_lambda_IAMB_Lner, verbose=True)
