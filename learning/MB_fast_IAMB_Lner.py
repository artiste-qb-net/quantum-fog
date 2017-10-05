from learning.MB_BasedLner import *
import operator


class MB_fast_IAMB_Lner(MB_BasedLner):
    """

    The MB_fast_IAMB_Lner (fast Incremental Association Markov Blanket
    Learner) is a subclass of MB_BasedLner. See docstring for MB_BasedLner
    for more info about this type of algo.

    See Shunkai Fu Thesis for pseudo code on which this class is based.

    Attributes
    ----------
    min_sam_per_cell : int
        The threshold for there to be sufficient data (for meaningful
        prediction of an element of the MB of a vertex) is proportional to
        this number according to t-test of statistics. Usually taken to be 5.
    vtx_to_MB : dict[str, list[str]]
        A dictionary mapping each vertex to a list of the vertices in its
        Markov Blanket. (The MB of a node consists of its parents, children
        and children's parents, aka spouses).

    """

    def __init__(self, states_df, alpha, min_sam_per_cell=5,
                 verbose=False, vtx_to_states=None, learn_later=False):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame
        alpha : float
        min_sam_per_cell : int
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
        self.min_sam_per_cell = min_sam_per_cell
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

        num_sam = len(self.states_df.index)
        vtx_to_size = {nd.name:  nd.size for nd in self.bnet.nodes}

        def cmi__(y, tar):  # H( y:tar | MB(tar)-y )
            return DataEntropy.cond_mut_info(self.states_df,
                [y], [tar], list(set(self.vtx_to_MB[tar]) - {y}))

        for tar in tar_list:
            self.vtx_to_MB[tar] = []
            y_cmi_list = []
            first_y_list = True
            while len(y_cmi_list) > 0 or first_y_list:
                insufficient_data = False

                # growing phase
                if self.verbose:
                    print('\n****begin growing phase')
                if first_y_list:
                    first_y_list = False
                    for y in set(vertices) - {tar}:
                        cmi = cmi__(y, tar)
                        if cmi > self.alpha:
                            y_cmi_list.append((y, cmi))
                else:
                    y_list, cmi_list = zip(*y_cmi_list)
                    y_cmi_list = [(y, cmi__(y, tar)) for y in y_list]
                    y_cmi_list.sort(key=operator.itemgetter(1), reverse=True)
                for y, cmi in y_cmi_list:
                    mb_size = np.prod([
                        vtx_to_size[vtx] for vtx in self.vtx_to_MB[tar]])
                    denom = vtx_to_size[y]*vtx_to_size[tar]*mb_size
                    if num_sam/denom >= self.min_sam_per_cell:
                        self.vtx_to_MB[tar].append(y)
                    else:
                        insufficient_data = True
                        break

                if self.verbose:
                    print('target, MB(tar) aft-growing, bef-shrinking:')
                    print(tar, self.vtx_to_MB[tar])
                    print('end growing phase')
                    print('****begin shrinking phase')

                # shrinking phase
                shrinking = False
                for y in self.vtx_to_MB[tar]:
                    if cmi__(y, tar) < self.alpha:
                        self.vtx_to_MB[tar].remove(y)
                        shrinking = True

                if insufficient_data and not shrinking:
                    print('insufficient data and not shrinking')
                    break
                else:
                    y_cmi_list = []
                    for y in vertices:
                        if y != tar and y not in self.vtx_to_MB[tar]:
                            cmi = cmi__(y, tar)
                            if cmi > self.alpha:
                                y_cmi_list.append((y, cmi))
                if self.verbose:
                    print('target, MB(tar) aft-shrinking:')
                    print(tar, self.vtx_to_MB[tar])
                    print('end shrinking phase')

if __name__ == "__main__":
    MB_BasedLner.MB_lner_test(MB_fast_IAMB_Lner, verbose=True)
