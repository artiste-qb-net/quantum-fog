from learning.MB_BasedLner import *


class MB_hitonPC_Lner(MB_BasedLner):
    """

    The MB_hitonPCLner (hiton Parents Children Learner, hiton means blanket
    in Greek) is a subclass of MB_BasedLner. See docstring for MB_BasedLner
    for more info about this type of algo.

    Whereas the first generation of MB_ algos found the MB of each node
    first and the neighbors of each node second, the hiton PC
    algorithm finds these in the opposite order.

    PC stands for Parents-Children. The PC set of a node is the same thing as
    its set of neighbors.

    See Shunkai Fu Thesis for pseudo code on which this class is based.


    Attributes
    ----------
    vtx_to_MB : dict[str, list[str]]
        A dictionary mapping each vertex to a list of the vertices in its
        Markov Blanket. (The MB of a node consists of its parents, children
        and children's parents, aka spouses).
    vtx_to_nbors : dict[str, list[str]]
        a dictionary mapping each vertex to a list of its neighbors. The
        literature also calls the set of neighbors of a vertex its PC (
        parents-children) set.

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

    def find_nbors(self):
        """
        This function is called in the parent class MB_BasedLner after the
        function find_MB() is. But in this subclass we want to find the
        neighbors before finding the MBs. We solve this quandary by
        overriding find_nbors() so that it does nothing, and defining a new
        function find_PC() that also finds the neighbors, but is called
        inside find_MB(). In MB_BasedLner, the function find_nbors() assumes
        that the MB of all nodes is known whereas in this subclass,
        the function find_PC() doesn't assume this.

        Returns
        -------
        None

        """
        pass

    def find_PC(self, vtx=None):
        """
        This function finds the PC set (Parents-Children, aka neighbors) of
        vtx and stores it in vtx_to_nbors. if vtx=None, it finds the PC set
        of all nodes.

        Parameters
        ----------
        vtx : str

        Returns
        -------
        None

        """

        vertices = self.states_df.columns
        if vtx is None:
            tar_list = vertices
        else:
            tar_list = [vtx]

        self.vtx_to_nbors = {}
        for tar in tar_list:
            self.vtx_to_nbors[tar] = []
            super_PC_of_tar = list(set(vertices) - {tar})
            while super_PC_of_tar:
                y_min = None
                mi_min = None
                for y in super_PC_of_tar:
                    mi = DataEntropy.mut_info(self.states_df, [tar], [y])
                    if mi_min is None or mi < mi_min:
                        y_min = y
                        mi_min = mi
                self.vtx_to_nbors[tar].append(y_min)
                super_PC_of_tar.remove(y_min)
                for y in self.vtx_to_nbors[tar]:
                    super_list = list(set(self.vtx_to_nbors[tar])-{y})
                    sub_list_exists = False
                    for combi_len in range(len(super_list)):
                        for sub_list in \
                                it.combinations(super_list, combi_len):
                            mi = DataEntropy.cond_mut_info(self.states_df,
                                    [tar], [y], list(sub_list))
                            if mi < self.alpha:
                                sub_list_exists = True
                                break
                        if sub_list_exists:
                            break
                    if sub_list_exists:
                        self.vtx_to_nbors[tar].remove(y)

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

        # need to find all nbors
        self.find_PC()

        vertices = self.states_df.columns

        if vtx is None:
            tar_list = vertices
        else:
            tar_list = [vtx]

        self.vtx_to_MB = {}
        for tar in tar_list:
            self.vtx_to_MB[tar] = []
            super_MB_of_tar = set(self.vtx_to_nbors[tar])
            for x in self.vtx_to_nbors[tar]:
                super_MB_of_tar = super_MB_of_tar | set(self.vtx_to_nbors[x])
            super_MB_of_tar = list(super_MB_of_tar - {tar})
            for x in super_MB_of_tar:
                for y in self.vtx_to_nbors[tar]:
                    super_list = list(set(vertices) - {tar, x, y})
                    sub_list_exists = False
                    for combi_len in range(len(super_list)):
                        for sub_list in it.combinations(super_list, combi_len):
                            mi = DataEntropy.cond_mut_info(self.states_df,
                                             [tar], [x], list(sub_list))
                            if mi < self.alpha:
                                sub_list_exists = True
                                break
                        if sub_list_exists:
                            break
                    if sub_list_exists:
                        super_MB_of_tar.remove(x)
                        break
            self.vtx_to_MB[tar] = super_MB_of_tar


if __name__ == "__main__":
    MB_BasedLner.MB_lner_test(MB_hitonPC_Lner, verbose=True)
