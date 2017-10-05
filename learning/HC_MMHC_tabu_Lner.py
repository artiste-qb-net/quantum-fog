from learning.HillClimbingLner import *
from learning.MB_MMPC_Lner import *
from learning.HC_TabuLner import *
from learning.HC_RandRestartLner import *


class HC_MMHC_tabu_Lner(HC_TabuLner):
    """
    The class HC_MMHC_tabu_Lner (Hill Climbing Min-Max Hill Climbing Tabu
    Learner) is a child of HC_TabuLner. It adds to the latter a search at
    the beginning of the learning process of the PC (parents children,
    aka neighbors) set of each node. This knowledge is then used in the
    move_allowed() function to forbid any 'add' moves unless they add arrows
    compatible with the PC list of each node.

    References
    ----------

    1. Tsamardinos I, Brown LE, Aliferis CF (2006). "The Max-Min
    Hill-Climbing Bayesian Network Structure Learning Algorithm". Machine
    Learning, 65(1), 31-78.


    Attributes
    ----------
    alpha : float
        threshold used for deciding whether a conditional or unconditional
        mutual info is said to be close to zero (independence) or not (
        dependence). The error in a data entropy is on the order of ln(n+1)
        - ln(n) \approx 1/n where n is the number of samples so 5/n is a
        good default value for alpha.
    vtx_to_nbors : dict[str, list[str]]
        a dictionary mapping each vertex to a list of its neighbors. The
        literature also calls the set of neighbors of a vertex its PC (
        parents-children) set.

    """

    def __init__(self, states_df, score_type, max_num_mtries,
            alpha=0, tabu_len=10,
            ess=1.0, verbose=False, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        tabu_len : int
        alpha : float
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
        # this is a good default value
        self.alpha = 5/len(states_df.index)

        lner = MB_MMPC_Lner(states_df, alpha, verbose,
                vtx_to_states, learn_later=True)
        lner.find_PC()
        self.vtx_to_nbors = lner.vtx_to_nbors

        HC_TabuLner.__init__(
            self, states_df, score_type, max_num_mtries,
            tabu_len,
            ess, verbose, vtx_to_states)

    def move_approved(self, move):
        """
        Returns bool indicating whether move is approved. Only moves that
        are approved by parent class HC_TabuLner are approved. Of those,
        all 'del' and 'rev' moves are approved but 'add' moves approved only
        if they add an edge from a vertex to one of its neighbors. Neighbors
        are given by the dictionary vtx_to_nbors which is calculated before
        the hill climbing starts.

        Parameters
        ----------
        move : tuple[str, str, str]

        Returns
        -------
        bool

        """
        if not HC_TabuLner.move_approved(self, move):
            return False
        # print('inside move approved')
        # print(move)
        # print(self.vtx_to_nbors)
        if move[2] == 'add':
            if move[0] not in self.vtx_to_nbors[move[1]]:
                return False
        return True

if __name__ == "__main__":
    HillClimbingLner.HC_lner_test(HC_MMHC_tabu_Lner)
