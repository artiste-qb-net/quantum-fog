import operator

from learning.NetStrucLner import *
from shannon_info_theory.DataEntropy import *


class ChowLiuTreeLner(NetStrucLner):
    """
    ChowLiuTreeLnr (Chow Liu Tree Learner) is a subclass of NetStrucLner. It
    learns a Chow Liu tree from a dataframe states_df. See Wikipedia for a
    description of CL trees and original references.

    In a CL tree, the root node has several children and each of those can
    have children and so on. Each child has a single unisex parent.

    The arrows must branch out rather than merge because, by definition,
    the probability of each node (except the root nodes) of a CL tree must
    be of the form P(b|a) where b and a are single nodes. i.e., each node
    must have a single parent. This is not the case if two arrows merge at a
    node.

    Each arrow is assigned the empirical mutual information (MI) between the
    nodes it connects. The MI decreases as one moves away from the root node.

    Even if the input dataframe is generated from a tree, the learned tree
    structure will not necessarily look like the original tree.

    References
    ----------
    1. Nicholas Cullen neuroBN at github

    Attributes
    ----------

    """

    def __init__(self, states_df, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.bnet. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df

        Returns
        -------
        None

        """
        NetStrucLner.__init__(self, False,
                              states_df, vtx_to_states)
        self.learn_net_struc()

    def learn_net_struc(self):
        """
        Learns the net structure (i.e., it learns the children of each node)

        Returns
        -------
        None

        """
        nd_names = self.states_df.columns
        num_nds = len(nd_names)
        df = self.states_df

        def mi__(j, k):
            return DataEntropy.mut_info(df, [nd_names[j]], [nd_names[k]])

        mi_array = np.zeros((num_nds, num_nds), dtype=float) - 1.0
        ew_list = []
        for j in range(num_nds):
            for k in range(j+1, num_nds):
                w = mi__(j, k)
                mi_array[j, k] = w
                ew_list.append((j, k, w))
        # ew = (j, k, w) = edge j->k and weight

        ew_list = self.prune_ew_list(mi_array, ew_list)

        # sort by weight, highest first
        ew_list.sort(key=operator.itemgetter(2), reverse=True)
        temp_ew_list = []
        # start node_ids set with the left node of first edge in ew_list
        node_ids = [ew_list[0][0]]
        while True:
            for j, k, w in ew_list:
                j_is_old = (j in node_ids)
                k_is_old = (k in node_ids)
                if j_is_old and not k_is_old:
                    self.ord_nodes[j].add_child(self.ord_nodes[k])
                    node_ids.append(k)
                elif not j_is_old and k_is_old:
                    self.ord_nodes[k].add_child(self.ord_nodes[j])
                    node_ids.append(j)
                elif j_is_old and k_is_old:
                    self.do_if_both_nds_old(j, k, node_ids)
                else:  # neither is old
                    temp_ew_list.append((j, k, w))
            if not temp_ew_list:
                break
            # no need to sort temp_ew_list
            # as it is already in decreasing w order
            ew_list = temp_ew_list
            temp_ew_list = []
            # start a second tree with a new root node
            node_ids.append(ew_list[0][0])

    def prune_ew_list(self, mi_array, ew_list):
        """
        This function takes as input an ew_list and returns that list pruned
        (i.e., with some of its items removed). In this class, it does no
        pruning but for subclasses like AracneLner it does.


        Parameters
        ----------
        mi_array : numpy.array
            a square array with the mutual information of nodes i and j at
            position (i, j) of the array, with i < j.
        ew_list : list[tuple[int, int, float]]
            an edge-weight (ew) list. An ew is a 3-tuple ( i, j, weight)
            representing an arrow i->j for ints i, j denoting vertices,
            with weight w equal to the mutual info between the two endpoints
            i, j of the arrow.

        Returns
        -------
        list[tuple(int, int, float)]

        """
        return ew_list

    def do_if_both_nds_old(self, j, k, nd_ids):
        """
        This function processes the case when j and k are both old, i.e.,
        have been visited already. For this class, it does nothing, which is
        why this class learns only trees.

        Parameters
        ----------
        j : int
        k : int
        nd_ids : list[int]

        Returns
        -------
        None

        """
        pass

if __name__ == "__main__":
    from graphs.BayesNet import *
    from learning.RandGen_NetParams import *
    from learning.NetParamsLner import *
    from examples_cbnets.SimpleTree7nd import *

    is_quantum = False
    csv_path = 'training_data_c/SimpleTree7nd.csv'
    num_samples = 500
    bnet = SimpleTree7nd.build_bnet()
    gen = RandGen_NetParams(is_quantum, bnet, num_samples)
    gen.write_csv(csv_path)
    states_df = pd.read_csv(csv_path)
    lnr = ChowLiuTreeLner(states_df)
    lnr.bnet.draw(algo_num=2)
