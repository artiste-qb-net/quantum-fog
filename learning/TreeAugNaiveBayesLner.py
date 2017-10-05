from learning.ChowLiuTreeLner import *


class TreeAugNaiveBayesLner(NetStrucLner):
    """
    TreeAugNaiveBayesLner (Tree Augmented Naive Bayes Learner) is a simple
    improvement of the Naive Bayes algorithm that combines Naive Bayes with
    Chow Liu Trees.

    Whereas in Naive Bayes, the only arrows are those emanating from a
    target node to all other nodes, TAN Bayes first builds a CL tree from
    all nodes except the target one, and then it adds arrows from the target
    node to all other nodes. The idea is that under Naive Bayes all
    non-target nodes are independent if the target is held fixed. In the TAN
    Bayes graph, the non-target nodes are given a chance to be correlated,
    even if the target is held fixed.

    Attributes
    ----------
    bnet : BayesNet
        a BayesNet in which we store what is learned
    ord_nodes : list[DirectedNode]
        a list of DirectedNode's named and in the same order as the column
        labels of self.states_df.
    tar_vtx : str
        target vertex. This node has arrows pointing to all other nodes.

    """

    def __init__(self, states_df, tar_vtx, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame
        tar_vtx : str
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
        self.tar_vtx = tar_vtx
        self.learn_net_struc()

    def learn_net_struc(self):
        """
        This function learns a graph structure (a hybrid of a Naive Bayes
        tree and a Chow Liu tree) from the data and stores what it learns in
        self.bnet.

        Returns
        -------
        None

        """

        nd_names = self.states_df.columns
        nd_names_sans_tar = list(set(nd_names) - {self.tar_vtx})
        num_nds = len(nd_names)
        df = self.states_df[nd_names_sans_tar]

        lnr = ChowLiuTreeLner(df)
        self.bnet = lnr.bnet
        tar_nd = DirectedNode(num_nds+1, self.tar_vtx)
        self.bnet.add_nodes({tar_nd})
        self.ord_nodes = [self.bnet.get_node_named(name) for name in nd_names]
        tar_nd.add_children([nd for nd in self.ord_nodes if nd != tar_nd])

if __name__ == "__main__":

    csv_path = 'training_data_c/SimpleTree7nd.csv'
    states_df = pd.read_csv(csv_path)
    tar_vtx = 'b0'
    lnr = TreeAugNaiveBayesLner(states_df, tar_vtx)
    lnr.bnet.draw(algo_num=2)

