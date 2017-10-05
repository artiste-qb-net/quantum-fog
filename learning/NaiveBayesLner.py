from learning.NetStrucLner import *
from graphs.BayesNet import *


class NaiveBayesLner(NetStrucLner):
    """
    NaiveBayesLner (Naive Bayes Learner). This class assumes a Naive Bayes
    structure without any regard for the data in states_df. This means it
    assumes a tree structure with arrows radiating from the specified target
    vertex to all other vertices mentioned in the list of column labels of
    states_df.

    Although ``naive", this model is often sufficiently good for
    classification purposes. In such problems, the non-target node names (
    column labels of dataframe) are called ``features" and the states of the
    target node are called ``classes" of the ``classifier"

    Attributes
    ----------
    tar_vtx : str
        The name of the node that will assume the role of ``target" or
        center of the graph.

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

        """
        NetStrucLner.__init__(self, False, states_df, vtx_to_states)
        self.tar_vtx = tar_vtx
        self.learn_net_struc()

    def learn_net_struc(self):
        """
        Stores in self.bnet the info that tar_vtx is parent of all other
        nodes.

        Returns
        -------
        None

        """
        tar_nd = self.bnet.get_node_named(self.tar_vtx)
        tar_nd.add_children([nd for nd in self.ord_nodes if nd != tar_nd])

if __name__ == "__main__":

    csv_path = 'training_data_c/SimpleTree7nd.csv'
    states_df = pd.read_csv(csv_path)
    lnr = NaiveBayesLner(states_df, 'a0')
    lnr.bnet.draw(algo_num=2)
