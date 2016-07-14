from learning.NetStrucLner import *
from graphs.Dag import *


class NaiveBayesLner(NetStrucLner):
    """
    NaiveBayesLner (Naive Bayes Learner). This class learns the Naive Bayes
    structure from an input states dataframe. This means it simply assumes a
    tree structure wherein arrows radiate from the specified target node to
    all other nodes mentioned in the list of columns of the dataframe.

    Although ``naive", this model is often sufficiently good for
    classification purposes. In such problems, the node names (columns of
    dataframe) are called ``features" and the states of the target node are
    called ``classes" of the ``classifier"

    Attributes
    ----------
    is_quantum : bool
        True for quantum bnets amd False for classical bnets
    dag : Dag
        a Dag (Directed Acyclic Graph) into which we load what is learned
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.
    ord_nodes : list[DirectedNode]
        a list of DirectedNode's named and in the same order as the column
        labels of self.states_df.
    tar_nd_name : str
        The name of the node that will assume the role of ``target" or
        center of the graph.

    """

    def __init__(self, states_df, tar_nd_name):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame
        tar_nd_name : str

        Returns
        -------

        """
        NetStrucLner.__init__(self, False, states_df)
        self.tar_nd_name = tar_nd_name
        self.learn_dag()

    def learn_dag(self):
        """
        Learns the dag (i.e., it learns the children of each node)

        Returns
        -------
        None

        """
        tar_nd = self.dag.get_node_named(self.tar_nd_name)
        tar_nd.add_children([nd for nd in self.ord_nodes if nd != tar_nd])

if __name__ == "__main__":

    csv_path = 'training_data_c\\simple_tree_7nd.csv'
    states_df = pd.read_csv(csv_path)
    lnr = NaiveBayesLner(states_df, 'a0')
    lnr.dag.draw(algo_num=2)
