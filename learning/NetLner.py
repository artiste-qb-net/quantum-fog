import pandas as pd


class NetLner:
    """
    NetLner (Net Learner).  NetParamsLner and NetStructLner are both
    subclasses of this class. NetParamsLner is a class for learning the
    parameters (i.e., the pots) of a net (either a cbnet or qbnet), whereas
    NetStrucLner is a class for learning the structure (i.e., the plain
    graph) of a net.

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

    """

    def __init__(self, is_quantum, states_df, dag):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        states_df : pandas.DataFrame
        dag : Dag

        Returns
        -------

        """
        self.is_quantum = is_quantum
        self.dag = dag
        self.states_df = states_df
        self.ord_nodes = None
        self.learn_nd_state_names()

    def learn_nd_state_names(self):
        """
        Finds the unique names in each column of states_df and makes those
        the state names of the nodes of dag.

        Returns
        -------
        None

        """
        # will take state names of learned net to be in alphabetical order.
        # Only if state names of true net are in alphabetical order
        # too will they match
        for nd in self.dag.nodes:
            # must turn numpy array to list
            nd.state_names = sorted(list(pd.unique(self.states_df[nd.name])))
            nd.size = len(nd.state_names)
