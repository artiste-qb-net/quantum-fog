from learning.NetLner import *
from graphs.Dag import *


class NetStrucLner(NetLner):
    """
    NetStrucLner (Net Structure Learner) is a subclass of NetLner. All net
    structure learner classes have this class as parent. This class learns
    the structure (i.e. the graph or skeleton or dag) based on empirical
    data about states given in a pandas dataframe called states_df.

    So far, Quantum Fog assumes that structure learning is all made, even in
    the quantum case, from state data representing incoherent measurements
    of all nodes. It's possible that in the future we will add some
    schemes for predicting graph structure in the quantum case that use some
    coherent measurements that yield both state and phase information.

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

    def __init__(self, is_quantum, states_df):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        states_df : pandas.DataFrame

        Returns
        -------

        """
        nd_names = states_df.columns
        ord_nodes = [DirectedNode(k, nd_names[k])
                          for k in range(len(nd_names))]
        dag = Dag(set(ord_nodes))
        NetLner.__init__(self, is_quantum, states_df, dag)
        self.ord_nodes = ord_nodes