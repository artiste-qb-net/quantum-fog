from graphs.BayesNet import *
import pandas as pd


class NetStrucLner:
    """
    Learning a Bayesian Network is usually done in two steps (1) learning
    the structure (i.e. the graph or skeleton) (2) learning the parameters (
    i.e., the pots).

    NetStrucLner (Net Structure Learner) is a super class for all classes
    that learn the structure of a net based on empirical data about states
    given in a pandas dataframe called states_df.

    NetParamsLner is a class for learning the parameters of a net (either a
    cbnet or qbnet).

    So far, Quantum Fog assumes that structure learning is all made, even in
    the quantum case, from state data representing incoherent measurements
    of all nodes. It's possible that in the future we will add some
    schemes for predicting graph structure in the quantum case that use some
    coherent measurements that yield both state and phase information.

    IMPORTANT: We will use the word 'vtx' = vertex to denote a node name and
    the word 'node' to denote a Node object.

    Attributes
    ----------
    bnet : BayesNet
        a BayesNet in which we store what is learned
    is_quantum : bool
        True for quantum bnets amd False for classical bnets
    ord_nodes : list[DirectedNode]
        a list of DirectedNode's named and in the same order as the column
        labels of self.states_df.
    states_df : pandas.DataFrame
        a Pandas DataFrame with training data. column = node and row =
        sample. Each row/sample gives the state of the col/node.

    """

    def __init__(self, is_quantum, states_df, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool
        states_df : pandas.DataFrame
        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.bnet. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df

        Returns
        -------

        """
        nd_names = states_df.columns
        ord_nodes = [BayesNode(k, nd_names[k])
                          for k in range(len(nd_names))]
        bnet = BayesNet(set(ord_nodes))
        self.is_quantum = is_quantum
        self.bnet = bnet
        self.states_df = states_df
        self.ord_nodes = ord_nodes
        if not vtx_to_states:
            bnet.learn_nd_state_names(states_df)
        else:
            bnet.import_nd_state_names(vtx_to_states)

    def fill_bnet_with_parents(self, vtx_to_parents):
        """
        Takes self.bnet with no arrows but named nodes a priori and fills it
        with arrows taken for a dictionary vtx_to_parents mapping vertices
        to a list of their parents

        Parameters
        ----------
        vtx_to_parents : dict[str, [str]]

        Returns
        -------

        """
        vertices = self.states_df.columns
        for vtx in vertices:
            nd = self.bnet.get_node_named(vtx)
            nd_parents = [self.bnet.get_node_named(pa_name)
                           for pa_name in vtx_to_parents[vtx]]
            nd.add_parents(nd_parents)

if __name__ == "__main__":
    print(5)
