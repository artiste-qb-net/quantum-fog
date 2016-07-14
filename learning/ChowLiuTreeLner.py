from learning.NetStrucLner import *
from learning.DataEntropy import *
import operator


class ChowLiuTreeLner(NetStrucLner):
    """
    ChowLiuTreeLnr (Chow Liu Tree Learner) is a subclass of NetStrucLner. It
    associates a Chow Liu tree structure with an input states dataframe (
    see Wikipedia for a description of such trees and original references).
    In a CL tree, the root node has several children and each of those can
    have children and so on. Each child has a single unisex parent. Each
    arrow is assigned the empirical mutual information (MI) between the
    nodes it connects. The MI decreases as one moves away from the root node.

    Even if the input dataframe is generated from a Chow Liu tree,
    the learned tree structure will not necessarily look like the original
    tree.

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

    # Ref. Nicholas Cullen neuroBN at github

    def __init__(self, states_df):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame

        Returns
        -------

        """
        NetStrucLner.__init__(self, False, states_df)
        self.learn_dag()

    def learn_dag(self):
        """
        Learns the dag (i.e., it learns the children of each node)

        Returns
        -------
        None

        """
        nd_names = self.states_df.columns
        num_nds = len(nd_names)
        df = self.states_df

        def mi(j, k):
            return DataEntropy.mut_info(df, [nd_names[j]], [nd_names[k]])

        # ew = (j, k, w) = edge j->k and weight
        ew_list = [(j, k, mi(j, k)) for
                j in range(num_nds) for k in range(j+1, num_nds)]
        print('ew_list\n', ew_list)
        # sort by weight, highest first
        ew_list.sort(key=operator.itemgetter(2), reverse=True)
        # start node_ids set with the left node of first edge in ew_list
        node_ids = {ew_list[0][0]}

        for j, k, w in ew_list:
            if j in node_ids and k not in node_ids:
                self.ord_nodes[j].add_child(self.ord_nodes[k])
                node_ids.add(k)
            elif j not in node_ids and k in node_ids:
                self.ord_nodes[k].add_child(self.ord_nodes[j])
                node_ids.add(j)

if __name__ == "__main__":
    from graphs.BayesNet import *
    from learning.RandNetParamsGen import *
    from learning.NetParamsLner import *
    from examples_cbnets.SimpleTree7nd import *

    is_quantum = False
    csv_path = 'training_data_c\\simple_tree_7nd.csv'
    num_samples = 500
    bnet = SimpleTree7nd.build_bnet()
    gen = RandNetParamsGen(is_quantum, bnet, num_samples)
    gen.write_csv(csv_path)
    states_df = pd.read_csv(csv_path)
    lnr = ChowLiuTreeLner(states_df)
    lnr.dag.draw(algo_num=2)

    bnet_emp = SimpleTree7nd.build_bnet()
    # forget pots of emp=empirical bnet because we want to learn them
    for nd in bnet_emp.nodes:
        nd.potential = None

    lnr = NetParamsLner(is_quantum, bnet_emp, states_df)
    lnr.learn_all_bnet_pots()
    for nd in bnet.nodes:
        print('\nnode=', nd.name)
        print('true:')
        print(nd.potential)
        print('empirical:')
        print(bnet_emp.get_node_named(nd.name).potential)
