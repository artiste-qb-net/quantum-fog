# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import copy as cp

from graphs.Graph import *


class MoralGraph(Graph):
    """
    Constructing a MoralGraph from a BayesNet is a preliminary step required
    by JoinTreeEngine. A MoralGraph is an undirected graph that is
    constructed from a directed graph by connecting all of the parents and
    dropping the direction of the edges.

    Attributes
    ----------

    """

    def __init__(self, dag):
        """
        Constructor

        Parameters
        ----------
        dag : Dag

        Returns
        -------

        """
        nodes = cp.deepcopy(dag.nodes)
        Graph.__init__(self, nodes)
        # undirect the nodes
        for nd in self.nodes:
            nd.undirect()
        # connect all pairs of parents
        for nd in self.nodes:
            pa_list = list(nd.parents)
            for k in range(len(pa_list)):
                pa1 = pa_list[k]
                for pa2 in pa_list[k+1:]:
                    if not pa1.has_neighbor(pa2):
                        pa1.add_neighbor(pa2)

if __name__ == "__main__":
    from examples_cbnets.HuaDar import *
    bnet = HuaDar.build_bnet()
    mo = MoralGraph(bnet)
    mo.print_neighbors()


