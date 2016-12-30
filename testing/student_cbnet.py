# noinspection PyUnresolvedReferences
import networkx as nx
# noinspection PyUnresolvedReferences
from graphs.Dag import *
# from nodes.BayesNode import *
# noinspection PyUnresolvedReferences
from Qubifer import *
# noinspection PyUnresolvedReferences
from potentials.DiscreteUniPot import *
# noinspection PyUnresolvedReferences
import pydotplus as pdp
import io
from scipy import misc

from graphs.BayesNet import *

from examples_cbnets.HuaDar import *
from examples_qbnets.QuWetGrass import *
from examples_cbnets.WetGrass import *
if __name__ == "__main__":
    path = '../examples_cbnets/student.bif'
    bnet = BayesNet.read_bif(path, False)

    for node in bnet.nodes:
        print("name: ", node.name)
        print("parents: ", [x.name for x in node.parents])
        print("children: ", [x.name for x in node.children])
        print("pot_arr: \n", node.potential.pot_arr)
        print("\n")


    bnet.draw(algo_num=1)

    path1 = '../examples_cbnets/student.dot'
    bnet.write_dot(path1)
#    st_graph=pdp.graphviz.graph_from_dot_data(path1)