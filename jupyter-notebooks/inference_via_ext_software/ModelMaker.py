from graphs.BayesNet import *
from nodes.BayesNode import *
from nodes.Node import *
import pprint as pp


class ModelMaker:
    """
    This class has no constructor. All its methods are static. Each method
    takes a BayesNet bnet as input and outputs a .py file containing a
    'model' of bnet for software X. X is an external software package for
    doing "Deep Probabilistic Programming", such as PyMC (a.k.a. PyMC2),
    PyMC3 and TensorFlow/Edward. All the methods have the following
    parameters in common


    file_prefix : str
        file prefix for .py file that will contain the model.
    bnet : BayesNet
        BayesNet object that is translated to model
    vtx_to_data : dict[str, list[int]]
        Dictionary mapping a node name (a.k.a vtx, vertex) to a list (
        repeats possible in this list) of states (given as 0 based ints)
        that were observed. The lists of states for each node is data from
        which one can derive an empirical distribution of the observations
        of that node. Nodes in this dict are "observed variables" or "data"
        whereas nodes not in this dict are "latent variables". Frequentists
        like to call probabilistic latent variables "latent variables" and
        deterministic latent variables "parameters".
    unfilled : bool
        If True, the model has undefined (unfilled) variables for the
        observations data. These undefined variables must be defined prior
        to calling this function. If False, the observations data variables
        are defined at the beginning of the model .py file based on the info
        in the vtx_to_data input.

    """
    @staticmethod
    def write_nd_names(bnet, f):
        """
        Writes to file stream f the list of node names of BayesNet bnet in
        lexicographic (alphabetic) and in topological (chronological) order.
        Also returns both of these node name lists.

        Parameters
        ----------
        bnet : BayesNet
        f : file stream
            file stream for writing returned by open()
        Returns
        -------
        list[str], list[str]

        """
        f.write("# node names in lexicographic (alphabetic) order\n")
        nd_names_lex_ord = sorted([nd.name for nd in bnet.nodes])
        f.write('nd_names_lex_ord = ' + str(nd_names_lex_ord) + '\n\n')

        f.write("# node names in topological (chronological) order\n")
        topo_indices = sorted([nd.topo_index for nd in bnet.nodes])
        nd_names_topo_ord = [bnet.get_node_with_topo_index(k).name for
                             k in topo_indices]
        f.write('nd_names_topo_ord = ' + str(nd_names_topo_ord) + '\n\n')

        return nd_names_lex_ord, nd_names_topo_ord

    @staticmethod
    def write_pymc2_model(file_prefix, bnet,
                          vtx_to_data=None, unfilled=True):
        """
        Writes a .py file containing a 'model' of bnet for software X= PyMC
        (a.k.a PyMC2, the precursor of PyMC3).

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        unfilled : bool

        Returns
        -------
        None

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        if not vtx_to_data:
            vtx_to_data = {}
        with open(file_prefix + "_pymc2.py", 'w') as f:
            f.write("import numpy as np\n")
            f.write("import pymc as pm2\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                ModelMaker.write_nd_names(bnet, f)

            if not unfilled:
                for vtx in nd_names_lex_ord:
                    if vtx in vtx_to_data:
                        f.write('did_obs_' + vtx + ' = ' + 'True\n')
                        f.write("data_" + vtx + ' = ' +
                                str(vtx_to_data[vtx]) + '\n\n')
                    else:
                        f.write('did_obs_' + vtx + ' = ' + 'False\n')
                        f.write("data_" + vtx + ' = None' + '\n\n')
            else:
                for vtx in nd_names_lex_ord:
                    f.write('# did_obs_' + vtx + ' = ' + 'False\n')
                    f.write("# data_" + vtx + ' = None' + '\n\n')
            f.write('\n')

            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                obs_str = 'value=' + "data_" + vtx + \
                          ', observed=' + 'did_obs_' + vtx
                if not nd.parents:
                    f.write('p_' + vtx + ' = np.')
                    pp.pprint(nd.potential.pot_arr, f)
                else:
                    f.write('@pm2.deterministic\n')
                    pa_str = ''
                    k_str = ''
                    for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                        pa_str += pa_nd.name + "1=" +\
                                  pa_nd.name + ', '
                        k_str += 'k' + str(index) + ', '
                    pa_str = pa_str[:-2]
                    k_str = k_str[:-2]
                    f.write('def p_' + vtx + '(' + pa_str + '):\n')
                    for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                        # next step seems redundant
                        # but get error without it
                        f.write(w4 + pa_nd.name + "1 = " + pa_nd.name + '\n')

                        f.write(w4 + 'k' + str(index) + ' = ' +
                                'int(' + pa_nd.name + '1.value), ' + '\n\n')
                    f.write(w4 + 'arr = np.')
                    pp.pprint(nd.potential.pot_arr, f)
                    f.write(w4 + 'return arr[' + k_str + ', :]\n')
                f.write(vtx +
                        " = pm2.Categorical(\n" + w4 +
                        "'" + vtx + "', " +
                        "p=p_" + vtx + ',\n' + w4 +
                        obs_str +
                        ")\n\n\n")

    @staticmethod
    def write_pymc3_model(model_name, file_prefix, bnet,
                          vtx_to_data=None, unfilled=True):
        """
        Writes a .py file containing a 'model' of bnet for software X= PyMC3.

        Parameters
        ----------
        model_name : str
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        unfilled : bool

        Returns
        -------
        None

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        if not vtx_to_data:
            vtx_to_data = {}
        with open(file_prefix + "_pymc3.py", 'w') as f:
            f.write("import numpy as np\n")
            f.write("import pymc3 as pm3\n")
            f.write("import theano\n")
            f.write("import theano.tensor as tt\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                ModelMaker.write_nd_names(bnet, f)

            if not unfilled:
                for vtx in nd_names_lex_ord:
                    if vtx in vtx_to_data:
                        f.write("data_" + vtx + ' = ' +
                                str(vtx_to_data[vtx]) + '\n\n')
                    else:
                        f.write("data_" + vtx + ' = None' + '\n\n')
            else:
                for vtx in nd_names_lex_ord:
                    f.write("# data_" + vtx + ' = None' + '\n\n')
            f.write('\n')
            f.write(model_name + ' = pm3.Model()\n')
            f.write("with " + model_name + ':\n')
            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                obs_str = 'observed=' + "data_" + vtx
                p_str = "p_" + vtx
                if not nd.parents:
                    f.write(w4 + 'p_' + vtx + ' = np.')
                    pp.pprint(nd.potential.pot_arr, f)
                else:
                    pa_str = ''
                    for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                        pa_str += pa_nd.name + ", "
                    pa_str = pa_str[:-2]

                    f.write(w4 + 'arr_' + vtx + ' = np.')
                    pp.pprint(nd.potential.pot_arr, f)
                    f.write(w4 + "p_" + vtx +
                            " = tt.squeeze(\n" + w8 +
                            "theano.shared(" +
                            "arr_" + vtx + ")[" +
                            pa_str + ', :])\n')
                f.write(w4 + vtx +
                        " = pm3.Categorical(\n" + w8 +
                        "'" + vtx + "', p=" +
                        p_str + ', ' +
                        obs_str + ")\n\n")

if __name__ == "__main__":
    def main():
        in_path = "../examples_cbnets/WetGrass.bif"
        bnet = BayesNet.read_bif(in_path, False)
        for unfilled in [False, True]:
            if unfilled:
                file_prefix = "../examples_cbnets/WetGrass_unfilled"
                vtx_to_data = None
            else:
                file_prefix = "../examples_cbnets/WetGrass"
                vtx_to_data = {'Cloudy': [1], "WetGrass": [0, 1]}
            ModelMaker.write_pymc2_model(file_prefix, bnet,
                                         vtx_to_data, unfilled)
            ModelMaker.write_pymc3_model('mod', file_prefix, bnet,
                                         vtx_to_data, unfilled)
    main()

