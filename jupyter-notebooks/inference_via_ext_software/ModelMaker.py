from graphs.BayesNet import *
from nodes.BayesNode import *
from nodes.Node import *
import pprint as pp


class ModelMaker:
    """
    This class has no constructor. All its methods are static. Each model
    writing method takes a BayesNet bnet as input and outputs a .py file
    containing a 'model' of bnet for software X. X is an external software
    package for doing "Deep Probabilistic Programming", such as PyMC (a.k.a.
    PyMC2), PyMC3 and TensorFlow/Edward. All the methods have the following
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
    def write_pymc2_model(file_prefix, bnet, vtx_to_data=None,
                          unfilled=True):
        """
        Writes a .py file containing a 'model' of bnet for software X= PyMC
        (a.k.a PyMC2, the precursor of PyMC3).

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        unfilled : bool
            means same as for write_pymc3_model()

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
    def write_pymc3_model(file_prefix, bnet, vtx_to_data=None,
                          unfilled=True):
        """
        Writes a .py file containing a 'model' of bnet for software X= PyMC3.

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        unfilled : bool
            If True, the model has undefined (unfilled) variables for the
            observations data. These undefined variables must be defined
            prior to calling this function. If False, the observations data
            variables are defined at the beginning of the model .py file
            based on the info in the vtx_to_data input.

        Returns
        -------
        None

        """
        model_name = 'mod'
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
                            " = theano.shared(arr_" +
                            vtx + ")[\n" + w8 +
                            pa_str + ', :]\n')
                f.write(w4 + vtx +
                        " = pm3.Categorical(\n" + w8 +
                        "'" + vtx + "', p=p_" +
                        vtx + ', ' +
                        obs_str + ")\n\n")

    @staticmethod
    def write_edward_model(file_prefix, bnet, vtx_to_data=None,
                           no_placeholders=True):
        """
        Writes a .py file containing a 'model' of bnet for software X= edward.

        A node (node name is called vertex, vtx) is observed iff
        vtx_to_data.get(vtx) is nonempty. A node that is not observed is
        called a latent variable.

        This function doesn't use the value of each key-value pair in
        vtx_to_data except to ascertain whether the value is empty or not.

        This function defines a query random variable for all nodes although
        such variables should only be used for unobserved nodes.

        A placeholder is used for vtx iff vtx is a root node (it has no
        parents), and vtx is observed, and no_placeholders = False.

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        no_placeholders : bool
            If True, doesn't use Placeholder() for any node. If False,
            uses them for a node iff the node has no parents (is a root
            node) and is observed (so vtx_to_data for key=name of node,
            is not empty).

        Returns
        -------
        None

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        if not vtx_to_data:
            vtx_to_data = {}
        with open(file_prefix + "_edward.py", 'w') as f:
            f.write("import numpy as np\n")
            f.write("import tensorflow as tf\n")
            f.write("import edward as ed\n")
            f.write("import edward.models as edm\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                ModelMaker.write_nd_names(bnet, f)

            f.write("with tf.name_scope('model'):\n")
            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                is_observed = bool(vtx_to_data.get(vtx))
                is_root_nd = not bool(nd.parents)
                # print('vtx:', vtx)
                # print('data, is_obs', vtx_to_data.get(vtx), is_observed)
                # print('pa, is_root', nd.parents, is_root_nd)
                if is_root_nd and is_observed and not no_placeholders:
                    f.write(w4 + vtx +
                            ' = tf.placeholder(tf.float32, [None, ' +
                            str(nd.size) + '],\n' +
                            w8 + 'name="' + vtx + '")\n\n')
                else:
                    if not nd.parents:
                        f.write(w4 + 'arr_' + vtx + ' = np.')
                        pp.pprint(nd.potential.pot_arr, f)
                        f.write(w4 + "p_" + vtx +
                                " = tf.convert_to_tensor(arr_" +
                                vtx + ", dtype=tf.float32)\n")
                    else:
                        pa_str = ''
                        for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                            pa_str += pa_nd.name + ", "
                        pa_str = pa_str[:-2]

                        f.write(w4 + 'arr_' + vtx + ' = np.')
                        pp.pprint(nd.potential.pot_arr, f)
                        f.write(w4 + "p_" + vtx +
                                " = tf.convert_to_tensor(arr_" +
                                vtx + ", dtype=tf.float32)[\n" + w8 +
                                pa_str + ', :]\n')
                    f.write(w4 + vtx +
                            " = edm.Categorical(\n" + w8 +
                            "probs=p_" + vtx +
                            ", name='" + vtx + "')\n\n")
            f.write("with tf.name_scope('posterior'):\n")
            for vtx in nd_names_lex_ord:
                nd = bnet.get_node_named(vtx)
                f.write(w4 + vtx + "_q" +
                        " = edm.Categorical(\n" + w8 +
                        "probs=tf.nn.softmax(tf.get_variable('" +
                        vtx + "_q/probs', shape=[" +
                        str(nd.size) + "])),\n" +
                        w8 + "name='" + vtx + "_q')\n\n")

if __name__ == "__main__":
    def main():
        in_path = "../examples_cbnets/WetGrass.bif"
        bnet = BayesNet.read_bif(in_path, False)
        vtx_to_data = {'Cloudy': [1], "WetGrass": [0, 1]}
        for unfilled in [False, True]:
            if unfilled:
                file_prefix = "../examples_cbnets/WetGrass_unfilled"
            else:
                file_prefix = "../examples_cbnets/WetGrass"
            ModelMaker.write_pymc2_model(file_prefix, bnet,
                                         vtx_to_data, unfilled)
            ModelMaker.write_pymc3_model(file_prefix, bnet,
                                         vtx_to_data, unfilled)
        for no_phs in [False, True]:
            if no_phs:
                file_prefix = "../examples_cbnets/WetGrass_no_phs"
            else:
                file_prefix = "../examples_cbnets/WetGrass"
            # print(file_prefix, ':')
            ModelMaker.write_edward_model(file_prefix, bnet,
                                         vtx_to_data, no_phs)
    main()

