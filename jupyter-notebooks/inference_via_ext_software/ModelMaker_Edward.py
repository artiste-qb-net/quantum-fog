from graphs.BayesNet import *
from nodes.BayesNode import *
from nodes.Node import *
import pprint as pp
import importlib
mm = importlib.import_module(
    "jupyter-notebooks.inference_via_ext_software.ModelMaker")


class ModelMaker_Edward:
    """
    This class has no constructor. All its methods are static. Each model
    writing method takes a BayesNet bnet as input and outputs a .py file
    containing a 'model' of bnet for TensorFlow/Edward. TensorFlow/Edward is
    an external software package for doing "Deep Probabilistic Programming".

    """

    @staticmethod
    def write_model_for_inf(file_prefix, bnet):
        """
        Writes a .py file containing an edward (external software) 'model'
        for inference based on bnet. By inference we mean finding the prob
        dist of some nodes conditioned on the other nodes taking given values.

        A node is a kind of random variable. We call the name (a str) of a
        node its vertex, vtx.  For all vtx, this function defines a query
        random variable called vtx + '_q'. The shape of this query random
        variable is [node_size] (node_size= the node's number of states)
        because we want to use it to approximate the node's prob dist for
        fixed evidentiary conditions.

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model

        Returns
        -------
        mod_file : str
            path to model .py file

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        mod_file = file_prefix + "_edward.py"
        with open(mod_file, 'w') as f:
            f.write("import numpy as np\n")
            f.write("import tensorflow as tf\n")
            f.write("import edward as ed\n")
            f.write("import edward.models as edm\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                mm.ModelMaker.write_nd_names(bnet, f)

            f.write("with tf.name_scope('model'):\n")

            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                pa_str = ''
                for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                    pa_str += pa_nd.name + ", "

                f.write(w4 + 'arr_' + vtx + ' = np.')
                pp.pprint(nd.potential.pot_arr, f)
                f.write(w4 + 'ten_' + vtx +
                        ' = tf.convert_to_tensor(arr_' +
                        vtx + ", dtype=tf.float32)\n")
                f.write(w4 + "p_" + vtx +
                        " = ten_" +
                        vtx + "[" +
                        pa_str + ':]\n')
                f.write(w4 + vtx +
                        " = edm.Categorical(\n" + w8 +
                        "probs=p_" + vtx +
                        ", name='" + vtx + "'" + ")\n\n")
            f.write("with tf.name_scope('posterior'):\n")
            for vtx in nd_names_lex_ord:
                nd = bnet.get_node_named(vtx)
                shape_str = '[' + str(nd.size) + ']'
                f.write(w4 + vtx + "_q" +
                        " = edm.Categorical(probs=tf.nn.softmax(\n" + w8 +
                        "tf.get_variable('p_" +
                        vtx + "_q', shape=" +
                        shape_str + ")),\n" +
                        w8 + "name='" + vtx + "_q')\n\n")
        return mod_file

    @staticmethod
    def write_model_for_param_learning(file_prefix, bnet,
            obs_vertices, inf_method='VA', propo_scale=.05):
        """
        Writes a .py file containing an edward (external software) 'model'
        for parameter learning based on bnet. By parameter learning, we mean
        learning the transition prob matrix of latent (i.e., hidden,
        unobserved) nodes based on data, i.e., multiple samples of observed
        nodes.

        The variable 'sam_size' (sample size of observed nodes) is used
        without value in the .py file and must be set to an int before
        running the .py file.

        A node is a kind of random variable. We call the name (a str) of a
        node its vertex, vtx. A node is observed iff its vtx is in the list
        obs_vertices. Observed nodes are observed sam_size times. A node
        that is not observed is called a latent node. Latent nodes that are
        also root nodes (have no parents) are often called "parameters".

        Iff vtx is not observed, and inf_method='VA', this function defines
        a query random variable called 'probs_' + vtx + '_q'. The shape of
        this query variable is the shape of the node's pot_arr because we
        want to use it to approximate the node's transition prob matrix.
        When inf_method='MC' instead of 'VA', the query random variable is
        called 'emp_probs_' + vtx + '_q' and it has shape [sam_size, pot_arr
        shape without parentheses].

        Iff vtx is observed, this function defines a placeholder random
        variable called vtx of shape=(sam_size, ).

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model
        obs_vertices : list[str]
            List of vertices that are observed sam_size times
        inf_method : str
            Inference method,
            'VA' for variational approximation method,
            'MC' for Monte Carlo
        propo_scale : float
            For inf_method='MC', proposal scale for all proposal functions.

        Returns
        -------
        mod_file : str
            path to model .py file

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        assert obs_vertices
        mod_file = file_prefix + "_edward.py"
        with open(mod_file, 'w') as f:
            f.write("import numpy as np\n")
            f.write("import tensorflow as tf\n")
            f.write("import edward as ed\n")
            f.write("import edward.models as edm\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                mm.ModelMaker.write_nd_names(bnet, f)

            f.write("with tf.name_scope('model'):\n")

            # list of vertices that use stack()
            # contains any vtx that is observed
            # or that has a parent that is observed
            use_stack = []
            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                if not nd.parents:
                    continue
                if vtx in obs_vertices:
                    use_stack.append(vtx)
                    continue
                else:
                    for pa_nd in nd.potential.ord_nodes[:-1]:
                        if pa_nd.name in obs_vertices:
                            use_stack.append(vtx)
                            continue

            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                if vtx in obs_vertices and not nd.parents:
                    f.write(w4 + vtx +
                            ' = tf.placeholder(tf.int32, shape=[' +
                            'sam_size],\n' +
                            w8 + 'name="' + vtx + '")\n\n')
                    continue
                pa_str = ''
                for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                    if pa_nd.name in use_stack:
                        pa_str += pa_nd.name + "[j], "
                    elif pa_nd.name in obs_vertices:
                        pa_str += pa_nd.name + "[j], "
                    else:
                        pa_str += pa_nd.name + ", "

                if vtx in obs_vertices:
                    array = nd.potential.pot_arr
                    f.write(w4 + 'arr_' + vtx + ' = np.')
                    pp.pprint(array, f)
                    f.write(w4 + 'ten_' + vtx +
                            ' = tf.convert_to_tensor(arr_' +
                            vtx + ", dtype=tf.float32)\n")
                    ten_or_probs = "ten_"
                else:
                    array = np.ones_like(nd.potential.pot_arr)
                    f.write(w4 + 'alpha_' + vtx + ' = np.')
                    pp.pprint(array, f)
                    f.write(w4 + 'probs_' + vtx +
                            ' = edm.Dirichlet(\n' + w8 +
                            "alpha_" + vtx + ".astype(np.float32)" +
                            ", name='probs_" + vtx + "')\n")
                    ten_or_probs = 'probs_'
                if vtx in use_stack:
                    f.write(w4 + "p_" + vtx +
                            " = tf.stack([\n" + w8 +
                            ten_or_probs + vtx +
                            "[" + pa_str + ':]\n' + w8 +
                            'for j in range(' +
                            'sam_size)])\n')
                else:
                    f.write(w4 + "p_" + vtx +
                            " = " + ten_or_probs + vtx +
                            "[" + pa_str + ':]\n')
                f.write(w4 + vtx +
                        " = edm.Categorical(\n" + w8 +
                        "probs=p_" + vtx +
                        ", name='" + vtx + "'" + ")\n\n")
            f.write("with tf.name_scope('posterior'):\n")
            for vtx in nd_names_lex_ord:
                nd = bnet.get_node_named(vtx)
                if vtx in obs_vertices:
                    if not nd.parents:
                        f.write(w4 + '# ' + vtx + ' = placeholder' + '\n\n')
                    else:
                        f.write(w4 + vtx +
                                '_ph = tf.placeholder(tf.int32, shape=[' +
                                'sam_size],\n' +
                                w8 + 'name="' + vtx + '_ph")\n\n')
                else:
                    shape_str = str(array.shape[:-1])
                    sam_shape_str = '(sam_size, ' + shape_str[1:]
                    if inf_method == 'VA':
                        f.write(w4 + 'probs_' + vtx + "_q" +
                                " = edm.Dirichlet(tf.nn.softplus(\n" + w8 +
                                "tf.get_variable('var_" +
                                vtx + "_q', shape=" +
                                shape_str + ")),\n" +
                                w8 + "name='probs_" + vtx + "_q')\n\n")
                    elif inf_method == 'MC':
                        f.write(w4 + 'emp_' + vtx + "_q" +
                                " = edm.Empirical(tf.nn.softmax(\n" + w8 +
                                "tf.get_variable('var_" +
                                vtx + "_q', shape=" +
                                sam_shape_str + ",\n" +
                                w8 + 'initializer=' +
                                'tf.constant_initializer(0.5))),\n' +
                                w8 + "name='emp_" + vtx + "_q')\n")
                        f.write(w4 + 'propo_' + vtx + "_q" +
                                " = edm.Normal(" +
                                'loc=emp_' + vtx +
                                "_q, scale=" + str(propo_scale) + ")\n\n")
                    else:
                        assert False, "Unexpected inference method."
        return mod_file

if __name__ == "__main__":
    def main():
        import os
        print('path1', os.getcwd())
        os.chdir('../../')
        print('path2', os.getcwd())

        in_path = "examples_cbnets/WetGrass.bif"
        bnet = BayesNet.read_bif(in_path, False)

        prefix0 = "jupyter-notebooks/" +\
                "inference_via_ext_software/model_examples_c/"

        file_prefix = prefix0 + "WetGrass_inf_obs_none"
        ModelMaker_Edward.write_model_for_inf(file_prefix, bnet)

        file_prefix = prefix0 + "WetGrass_par_VA_obs_CW"
        obs_vertices = ['Cloudy', "WetGrass"]
        ModelMaker_Edward.write_model_for_param_learning(file_prefix,
                bnet, obs_vertices)

        file_prefix = prefix0 + "WetGrass_par_MC_obs_CW"
        obs_vertices = ['Cloudy', "WetGrass"]
        ModelMaker_Edward.write_model_for_param_learning(file_prefix,
                bnet, obs_vertices, inf_method='MC')

        file_prefix = prefix0 + "WetGrass_par_VA_obs_CRW"
        obs_vertices = ['Cloudy', "Rain",  "WetGrass"]
        ModelMaker_Edward.write_model_for_param_learning(file_prefix,
                bnet, obs_vertices)
    main()

