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
    PyMC2), PyMC3 and TensorFlow/Edward.

    """

    @staticmethod
    def batch_gen(arrays, batch_size):
        """
        (This generator function was copied from Edward Tutorials) If arrays
        =[array0, array1, ...], it returns a list of batches, batches = [
        batch0, batch1, ...], one batch for each array in `arrays'. batch0
        is a subarray of array0 with dimension along axis=0 equal to
        batch_size.

        Parameters
        ----------
        arrays : list[np.ndarray]
        batch_size : int

        Returns
        -------
        list[np.ndarray]

        """
        starts = [0] * len(arrays)  # pointers to where we are in iteration
        while True:
            batches = []
            for i, array in enumerate(arrays):
                start = starts[i]
                stop = start + batch_size
                diff = stop - array.shape[0]
                if diff <= 0:
                    batch = array[start:stop]
                    starts[i] += batch_size
                else:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                batches.append(batch)
            yield batches

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
            file stream for writing that is returned by open()
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
        Writes a .py file containing a 'model' of bnet for external software
        X= PyMC (a.k.a PyMC2, the precursor of PyMC3).

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model

        vtx_to_data : dict[str, list[int]]
            means same as for write_pymc3_model()

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
        Writes a .py file containing a 'model' of bnet for external software
        X= PyMC3.

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model
        vtx_to_data : dict[str, list[int]]
            Dictionary mapping a node name (a.k.a vtx, vertex) to a list (
            repeats possible in this list) of states (given as 0 based ints)
            that were observed. The lists of states for each node is data
            from which one can derive an empirical distribution of the
            observations of that node. Nodes in this dict are "observed
            variables" or "data" whereas nodes not in this dict are "latent
            variables". Latent nodes that are also root nodes are often
            called "parameters".

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
    def write_edward_model_for_inf(file_prefix, bnet):
        """
        Writes a .py file containing an edward (external software) 'model'
        for inference based on bnet. By inference we mean finding the prob
        dist of some nodes conditioned on the other nodes taking given values.

        A node is a kind of random variable. We call the name (a str) of a
        node its vertex, vtx.  For all vtx, this function defines a query
        random variable called vtx + '_q'. The shape of this query random
        variable is [node_size] (node_size= the node's number of states)
        because we want to use it to approximate the node's prob dist for
        fixed conditions.

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model

        Returns
        -------
        None

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
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
                        " = edm.Categorical(\n" + w8 +
                        "probs=tf.nn.softmax(tf.get_variable('p_" +
                        vtx + "_q', shape=" +
                        shape_str + ")),\n" +
                        w8 + "name='" + vtx + "_q')\n\n")

    @staticmethod
    def write_edward_model_for_param_learning(file_prefix, bnet,
                                              obs_vertices):
        """
        Writes a .py file containing an edward (external software) 'model'
        for parameter learning based on bnet. By parameter learning, we mean
        learning the transition prob matrix of latent (i.e., unobserved)
        nodes based on data, i.e., multiple samples of observed nodes.

        The variable 'sam_size' (sample size of observed nodes) is used
        without value in the .py file and must be set to an int before
        running the .py file.

        A node is a kind of random variable. We call the name (a str) of a
        node its vertex, vtx. A node is observed iff it's in the list
        obs_vertices. Observed nodes are observed sam_size times. A node
        that is not observed is called a latent node. Latent nodes that are
        also root nodes (have no parents) are often called "parameters".

        Iff vtx is not observed, this function defines a query random
        variable called 'probs_' + vtx + '_q'. The shape of this query
        variable is the shape of the node's pot_arr because we want to use
        it to approximate the node's transition prob matrix.

        Iff vtx is observed, this function defines a placeholder random
        variable called vtx of shape=(sam_size, ).

        When an observed node is a parent of an unobserved one, all the
        values (sam_size of them) of the observed node are reduced to a
        single value before they enter the unobserved node. This reduction
        is done via a function called domi() defined at the beginning of the
        model file.

        Parameters
        ----------
        file_prefix : str
            file prefix for .py file that will contain the model.
        bnet : BayesNet
            BayesNet object that is translated to model
        obs_vertices : list[str]
            List of vertices that are observed sam_size times

        Returns
        -------
        None

        """
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        assert obs_vertices
        with open(file_prefix + "_edward.py", 'w') as f:
            f.write("import numpy as np\n")
            f.write("import tensorflow as tf\n")
            f.write("import edward as ed\n")
            f.write("import edward.models as edm\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                ModelMaker.write_nd_names(bnet, f)

            f.write("# dominant, most common state\n")
            f.write("def domi(rv):\n")
            f.write(w4 + "return tf.argmax(tf.bincount(rv))\n\n")

            f.write("with tf.name_scope('model'):\n")

            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                pa_str = ''
                if vtx in obs_vertices and not nd.parents:
                    f.write(w4 + vtx +
                            ' = tf.placeholder(tf.int32, shape=[' +
                            'sam_size],\n' +
                            w8 + 'name="' + vtx + '")\n\n')
                    continue
                for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                    if vtx in obs_vertices:
                        if pa_nd.name in obs_vertices:
                            pa_str += pa_nd.name + "[j], "
                        else:
                            pa_str += pa_nd.name + ", "
                    else:
                        if pa_nd.name in obs_vertices:
                            pa_str += "domi(" + pa_nd.name + "), "
                        else:
                            pa_str += pa_nd.name + ", "

                if vtx in obs_vertices:
                    array = nd.potential.pot_arr
                    f.write(w4 + 'arr_' + vtx + ' = np.')
                    pp.pprint(array, f)
                    f.write(w4 + 'ten_' + vtx +
                            ' = tf.convert_to_tensor(arr_' +
                            vtx + ", dtype=tf.float32)\n")
                    f.write(w4 + "p_" + vtx +
                            " = tf.stack([\n" + w8 +
                            "ten_" + vtx +
                            "[" + pa_str + ':]\n' + w8 +
                            'for j in range(' +
                            'sam_size)])\n')
                else:
                    array = np.ones_like(nd.potential.pot_arr)
                    f.write(w4 + 'alpha_' + vtx + ' = np.')
                    pp.pprint(array, f)
                    f.write(w4 + 'probs_' + vtx +
                            ' = edm.Dirichlet(\n' + w8 +
                            "alpha_" + vtx + ".astype(np.float32)" +
                            ", name='probs_" + vtx + "')\n")
                    f.write(w4 + "p_" + vtx +
                            " = probs_" +
                            vtx + "[" +
                            pa_str + ':]\n')
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
                    f.write(w4 + 'probs_' + vtx + "_q" +
                            " = edm.Dirichlet(\n" + w8 +
                            "tf.nn.softplus(tf.get_variable('pos_" +
                            vtx + "_q', shape=" +
                            shape_str + ")),\n" +
                            w8 + "name='probs_" + vtx + "_q')\n\n")

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
            ModelMaker.write_pymc2_model(file_prefix, bnet, vtx_to_data,
                                         unfilled)
            ModelMaker.write_pymc3_model(file_prefix, bnet, vtx_to_data,
                                         unfilled)

        file_prefix = "../examples_cbnets/WetGrass_obs_none"
        ModelMaker.write_edward_model_for_inf(file_prefix, bnet)

        file_prefix = "../examples_cbnets/WetGrass_obs_CW"
        obs_vertices = ['Cloudy', "WetGrass"]
        ModelMaker.write_edward_model_for_param_learning(file_prefix, bnet,
                obs_vertices)

        file_prefix = "../examples_cbnets/WetGrass_obs_CRW"
        obs_vertices = ['Cloudy', "Rain",  "WetGrass"]
        ModelMaker.write_edward_model_for_param_learning(file_prefix, bnet,
                obs_vertices)
    main()

