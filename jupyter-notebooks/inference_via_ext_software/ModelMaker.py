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
    parameters in common:


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
        Writes a .py file containing a 'model' of bnet for external software
        X= PyMC3.

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
                           sample_size=None):
        """
        Writes a .py file containing a 'model' of bnet for external software
        X= edward.

        A node (node name is called vertex, vtx) is observed iff
        vtx_to_data.get(vtx) is nonempty. A node that is not observed is
        called a latent variable.

        This function doesn't use the value of each key-value pair in
        vtx_to_data except to ascertain whether the value is empty or not.

        This function defines: for each observed vtx, sample_size many
        random variables, and for each unobserved vtx, 1 random variable.

        For each vertex vtx,

            if vtx_to_data.get(vtx) is empty, this function defines a query
            random variable (ending in _q) of shape=(1,)

            if vtx_to_data.get(vtx) is nonempty, this function defines a
            placeholder random variable (ending in _ph) of shape=(
            sample_size,)

        When an observed variable is a parent of an unobserved one, all the
        values (sample_size of them) of the observed variable are reduced to
        a single value before they enter the unobserved node. This reduction
        is done via a function called domi() defined at the beginning of the
        model file.

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        vtx_to_data : dict[str, list[int]]
        sample_size : int|None
            sample size of observed nodes

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

            if vtx_to_data:
                f.write("# dominant, most common state\n")
                f.write("def domi(rv):\n")
                f.write(w4 + "return tf.argmax(tf.bincount(rv))\n\n")

            f.write("with tf.name_scope('model'):\n")

            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
                is_observed = bool(vtx_to_data.get(vtx))
                pa_str = ''
                for index, pa_nd in enumerate(nd.potential.ord_nodes[:-1]):
                    if is_observed:
                        if vtx_to_data.get(pa_nd.name):
                            pa_str += pa_nd.name + "[j], "
                        else:
                            pa_str += pa_nd.name + ", "
                    else:
                        if vtx_to_data.get(pa_nd.name):
                            pa_str += "domi(" + pa_nd.name + "), "
                        else:
                            pa_str += pa_nd.name + ", "

                f.write(w4 + 'arr_' + vtx + ' = np.')
                pp.pprint(nd.potential.pot_arr, f)
                f.write(w4 + 'ten_' + vtx +
                        ' = tf.convert_to_tensor(arr_' +
                        vtx + ", dtype=tf.float32)\n")
                if is_observed:
                    f.write(w4 + "p_" + vtx +
                            " = tf.stack([\n" + w8 +
                            "ten_" + vtx +
                            "[" + pa_str + ':]\n' + w8 +
                            'for j in range(' +
                            str(sample_size) + ')])\n')
                else:
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
                is_observed = bool(vtx_to_data.get(vtx))
                if is_observed:
                    f.write(w4 + vtx +
                            '_ph = tf.placeholder(tf.int32, shape=[' +
                            str(sample_size) + '],\n' +
                            w8 + 'name="' + vtx + '_ph")\n\n')
                else:
                    f.write(w4 + vtx + "_q" +
                            " = edm.Categorical(\n" + w8 +
                            "probs=tf.nn.softmax(tf.get_variable('" +
                            vtx + "_q/probs', shape=[" +
                            str(nd.size) + "])),\n" +
                            w8 + "name='" + vtx + "_q')\n\n")
    @staticmethod
    def batch_gen(arrays, batch_size):
        """
        (This function was copied from Edward Tutorials) If arrays =[array0,
        array1], it returns the list of batches, batches = [batch0, batch1],
        one batch for each array in `arrays'. batch0 is a subarray of array0
        with dimension along axis=0 equal to batch_size.

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
        for no_phs in [False, True]:
            if no_phs:
                file_prefix = "../examples_cbnets/WetGrass_no_phs"
                vtx_to_data = None
                sam_size = None
            else:
                file_prefix = "../examples_cbnets/WetGrass"
                vtx_to_data = {'Cloudy': [1], "WetGrass": [0, 1]}
                sam_size = 100
            # print(file_prefix, ':')
            ModelMaker.write_edward_model(file_prefix, bnet, vtx_to_data,
                                          sample_size= sam_size)

        file_prefix = "../examples_cbnets/WetGrass_CRW_obs"
        vtx_to_data = {'Cloudy': [0], "Rain": [0],  "WetGrass": [0]}
        sam_size = 100
        ModelMaker.write_edward_model(file_prefix, bnet, vtx_to_data,
                                              sample_size= sam_size)
    main()

