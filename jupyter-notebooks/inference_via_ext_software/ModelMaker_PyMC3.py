from graphs.BayesNet import *
from nodes.BayesNode import *
from nodes.Node import *
import pprint as pp
import importlib
mm = importlib.import_module(
    "jupyter-notebooks.inference_via_ext_software.ModelMaker")


class ModelMaker_PyMC3:
    """
    This class has no constructor. All its methods are static. Each model
    writing method takes a BayesNet bnet as input and outputs a .py file
    containing a 'model' of bnet for PyMC3. PyMC3 is an external software
    package for doing "Deep Probabilistic Programming". The following
    parameters often arise in the functions of this class:

    file_prefix : str
        file prefix for .py file that will contain the model.
    bnet : BayesNet
        BayesNet object that is translated to model
    obs_vertices : 'all' | 'none' | None | list[str]
         list of node names (aka vertices). Can also be 'all' if all nodes
         are observed and 'none' or None if none are. If vtx is in this
         list, the categorical dist of the vtx has 'observed=data_' + vtx as
         one of its arguments. The user must define 'data_' + vtx before
         using this script. Nodes that are not observed are said to be
         'latent'. Latent nodes that are also root nodes are often called
         "parameters".

    """

    @staticmethod
    def write_model_for_inf(file_prefix, bnet, obs_vertices):
        """
        Writes a .py file containing a pymc3 (external software) 'model' for
        inference based on bnet. By inference we mean finding the prob dist
        of some nodes conditioned on the other nodes taking given values.

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        obs_vertices : 'all' | 'none' | None | list[str]

        Returns
        -------
        mod_file : str
            path to model .py file

        """
        model_name = 'mod'
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        obs_all = False
        if obs_vertices == 'all':
            obs_all = True
        obs_none = False
        if obs_vertices == 'none' or not obs_vertices:
            obs_none = True
            obs_vertices = []

        mod_file = file_prefix + "_pymc3.py"
        with open(mod_file, 'w') as f:
            f.write("import numpy as np\n")
            f.write("import theano as th\n")
            f.write("import theano.tensor as tt\n")
            f.write("import pymc3 as pm3\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                mm.ModelMaker.write_nd_names(bnet, f)

            if not obs_all and not obs_none:
                for vtx in nd_names_lex_ord:
                    if vtx in obs_vertices:
                        f.write("# data_" + vtx + ' = ' + '\n\n')
            elif obs_all:
                for vtx in nd_names_lex_ord:
                    f.write("# data_" + vtx + ' = ' + '\n\n')
            f.write('\n')
            f.write(model_name + ' = pm3.Model()\n')
            f.write("with " + model_name + ':\n')
            for vtx in nd_names_topo_ord:
                nd = bnet.get_node_named(vtx)
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
                            " = th.shared(arr_" +
                            vtx + ")[\n" + w8 +
                            pa_str + ', :]\n')
                obs_str = ''
                if obs_all or vtx in obs_vertices:
                    obs_str = ', observed=data_' + vtx
                f.write(w4 + vtx +
                        " = pm3.Categorical(\n" + w8 +
                        "'" + vtx + "', p=p_" +
                        vtx +
                        obs_str + ")\n\n")
        return mod_file

    @staticmethod
    def write_model_for_param_learning(file_prefix, bnet,
                                       obs_vertices):
        """
        Writes a .py file containing a pymc3 (external software) 'model' for
        parameter learning based on bnet. By parameter learning, we mean
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

        Parameters
        ----------
        file_prefix : str
        bnet : BayesNet
        obs_vertices : 'all' | 'none' | None | list[str]

        Returns
        -------
        mod_file : str
            path to model .py file

        """

        model_name = 'mod'
        w4 = '    '  # 4 white spaces
        w8 = w4 + w4
        assert obs_vertices
        mod_file = file_prefix + "_pymc3.py"
        with open(mod_file, 'w') as f:
            f.write("import numpy as np\n")
            f.write("import theano as th\n")
            f.write("import theano.tensor as tt\n")
            f.write("import pymc3 as pm3\n")
            f.write('\n\n')

            nd_names_lex_ord, nd_names_topo_ord =\
                mm.ModelMaker.write_nd_names(bnet, f)

            for vtx in nd_names_lex_ord:
                if vtx in obs_vertices:
                    f.write("# data_" + vtx + ' = ' + '\n\n')
            f.write('\n')
            f.write(model_name + ' = pm3.Model()\n')
            f.write("with " + model_name + ':\n')

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
                            ' = th.shared(arr_' +
                            vtx + ")\n")
                    ten_or_probs = "ten_"
                else:
                    array = np.ones_like(nd.potential.pot_arr)
                    f.write(w4 + 'alpha_' + vtx + ' = np.')
                    pp.pprint(array, f)
                    f.write(w4 + 'probs_' + vtx +
                            ' = pm3.Dirichlet(\n' + w8 +
                            "'probs_" + vtx +
                            "', a=alpha_" + vtx +
                            ", shape=" + str(array.shape) + ")\n")
                    ten_or_probs = 'probs_'
                if vtx in use_stack:
                    f.write(w4 + "p_" + vtx +
                            " = tt.stack([\n" + w8 +
                            ten_or_probs + vtx +
                            "[" + pa_str + ':]\n' + w8 +
                            'for j in range(' +
                            'sam_size)])\n')
                else:
                    f.write(w4 + "p_" + vtx +
                            " = " + ten_or_probs + vtx +
                            "[" + pa_str + ':]\n')
                obs_str = ''
                if vtx in obs_vertices:
                    obs_str = ", observed=data_" + vtx
                shape_str = ''
                if vtx in use_stack or vtx in obs_vertices:
                    shape_str = ', shape=sam_size'
                f.write(w4 + vtx +
                        " = pm3.Categorical(\n" + w8 +
                        "'" + vtx + "', p=p_" + vtx +
                        obs_str + shape_str + ")\n\n")
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

        file_prefix = prefix0 + "WetGrass_inf_obs_all"
        obs_vertices = 'all'
        ModelMaker_PyMC3.write_model_for_inf(file_prefix, bnet, obs_vertices)

        file_prefix = prefix0 + "WetGrass_inf_obs_CW"
        obs_vertices = ['Cloudy', "WetGrass"]
        ModelMaker_PyMC3.write_model_for_inf(file_prefix, bnet, obs_vertices)

        file_prefix = prefix0 + "WetGrass_par_obs_CW"
        obs_vertices = ['Cloudy', "WetGrass"]
        ModelMaker_PyMC3.write_model_for_param_learning(file_prefix, bnet,
                                                        obs_vertices)

        file_prefix = prefix0 + "WetGrass_par_obs_CRW"
        obs_vertices = ['Cloudy', "Rain",  "WetGrass"]
        ModelMaker_PyMC3.write_model_for_param_learning(file_prefix, bnet,
                                                        obs_vertices)
    main()

