import numpy as np
import itertools as it
import networkx as nx


class BifTool:

    """

    A .bif file is a popular and simple text file format for saving Bayesian
    Networks. There are several very helpful Bayesian Networks Repositories
    on the internet that collect Bnets in .bif and other formats. This is a
    simple stand-alone Python class from Quantum Fog that reads/writes a
    .bif file and loads it into convenient attributes. Different Python
    Bayesian network programs can access the attributes of this class to
    fill their own native attributes.

    This class can handle both real and complex valued CPT = Conditional
    Probability Distribution. real positive CPT for CBnets (
    is_quantum==False) and complex CPT for QBnets (is_quantum==True)

    As an example, here is the famous Asia network in bif format:

    network unknown {

    variable asia {
      type discrete [ 2 ] { yes, no };
    }
    variable tub {
      type discrete [ 2 ] { yes, no };
    }
    variable smoke {
      type discrete [ 2 ] { yes, no };
    }
    variable lung {
      type discrete [ 2 ] { yes, no };
    }
    variable bronc {
      type discrete [ 2 ] { yes, no };
    }
    variable either {
      type discrete [ 2 ] { yes, no };
    }
    variable xray {
      type discrete [ 2 ] { yes, no };
    }
    variable dysp {
      type discrete [ 2 ] { yes, no };
    }
    probability ( asia ) {
      table 0.01, 0.99;
    }
    probability ( tub | asia ) {
      (yes) 0.05, 0.95;
      (no) 0.01, 0.99;
    }
    probability ( smoke ) {
      table 0.5, 0.5;
    }
    probability ( lung | smoke ) {
      (yes) 0.1, 0.9;
      (no) 0.01, 0.99;
    }
    probability ( bronc | smoke ) {
      (yes) 0.6, 0.4;
      (no) 0.3, 0.7;
    }
    probability ( either | lung, tub ) {
      (yes, yes) 1.0, 0.0;
      (no, yes) 1.0, 0.0;
      (yes, no) 1.0, 0.0;
      (no, no) 0.0, 1.0;
    }
    probability ( xray | either ) {
      (yes) 0.98, 0.02;
      (no) 0.05, 0.95;
    }
    probability ( dysp | bronc, either ) {
      (yes, yes) 0.9, 0.1;
      (no, yes) 0.7, 0.3;
      (yes, no) 0.8, 0.2;
      (no, no) 0.1, 0.9;
    }
    }


    Attributes
    ----------
    is_quantum : bool
    nd_sizes : dict[str, int]
    parents : dict[str, list[str]]
    pot_arrays : dict[str, numpy.ndarray]
    states : dict[str, list[str]]

    """

    def __init__(self, is_quantum=False):
        """
        Constructor

        Parameters
        ----------
        is_quantum : bool

        Returns
        -------

        """

        self.is_quantum = is_quantum
        self.nd_sizes = {}
        self.states = {}
        self.parents = {}
        self.pot_arrays = {}

    def describe_yourself(self):
        """
        For debugging purposes

        Returns
        -------

        """
        print("\nBifTool attributes:")
        print("is_quantum= ", self.is_quantum, "\n")
        print(self.nd_sizes, "\n")
        print(self.states, "\n")
        print(self.parents, "\n")
        print(self.pot_arrays)

    def read_bif(self, path):
        """
        Reads a .bif file (really just a .txt file)

        Parameters
        ----------
        path : str

        Returns
        -------

        """

        def fix(in_str, bad_chs, sub):
            """
            This replaces in 'in_str' each character of 'bad_chs' by a 'sub'

            Parameters
            ----------
            in_str : str
            bad_chs : str
            sub : str

            Returns
            -------
            str

            """
            for c in bad_chs:
                in_str = in_str.replace(c, sub)
            return in_str

        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if 'variable' in line:
                    fix(line, "{", "")
                    split = line.split()
                    node = split[1]

                    new_split = fix(f.readline(), '[]{,};', ' ').split()
                    self.nd_sizes[node] = int(new_split[2])
                    self.states[node] = new_split[3:]
                elif 'probability' in line:
                    split = fix(line, "(|,){", ' ').split()
                    node = split[1]
                    if len(split) == 2:
                        parents = []
                    else:
                        parents = split[2:]
                    self.parents[node] = parents
                    num_parents = len(parents)
                    nd_size = self.nd_sizes[node]
                    parent_sizes = [self.nd_sizes[pa] for pa in parents]

                    if not self.is_quantum:
                        ty = np.float64
                    else:
                        ty = np.complex128

                    self.pot_arrays[node] = \
                        np.zeros(parent_sizes + [nd_size], dtype=ty)

                    if num_parents != 0:
                        x = (range(parent_sizes[k])
                             for k in range(num_parents))
                        generator = it.product(*x)
                    else:
                        generator = [0]
                    for index in generator:
                        new_line = fix(f.readline(), ')', ',')
                        new_line = fix(new_line, '(;', '')
                        # remove whitespace from beginning and end of new_line
                        new_line = new_line.strip()
                        if num_parents == 0:
                            # root nodes don't have parentheses enclosing
                            # state so replace first blank space by comma
                            new_line = new_line.replace(' ', ',', 1)
                        # now new_line is in proper comma separated form
                        new_split = new_line.split(',')[-nd_size:]
                        if not self.is_quantum:
                            pot_vals = list(map(float, new_split))
                        else:
                            pot_vals = list(map(complex, new_split))
                        if num_parents != 0:
                            padded_index = \
                                tuple(list(index) + [slice(None)])
                        else:
                            padded_index = slice(None)
                        self.pot_arrays[node][padded_index] = pot_vals
                if line == '':
                    # self.describe_yourself()
                    break

    def write_bif(self, path):
        """
        Writes a .bif file.


        Parameters
        ----------
        path : str

        Returns
        -------

        """

        # self.describe_yourself()
        with open(path, 'w') as f:
            f.write('network unknown {\n')
            f.write('\n')

            for node, nd_size in self.nd_sizes.items():
                f.write('variable ' + node + ' {\n')

                line = 'type discrete [ ' + str(nd_size) + ' ] { '
                for st in self.states[node]:
                    line += st + ", "
                line = line[:-2] + " };\n"
                f.write(line)
                f.write("}\n")

            for node in self.nd_sizes:
                line = 'probability ( ' + node + ' | '
                parents = self.parents[node]
                num_parents = len(parents)
                parent_sizes = [self.nd_sizes[pa] for pa in parents]
                pot_arr = self.pot_arrays[node]
                for pa in parents:
                    line += pa + ", "
                line = line[:-2] + ' ) {\n'
                f.write(line)

                if num_parents != 0:
                    x = (range(parent_sizes[k])
                         for k in range(num_parents))
                    generator = it.product(*x)
                else:
                    generator = [0]
                for index in generator:
                    line = "\t"
                    if num_parents != 0:
                        line += "("
                        for pa, st in dict(zip(parents, index)).items():
                            line += self.states[pa][st] + ", "
                    else:
                        line += "table, "

                    line = line[:-2]
                    if num_parents != 0:
                        line += ") "
                    else:
                        line += " "
                    if num_parents != 0:
                        padded_index = \
                            tuple(list(index) + [slice(None)])
                    else:
                        padded_index = slice(None)

                    # print("\n", node)
                    # print(parents)
                    # print(padded_index)
                    arr_str = np.array2string(pot_arr[padded_index],
                        precision=7, separator=',')
                    line += arr_str[1:-1]
                    line += ";\n"
                    f.write(line)
                f.write("}\n")
            f.write("}\n")

    def bif2dot(self, in_path, out_path):
        """
        This function reads a bif file and writes a dot (graphviz) file.

        Parameters
        ----------
        in_path : str
            path to input bif file
        out_path : str
            path to output dot file

        Returns
        -------
        None

        """

        self.read_bif(in_path)

        nx_graph = nx.DiGraph()
        vtx_list = self.parents.keys()
        for vtx in vtx_list:
            nx_graph.add_node(vtx)
            for pa_vtx in self.parents[vtx]:
                nx_graph.add_edge(pa_vtx, vtx)

        nx.nx_pydot.write_dot(nx_graph, out_path)

if __name__ == "__main__":
    in_path = "examples_cbnets/asia.bif"
    out_path = "examples_cbnets/asia_copy.bif"
    tool = BifTool()
    tool.read_bif(in_path)
    tool.write_bif(out_path)

    from graphs.BayesNet import *
    in_path = "examples_cbnets/WetGrass.bif"
    out_path = "examples_cbnets/WetGrass_test1.dot"
    bnet = BayesNet.read_bif(in_path, False)
    bnet.write_dot(out_path)

    # the function bif2dot() avoids calls to any QFog files except this one
    tool = BifTool()
    in_path = "examples_cbnets/WetGrass.bif"
    out_path = "examples_cbnets/WetGrass_test2.dot"
    tool.bif2dot(in_path, out_path)




