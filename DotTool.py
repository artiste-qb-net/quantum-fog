import graphviz as gv
from IPython.display import display, Image
from PIL.Image import open as open_image
import matplotlib.pyplot as plt
import pydotplus as pdp
import networkx as nx


class DotTool:
    """
    This class has no constructor or attributes. It stores static methods
    that help to deal with dot files (drawing them, turning them to networkx
    graphs, etc.)

    """

    @staticmethod
    def draw(dot_file_path, jupyter=True):
        """
        This method uses graphviz to draw the dot file located at
        dot_file_path. It creates a temporary file called tempo.png with a
        png of the dot file. If jupyter=True, it embeds the png in a jupyter
        notebook. If jupyter=False, it opens a window showing the png.

        Parameters
        ----------
        dot_file_path : str
        jupyter : bool


        Returns
        -------
        None

        """
        s = gv.Source.from_file(dot_file_path)

        # using display(s) will draw the graph but will not embed it
        # permanently in the notebook. To embed it permanently,
        # must generate temporary image file and use Image().
        # display(s)

        x = s.render("tempo", format='png', view=False)
        if jupyter:
            display(Image(x))
        else:
            open_image("tempo.png").show()

    @staticmethod
    def fix_dot_file(dot_file_path):
        """
        nx.nx_pydot.from_pydot(dot_file_path) does not understand if
        dot_file_path has lines such as "X->A,B;" where X has multiple
        children. This method writes a temporary dot file, and returns the
        path to it. Each line of dot_file_path with multiple children is
        replaced in the temporary dot file by multiple lines with only one
        children. For example, a line "X->A,B;" is replaced by two lines
        "X->A;" and "X->B;".

        Parameters
        ----------
        dot_file_path : str
            path to input dot file

        Returns
        -------
        str
            path to temporary, simplified dot file

        """
        out_dot_file = "tempo.dot"
        with open(dot_file_path) as f:
            in_lines = f.readlines()
        with open(out_dot_file, 'w') as f:
            for line in in_lines:
                if "->" not in line:
                    f.write(line)
                else:
                    split_list = line.split(sep="->")
                    # print("ffgg", split_list)
                    pa = split_list[0].strip()
                    ch_list = split_list[1].split(",")
                    ch_list = [x.strip().strip(";").strip() for x in ch_list]
                    # print("ffgg", pa)
                    # print("ffgg", ch_list)
                    for ch in ch_list:
                        f.write(pa + "->" + ch + ";\n")
        return out_dot_file

    @staticmethod
    def nx_graph_from_dot_file(dot_file_path):
        """
        This has the same input and output as nx.nx_pydot.from_pydot(
        dot_file_path), but it understands lines in dot_file_path with
        multiple children (e.g., X->A,B;)

        Parameters
        ----------
        dot_file_path

        Returns
        -------
        nx_graph : nx.DiGraph
            networkx graph. To plot it, use
            nx.draw(nx_graph, with_labels=True, node_color='white')
            plt.show()
        """
        dot_file_path = DotTool.fix_dot_file(dot_file_path)

        # st = str(gv.Source.from_file(dot_file_path))
        # print('dot_file_string\n', st)

        # this does not understand dot statements like X->Y,Z;
        nx_graph = nx.nx_pydot.read_dot(dot_file_path)
        # print("aasdd", list(nx_graph.edges()))
        # print("aasdd", list(nx_graph.nodes()))

        # this does not understand dot statements like X->Y,Z; either
        # pdp_graph = pdp.graph_from_dot_file(dot_file_path)
        # print("cccfff\n", pdp_graph.to_string())
        # print("xxxcv", [x.get_name() for x in pdp_graph.get_node_list()])
        # nx_graph = nx.nx_pydot.from_pydot(pdp_graph)

        return nx_graph

    @staticmethod
    def write_dot_file_from_nx_graph(nx_graph, dot_file_path):
        """
        This method takes as input an nx_graph and it writes a dot file from
        it. The output dot file's path is dot_file_path.

        Parameters
        ----------
        nx_graph : nx.DiGraph
        dot_file_path : str
            path of output dot file

        Returns
        -------
        None

        """
        nx.drawing.nx_pydot.write_dot(nx_graph, dot_file_path)

    def get_nx_graph(self):
        """
        This method returns an nx.DiGraph with the same structure as self.

        Returns
        -------
        nx.DiGraph

        """
        nx_graph = nx.DiGraph()
        for pa_nd in self.nodes:
            for ch_nd in pa_nd.children:
                nx_graph.add_edge(pa_nd.name, ch_nd.name)
        return nx_graph


if __name__ == "__main__":

    def main():
        path = 'dot_lib/good_bad_trols_G3.dot'
        DotTool.draw(path, jupyter=False)
        nx_graph = DotTool.nx_graph_from_dot_file(path)
        nx.draw(nx_graph, with_labels=True, node_color='white')
        plt.show()
        DotTool.write_dot_file_from_nx_graph(nx_graph, "tempo9.dot")

    main()



