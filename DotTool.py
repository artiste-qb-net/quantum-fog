import graphviz as gv
from IPython.display import display, Image
from PIL.Image import open as open_image
import matplotlib.pyplot as plt
import pydotplus as pdp
import networkx as nx
import os


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

        x = s.render("tempo123", format='png', view=False)
        os.remove("tempo123")
        if jupyter:
            display(Image(x))
        else:
            open_image("tempo123.png").show()

    @staticmethod
    def read_dot_file(dot_file_path):
        """
        Unfortunately, the networkx function for reading a dot file is broken.

        # does not understand dot statements like X->Y,Z;
            nx_graph = nx.nx_pydot.read_dot(dot_file_path)

        This function will read a dot file of a very basic form only. An
        example of the basic form is:

        dot = "digraph G {\n" \
                  "a->b;\n" \
                  "a->s;\n" \
                  "n->s,a,b;\n" \
                  "b->s;\n"\
                  "}"

        Parameters
        ----------
        dot_file_path: str

        Returns
        -------
        list, list

        """
        nodes = []
        edges = []
        with open(dot_file_path) as f:
            in_lines = f.readlines()
            for line in in_lines:
                if "->" in line:
                    split_list = line.split(sep="->")
                    # print("ffgg", split_list)
                    pa = split_list[0].strip()
                    if pa not in nodes:
                        nodes.append(pa)
                    ch_list = split_list[1].split(",")
                    ch_list = [x.strip().strip(";").strip() for x in ch_list]
                    # print("ffgg", pa)
                    # print("ffgg", ch_list)
                    for ch in ch_list:
                        edges.append((pa, ch))
                        if ch not in nodes:
                            nodes.append(ch)

        return nodes, edges

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
        # this does not understand dot statements like X->Y,Z;
        # nx_graph = nx.nx_pydot.read_dot(dot_file_path)

        nodes, edges = DotTool.read_dot_file(dot_file_path)
        g = nx.DiGraph()
        g.add_edges_from(edges)

        return g

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


if __name__ == "__main__":

    def main():
        dot = "digraph G {\n" \
              "a->b;\n" \
              "a->s;\n" \
              "n->s,a,b;\n" \
              "b->s\n" \
              "}"
        with open("tempo13.txt", "w") as file:
            file.write(dot)
        path = 'tempo13.txt'
        DotTool.draw(path, jupyter=False)
        nx_graph = DotTool.nx_graph_from_dot_file(path)
        nx.draw(nx_graph, with_labels=True, node_color='white')
        plt.show()
        DotTool.write_dot_file_from_nx_graph(nx_graph, "tempo9.dot")

    main()



