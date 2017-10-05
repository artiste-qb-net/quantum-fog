# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# from BayesNet import *
# noinspection PyUnresolvedReferences


class InferenceEngine:
    """
    This is the parent class of all inference engines.

    Attributes
    ----------
    bnet : BayesNet
    bnet_ord_nodes : list[BayesNode]
        list of nodes of bnet ordered alphabetically by node name
    is_quantum : bool
    verbose : bool

    """

    def __init__(self, bnet, verbose=False, is_quantum=False):
        """
        Constructor

        Parameters
        ----------
        bnet : BayesNet
        verbose : bool
        is_quantum : bool

        Returns
        -------
        None

        """
        self.bnet = bnet
        self.verbose = verbose
        self.is_quantum = is_quantum
        sorted_nd_names = sorted([nd.name for nd in self.bnet.nodes])
        self.bnet_ord_nodes = [self.bnet.get_node_named(name) for
                        name in sorted_nd_names]

    @staticmethod
    def print_annotated_story(annotated_story):
        """
        An annotated story is a dictionary that maps each node to its
        current state. This function prints an annotated story in
        alphabetical order of node names.

        Parameters
        ----------
        annotated_story : dict(BayesNode, int)

        Returns
        -------
        None

        """
        pairs = sorted([(node.name, str(annotated_story[node]))
                    for node in annotated_story.keys()])
        story_line = ""
        for x, y in pairs:
            story_line += x + "=" + y + ", "
        print(story_line[:-2])

if __name__ == "__main__":
    print(5)
