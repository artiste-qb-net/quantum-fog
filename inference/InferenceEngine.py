# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# from BayesNet import *


class InferenceEngine:
    """
    This is the parent class of all inference engines.

    Attributes
    ----------
    bnet : BayesNet
    do_print : bool
    is_quantum : bool

    """

    def __init__(self, bnet, do_print=False, is_quantum=False):
        """
        Constructor

        Parameters
        ----------
        bnet : BayesNet
        do_print : bool
        is_quantum : bool

        Returns
        -------

        """
        self.bnet = bnet
        self.do_print = do_print
        self.is_quantum = is_quantum

    @staticmethod
    def print_annotated_story(annotated_story):
        """
        Prints in a pretty way an annotated story, which is a dictionary
        mapping all nodes to their current state.

        Parameters
        ----------
        annotated_story : dict(BayesNode, int)

        Returns
        -------

        """
        story_line = ""
        for node in annotated_story.keys():
            story_line += node.name + "="
            story_line += str(annotated_story[node]) + ", "
        print(story_line[:-2])


if __name__ == "__main__":
    print(5)
