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
    verbose : bool
    is_quantum : bool

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
        None

        """
        story_line = ""
        for node in annotated_story.keys():
            story_line += node.name + "="
            story_line += str(annotated_story[node]) + ", "
        print(story_line[:-2])

if __name__ == "__main__":
    print(5)
