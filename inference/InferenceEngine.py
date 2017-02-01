# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# from BayesNet import *
# noinspection PyUnresolvedReferences
from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring
from fractions import Fraction

class InferenceEngine:
    """
    This is the parent class of all inference engines.

    Attributes
    ----------
    bnet : BayesNet
    verbose : bool
    is_quantum : bool
    print_format : string

    """

    def __init__(self, bnet, verbose=False, is_quantum=False, print_format='text'):
        """
        Constructor

        Parameters
        ----------
        bnet : BayesNet
        verbose : bool
        is_quantum : bool
        print_format : string
        Returns
        -------

        """
        self.bnet = bnet
        self.verbose = verbose
        self.is_quantum = is_quantum
        self.print_format = print_format

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

    def annotated_story_table_content(table, pot_val, story_counter,
                        annotated_story, number_format='Float'):
        """
        Assembles a joint potential table row, i.e. story counter, annotated
        story and potential values, as table row in HTML mark-up.

        Parameters
        ----------
        table : HTML/XML table element
        pot_val: complex
            Potential value
        annotated_story : dict(BayesNode, int)
        number_format: str

        Returns
        -------
        None

        """
        new_row = SubElement(table, 'tr')
        new_cell1 = SubElement(new_row, 'td')
        new_cell1.text = str(story_counter)
        InferenceEngine.annotated_story_table_body(annotated_story, new_row)
        new_cell2 = SubElement(new_row, 'td')
        pot_val_str = InferenceEngine.formated_number_str(
                    pot_val,number_format)
        new_cell2.text = pot_val_str

    def annotated_story_table_body(annotated_story, HTML_table_row):
        """
        Returns an annotated story, which is a dictionary mapping all nodes
        to their current state, as table body in HTML mark-up.

        Parameters
        ----------
        annotated_story : dict(BayesNode, int)
        HTML_table_row : HTML/XML table row element

        Returns
        -------
        None

        """
        for node in annotated_story.keys():
            new_cell = SubElement(HTML_table_row, 'td')
            new_cell.text=str(annotated_story[node])

    def formated_number_str(num, num_format):
        """

        Parameters
        ----------
        num : float
        num_format : str

        Returns
        -------
        str

        """
        if num_format == 'Fraction':
            return str(Fraction.from_float(num).limit_denominator(100))
        if num_format == 'Percentage':
            return "{:.2%}".format(num)
        if num_format == 'Float':
            return str(num)
        else:
            return num_format.format(num)

if __name__ == "__main__":
    print(5)
