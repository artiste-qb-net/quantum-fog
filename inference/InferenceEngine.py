# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

# from BayesNet import *
# noinspection PyUnresolvedReferences
import xml.etree.ElementTree as xet
from fractions import Fraction


class InferenceEngine:
    """
    This is the parent class of all inference engines.

    Attributes
    ----------
    bnet : BayesNet
    verbose : bool
    is_quantum : bool
    print_format : str
        verbose print format. Either 'text' or 'HTML'

    """

    def __init__(self, bnet, verbose=False, is_quantum=False,
                 print_format='text'):
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
        None

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

    @staticmethod
    def add_row_to_story_table(table, pot_val, story_counter,
                               annotated_story, number_format='Float'):
        """
        Assembles a joint potential table row, i.e. story counter, annotated
        story and potential values, as table row in HTML mark-up.

        Parameters
        ----------
        table : xet.Element
            HTML/XML table element
        pot_val: float|complex
            Potential value
        story_counter : int
        annotated_story : dict(BayesNode, int)
        number_format: str

        Returns
        -------
        None

        """
        new_row = xet.SubElement(table, 'tr')

        new_cell = xet.SubElement(new_row, 'td')
        new_cell.text = str(story_counter)

        for node in annotated_story.keys():
            new_cell = xet.SubElement(new_row, 'td')
            new_cell.text = str(annotated_story[node])

        new_cell = xet.SubElement(new_row, 'td')
        pot_val_str = InferenceEngine.formatted_number_str(
                    pot_val, number_format)
        new_cell.text = pot_val_str

    @staticmethod
    def formatted_number_str(num, num_format):
        """
        Returns formatted string for num

        Parameters
        ----------
        num : float|complex
        num_format : str

        Returns
        -------
        str

        """
        if num_format == 'Fraction':
            return str(Fraction.from_float(num).limit_denominator(100))
        elif num_format == 'Percentage':
            return "{:.2%}".format(num)
        elif num_format == 'Float':
            return str(num)
        else:
            return num_format.format(num)

if __name__ == "__main__":
    print(5)
