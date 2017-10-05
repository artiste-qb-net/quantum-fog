# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import itertools as it
from inference.InferenceEngine import *
from potentials.DiscreteUniPot import *
# noinspection PyUnresolvedReferences
import xml.etree.ElementTree as xet
from IPython.display import display, HTML
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Utilities as ut


class EnumerationEngine(InferenceEngine):
    """
    EnumerationEngine is an InferenceEngine that enumerates all possible
    instantiations (aka stories, histories, Feynman paths) consistent with
    the active states of each node. Thus it uses a brute force method.

    Attributes
    ----------

    """

    def __init__(self, bnet, verbose=False, is_quantum=False):
        """
        Constructor. Note that the constructor of every inference engine is
        designed so that one of its objects can be created just once at the
        beginning and then reused to calculate probabilities under several
        evidence assumptions.

        Parameters
        ----------
        bnet : BayesNet
        verbose : bool
        is_quantum : bool

        Returns
        -------
        None

        """
        InferenceEngine.__init__(self, bnet, verbose, is_quantum)

    @staticmethod
    def add_row_to_story_table(table, story_counter, pot_val,
                               story, pot_val_format='Float'):
        """
        Assembles a table row with story counter, story and potential value,
        this in HTML mark-up.

        Parameters
        ----------
        table : xet.Element
            HTML/XML table element
        story_counter : int
        pot_val: float|complex
            Potential value
        story : list[int]
        pot_val_format: str

        Returns
        -------
        None

        """
        new_row = xet.SubElement(table, 'tr')

        new_cell = xet.SubElement(new_row, 'td')
        new_cell.text = str(story_counter)

        for val in story:
            new_cell = xet.SubElement(new_row, 'td')
            new_cell.text = str(val)

        new_cell = xet.SubElement(new_row, 'td')
        pot_val_str = ut.formatted_number_str(pot_val, pot_val_format)
        new_cell.text = pot_val_str

    def get_unipot_list(self, node_list, normalize=False,
                        print_stories=False, print_format='text',
                        events='all', pot_val_format='Float'):
        """
        For each node in node_list, this method returns a uni-potential that
        gives the probabilities for the states of that node. Obviously,
        such a PD has the active states of the node as support.

        Calculating the unipot list with EnumerationEngine entails
        generating all possible stories. This is interesting info that you
        might want to look at it. You can print stories by setting
        print_stories=True. If you do, events='null' prints only null
        stories (zero probability stories), events='nonull' prints only
        nonull stories, and events='all' prints all stories.

        Parameters
        ----------
        node_list : list[BayesNode]
        normalize : bool
        print_stories : bool
        print_format : str
            Either 'text' or 'HTML'
        events : str
            either 'all', 'null' or 'nonull'
        pot_val_format : str

        Returns
        -------
        list[DiscreteUniPot]

        """

        assert set(node_list) <= self.bnet.nodes
        pot_list = [DiscreteUniPot(self.is_quantum, node, bias=0)
                    for node in node_list]

        total_pot_val = 0.
        if normalize:
            for cur_story in self.story_generator():
                annotated_story = dict(zip(
                    self.bnet_ord_nodes, cur_story))
                pot_val = self.get_story_potential_val(annotated_story)
                total_pot_val += pot_val

        table = None
        # print first row
        if print_stories and print_format == 'HTML':
            table = xet.Element('table')
            header_row = xet.SubElement(table, 'tr')
            header_list = ['Story #']
            for nd in self.bnet_ord_nodes:
                header_list.append(nd.name)
            header_list.append('Potential')
            for h in header_list:
                cell = xet.SubElement(header_row, 'th')
                cell.text = h

        story_counter = 0
        for cur_story in self.story_generator():
            story_counter += 1
            annotated_story = dict(zip(self.bnet_ord_nodes, cur_story))
            pot_val = self.get_story_potential_val(annotated_story)
            # if normalize=True, total_pot_val is already calculated,
            # so don't increment anymore
            if not normalize:
                total_pot_val += pot_val

            state_list = [annotated_story[nd] for nd in node_list]
            for (pot, state) in zip(pot_list, state_list):
                pot[state] += pot_val

            print_cur_story = (events == 'null' and abs(pot_val) < 1e-6) \
                or (events == 'nonull' and abs(pot_val) > 1e-6) \
                or (events == 'all')

            if print_stories and print_cur_story:
                if normalize:
                    pot_val /= total_pot_val
                if print_format == 'text':
                    print("[", story_counter, "] pot_val=", pot_val)
                    InferenceEngine.print_annotated_story(annotated_story)
                    print("\n")
                elif print_format == 'HTML':
                    EnumerationEngine.add_row_to_story_table(table,
                            story_counter, pot_val,
                            cur_story, pot_val_format)
        # print last row
        if print_stories:
            if normalize:
                total_pot_val = 1
            if print_format == 'text':
                print("tot_pot_val= ", total_pot_val,
                      "# equals 1 if no evidence and normalize=False")
                print("\n")
            elif print_format == 'HTML':
                if events != 'null':
                    total_row = xet.SubElement(table, 'tr')
                    total_cell1 = xet.SubElement(total_row, 'th')
                    total_cell2 = xet.SubElement(total_row, 'th')
                    total_cell1.set('colspan', str(len(self.bnet_ord_nodes)+1))
                    total_cell1.text = str('Potential Total:')
                    total_pot_val_str = ut.formatted_number_str(
                        total_pot_val, pot_val_format)
                    total_cell2.text = total_pot_val_str
                table_str = xet.tostring(table).decode('UTF-8')
                # print(xet.tostring(table).decode('UTF-8'))
                display(HTML(table_str))

        pot_list1 = []
        for pot in pot_list:
            pot.normalize_self()
            if self.is_quantum:
                pot1 = pot.get_probs_from_amps()
            else:
                pot1 = pot
            pot_list1.append(pot1)
        return pot_list1

    def story_generator(self):
        """
        Generate the next possible story constrained by the evidence (i.e.,
        by the active states of each node).

        Returns
        -------
        itertools.product

        """

        x = [node.active_states for node in self.bnet_ord_nodes]
        return it.product(*x)

    def get_story_potential_val(self, annotated_story):
        """
        Given an annotated story (i.e., a dictionary that maps each node to
        its current state), this function returns a float for CBnets and a
        complex for QBnet. The returned value is the pot value for that
        particular annotated story.

        Parameters
        ----------
        annotated_story : dict[Node, int]

        Returns
        -------
        complex

        """
        pot_val = 1
        for node in self.bnet.nodes:
            pot = node.potential
            states = tuple(annotated_story[v] for v in pot.ord_nodes)
            pot_val *= pot[states]
        return pot_val

if __name__ == "__main__":
    print("------------------------HuaDar")
    from examples_cbnets.HuaDar import *
    bnet = HuaDar.build_bnet()
    inf_eng = EnumerationEngine(bnet, verbose=True)

    # introduce some evidence after creating engine
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]

    pot_list = inf_eng.get_unipot_list(inf_eng.bnet_ord_nodes,
                                       print_stories=True)
    for pot in pot_list:
        print(pot, "\n")

    print("------------------------Monty_Hall")
    from graphs.BayesNet import *
    path_bif = '../examples_cbnets/Monty_Hall.bif'
    bnet = BayesNet.read_bif(path_bif, False)
    inf_eng = EnumerationEngine(bnet, verbose=True)

    # introduce some evidence after creating engine
    bnet.get_node_named("1st_Choice").active_states = [0]
    bnet.get_node_named("Monty_Opens").active_states = [1]

    pot_list = inf_eng.get_unipot_list(inf_eng.bnet_ord_nodes,
                                       print_stories=True)
    for pot in pot_list:
        print(pot, "\n")
