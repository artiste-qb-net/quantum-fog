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


class EnumerationEngine(InferenceEngine):
    """
    EnumerationEngine is an InferenceEngine that enumerates all (brute force
    method) possible instantiations (aka stories, histories, Feynman paths)
    consistent with the active states of each node.

    Attributes
    ----------
    bnet_ord_nodes : list[BayesNode]
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
        print_format : str

        Returns
        -------
        None

        """
        InferenceEngine.__init__(self, bnet, verbose, is_quantum,
                                 print_format)
        self.bnet_ord_nodes = list(self.bnet.nodes)

    def get_unipot_list(self, node_list, null_events='All',
                        number_format='Float', normalize=False):
        """
        For each node in node_list, this method returns a uni-potential that
        gives the probabilities for the states of that node. Obviously,
        such a PD has the active states of the node as support. You can
        filter out null_events (zero probability events) by setting
        null_events='None'

        Parameters
        ----------
        node_list : list[BayesNode]
        null_events : str
            either 'All', 'Only' or 'None'
        number_format : str
        normalize : bool

        Returns
        -------
        list[DiscreteUniPot]

        """

        assert(set(node_list) <= self.bnet.nodes)
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
        story_counter = 0
        for cur_story in self.story_generator():
            story_counter += 1
            annotated_story = dict(zip(self.bnet_ord_nodes, cur_story))
            pot_val = self.get_story_potential_val(annotated_story)
            if not normalize:
                total_pot_val += pot_val

            state_list = [annotated_story[v] for v in node_list]
            for (pot, state) in zip(pot_list, state_list):
                pot[state] += pot_val

            if self.verbose:
                # this makes an ordered dictionary and
                # orders node names alphabetically
                annotated_story = OrderedDict(sorted(annotated_story.items(),
                           key=lambda item: item[0]))
                if self.print_format == 'text':
                    print("[", story_counter, "] pot_val=", pot_val)
                    InferenceEngine.print_annotated_story(annotated_story)
                    print("\n")
                elif self.print_format == 'HTML':
                    if story_counter == 1:
                        table = xet.Element('table')
                        header_row = xet.SubElement(table, 'tr')
                        header_list = []
                        header_list.append('Story #')
                        [header_list.append(n.name) for n
                                            in annotated_story.keys()]
                        header_list.append('Potential')
                        for h in header_list:
                            cell = xet.SubElement(header_row, 'th')
                            cell.text = h
                    if normalize:
                        pot_val /= total_pot_val

                    grow_table = (null_events == 'Only' and not pot_val) or\
                                 (null_events == 'None' and pot_val) or\
                                 (null_events == 'All')

                    if grow_table:
                        InferenceEngine.add_row_to_story_table(table, pot_val,
                            story_counter, annotated_story, number_format)

        if self.verbose:
            if self.print_format == 'text':
                print("tot_pot_val= ", total_pot_val,
                      "# equals 1 if you comment out the evidence")
                print("\n")
            elif self.print_format == 'HTML':
                if normalize:
                    total_pot_val = 1
                if null_events != 'Only':
                    total_row = xet.SubElement(table, 'tr')
                    total_cell1 = xet.SubElement(total_row, 'th')
                    total_cell2 = xet.SubElement(total_row, 'th')
                    total_cell1.set('colspan', str(len(self.bnet_ord_nodes)+1))
                    total_cell1.text = str('Potential Total:')
                    total_pot_val_str = InferenceEngine.formatted_number_str(
                        total_pot_val, number_format)
                    total_cell2.text = total_pot_val_str
                Dist_Table = HTML(xet.tostring(table).decode('UTF-8'))
                # print(xet.tostring(table).decode('UTF-8'))
                display(Dist_Table)

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
        Given an annotated story (i.e., a dictionary that maps all nodes to
        their current state), it returns a float for CBnets and a complex
        for QBnet. The returned value is the pot value for that particular
        annotated story.

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

    @staticmethod
    def plot_node_pd(ax, node_name, pd_df):
        """
        Plots probability distribution pd_df for the node named node_name

        Parameters
        ----------
        ax : plt.axes.Axes
        node_name : str
        pd_df : pd.DataFrame

        Returns
        -------
        None

        """
        y_pos = np.arange(len(pd_df.index)) + .5
        plt.sca(ax)
        plt.yticks(y_pos, pd_df.index)
        ax.invert_yaxis()

        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlim(0, 1)

        ax.grid(True)
        ax.set_title(node_name)
        ax.barh(y_pos, pd_df.values, align='center')

from examples_cbnets.HuaDar import *
if __name__ == "__main__":

    bnet = HuaDar.build_bnet()

    # introduce some evidence
    bnet.get_node_named("D").active_states = [0]
    bnet.get_node_named("G").active_states = [1]

    inf_eng = EnumerationEngine(bnet, verbose=True)
    id_nums = sorted([node.id_num for node in bnet.nodes])
    node_list = [bnet.get_node_with_id_num(k) for k in id_nums]

    # this is simpler but erratic
    # node_list = list(bnet.nodes)

    pot_list = inf_eng.get_unipot_list(node_list)
    for pot in pot_list:
        print(pot, "\n")

    from graphs.BayesNet import *

    path_bif = '../examples_cbnets/Monty_Hall.bif'
    bnet = BayesNet.read_bif(path_bif, False)
    bnet.get_node_named("1st_Choice").active_states = [0]
    bnet.get_node_named("Monty_Opens").active_states = [1]
    brute_eng = EnumerationEngine(bnet, verbose=True)
    pot_list = brute_eng.get_unipot_list(node_list)
    for pot in pot_list:
        print(pot, "\n")