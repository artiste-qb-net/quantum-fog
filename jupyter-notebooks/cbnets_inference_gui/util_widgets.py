import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import ipywidgets as widgets

from graphs.BayesNet import *
from inference.JoinTreeEngine import *


def run_gui(bnet):
    """
    Generates and runs a widgets gui (graphical user interface) for doing
    inferences (more specifically, for displaying a probability bar plot for
    each node) conditioned on evidence entered by user into gui.

    Parameters
    ----------
    bnet : BayesNet

    Returns
    -------
    None

    """
    engine = JoinTreeEngine(bnet)

    node_list = list(bnet.nodes)
    num_nds = len(node_list)
    nd_names = sorted([nd.name for nd in node_list])
    # print(nd_names)

    display(widgets.Label(value="Active states for each node:"))

    active_wdg_list = []
    for vtx in nd_names:
        st_names = bnet.get_node_named(vtx).state_names
        st_names1 = ['All States'] + st_names
        sel_wdg = widgets.SelectMultiple(
                options=dict(zip(st_names1, range(-1, len(st_names)))),
                value=(-1,),
                description=vtx + ":"
            )
        active_wdg_list.append(sel_wdg)
        display(sel_wdg)
    # print(active_wdg_list)

    display((widgets.Label(value="Desired Node Prob Plots:")))
    plotted_nds_wdg = widgets.SelectMultiple(
                options=['All Nodes'] + nd_names,
                value=['All Nodes'],
            )
    display(plotted_nds_wdg)

    run_wdg = widgets.Button(
            description='Refresh Node Prob Plots',
        )
    run_wdg.layout.width = '40%'
    run_wdg.button_style = 'danger'
    display(run_wdg)

    # intialize each time run cell
    for nd in bnet.nodes:
        nd.active_states = range(nd.size)
    plotted_nds = engine.bnet_ord_nodes

    def active_wdg_do(title, change):
        nd = bnet.get_node_named(title)
        if -1 in change['new']:
            nd.active_states = range(nd.size)
        else:
            nd.active_states = list(change['new'])
        # print(title, change, nd.active_states)
    for active_wdg in active_wdg_list:
        title = active_wdg.description[:-1]
        # must store 'title' each time or all functions will use
        # value of 'title' at end of loop
        # Thanks to Jason Grout for pointing this out
        fun = (lambda x, title=title: active_wdg_do(title, x))
        active_wdg.observe(fun, names='value')

    def plotted_nds_wdg_do(change):
        # print("inside_plotted_do")
        # make plotted_nds global so changes get outside function
        global plotted_nds
        if 'All Nodes' in change['new']:
            plotted_nds = engine.bnet_ord_nodes
        else:
            plotted_nds = [bnet.get_node_named(name)
                           for name in sorted(list(change['new']))]
    plotted_nds_wdg.observe(plotted_nds_wdg_do, names='value')

    def single_pd(ax, node_name, pd_df):
            y_pos = np.arange(len(pd_df.index)) + .5
            plt.sca(ax)
            plt.yticks(y_pos, pd_df.index)
            ax.invert_yaxis()

            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xlim(0, 1)

            ax.grid(True)
            ax.set_title(node_name)
            ax.barh(y_pos, pd_df.values, align='center')

    def run_wdg_do(b):
        # clear_output()
        plt.close('all')
        num_ax = len(plotted_nds)
        fig, ax_list = plt.subplots(nrows=num_ax, ncols=1)
        if num_ax == 1:
            ax_list = [ax_list]
        jtree_pot_list = engine.get_unipot_list(plotted_nds)
        for k in range(num_ax):
            vtx = plotted_nds[k].name
            print(vtx)
            print('Active States:', list(plotted_nds[k].active_states))
            print(jtree_pot_list[k])
            print("\n")
            df = pd.DataFrame(jtree_pot_list[k].pot_arr,
                              index=plotted_nds[k].state_names)
            single_pd(ax_list[k], vtx, df)
        plt.tight_layout()
        plt.show()

    run_wdg.on_click(run_wdg_do)