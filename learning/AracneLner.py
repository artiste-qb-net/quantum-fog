from learning.ChowLiuTreeLner import *
import itertools as it


class AracneLner(ChowLiuTreeLner):
    """
    The Aracne network was first used in Ref. 1. It is a simple variation of
    the old Chow Liu Tree. It differs from the CL tree in 2 ways. We
    implement those two differences by subclassing the ChowLiuTree class,
    but overriding two of its methods.

    As with CL trees, each edge is assigned the mutual info of its two
    endpoint nodes.

    Whereas CL trees use all edges, no matter how small their mutual info
    is, arcane considers all triangles with nodes as vertices, and throws
    away the edge of the triangle with the smallest mutual info. Then it
    points the arrows in the 2 remaining edges so that the 2 arrows point in
    the same direction, going from the edge with the larger to the edge with
    the smaller mutual info. If the vertices of the triangle are x, y, z,
    this is all consistent with the data processing inequality (see Cover
    Thomas book on Info theory) H(x : z) <= min [ H(x:y), H(y:z) ] for the
    bnet x->y->z

    Whereas CL trees are trees, arcane graphs can have loops. This is
    achieved by allowing connections between 2 nodes that have been used
    before, something CL trees do not allow.

    References
    ----------

    1. ARACNE: an algorithm for the reconstruction of gene regulatory
    networks in a mammalian cellular context, by  Margolin AA1, Nemenman I,
    Basso K, Wiggins C, Stolovitzky G, Dalla Favera R, Califano A. (BMC
    Bioinformatics. 2006 Mar 20;7 Suppl 1:S7)

    Attributes
    ----------

    """

    def __init__(self, states_df, vtx_to_states=None):
        """
        Constructor

        Parameters
        ----------
        states_df : pandas.DataFrame

        vtx_to_states : dict[str, list[str]]
            A dictionary mapping each node name to a list of its state names.
            This information will be stored in self.bnet. If
            vtx_to_states=None, constructor will learn vtx_to_states
            from states_df

        Returns
        -------
        None

        """
        ChowLiuTreeLner.__init__(self, states_df, vtx_to_states)

    def prune_ew_list(self, mi_array, ew_list):
        """
        This function takes as input an ew_list and returns that list pruned
        (i.e., with some of its items removed). Items are removed by
        considering vtx triangles i,j,k and removing one of the 3 edges of
        the triangle, the one with smallest mutual info.

        Parameters
        ----------
        mi_array : numpy.array
            a square array with the mutual information of nodes i and j at
            position (i, j) with i < j.
        ew_list : list[tuple(int, int, float)]
            an edge-weight (ew) list. An ew is a 3-tuple ( i, j, weight)
            representing an arrow i->j for ints i, j denoting vertices,
            with weight w equal to the mutual info between the two endpoints
            i, j of the arrow.

        Returns
        -------
        list[tuple(int, int, float)]

        """
        num_nds = len(self.states_df.columns)
        for i, j, k in it.combinations(range(num_nds), 3):
            x = [mi_array[i, j], mi_array[j, k], mi_array[k, i]]
            if x[0] <= min(x[1], x[2]):
                mi_array[i, j] = -1
                mi_array[j, i] = -1
            elif x[1] <= min(x[0], x[2]):
                mi_array[j, k] = -1
                mi_array[k, j] = -1
            elif x[2] <= min(x[0], x[1]):
                mi_array[k, i] = -1
                mi_array[i, k] = -1
        new_ew_list = []
        for j in range(num_nds):
            for k in range(j+1, num_nds):
                w = mi_array[j, k]
                if w > 0:
                    new_ew_list.append((j, k, w))
        return new_ew_list

    def do_if_both_nds_old(self, j, k, nd_ids):
        """
        This function processes the case when j and k are both old, i.e.,
        have been visited already.

        Parameters
        ----------
        j : int
            j, k are integers (corresponding to position of node in the list
            states_df.columns) for an arrow j->k
        k : int
        nd_ids : list[int]
            list of integers corresponding to previously visited vertices in
            the order in which they were visited. Since j and k are old (
            i.e., have been visited already), they are both already in
            nd_ids. We will draw an arrow from the first of j, k that was
            visited to the second.

        Returns
        -------
        None

        """
        if nd_ids.index(j) < nd_ids.index(k):
            older, younger = j, k
        else:
            older, younger = k, j
        self.ord_nodes[older].add_child(self.ord_nodes[younger])

if __name__ == "__main__":

    csv_path = 'training_data_c/SimpleTree7nd.csv'
    states_df = pd.read_csv(csv_path)
    lnr = AracneLner(states_df)
    lnr.bnet.draw(algo_num=1)

