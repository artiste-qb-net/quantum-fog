from graphs.BayesNet import *
from nodes.BayesNode import *
from nodes.Node import *
import pprint as pp


class ModelMaker:
    """
    This class has no constructor. All its methods are static. It contains
    functions that arise in other, more specific ModelMaker classes.

    """

    @staticmethod
    def batch_gen(arrays, batch_size):
        """
        (This generator function was copied from Edward Tutorials) If arrays
        =[array0, array1, ...], it returns a list of batches, batches = [
        batch0, batch1, ...], one batch for each array in `arrays'. batch0
        is a subarray of array0 with dimension along axis=0 equal to
        batch_size.

        Parameters
        ----------
        arrays : list[np.ndarray]
        batch_size : int

        Returns
        -------
        list[np.ndarray]

        """
        starts = [0] * len(arrays)  # pointers to where we are in iteration
        while True:
            batches = []
            for i, array in enumerate(arrays):
                start = starts[i]
                stop = start + batch_size
                diff = stop - array.shape[0]
                if diff <= 0:
                    batch = array[start:stop]
                    starts[i] += batch_size
                else:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                batches.append(batch)
            yield batches

    @staticmethod
    def write_nd_names(bnet, f):
        """
        Writes to file stream f the list of node names of BayesNet bnet in
        lexicographic (alphabetic) and in topological (chronological) order.
        Also returns both of these node name lists.

        Parameters
        ----------
        bnet : BayesNet
        f : file stream
            file stream (for writing) that is returned by open()

        Returns
        -------
        list[str], list[str]

        """
        f.write("# node names in lexicographic (alphabetic) order\n")
        nd_names_lex_ord = sorted([nd.name for nd in bnet.nodes])
        f.write('nd_names_lex_ord = ' + str(nd_names_lex_ord) + '\n\n')

        f.write("# node names in topological (chronological) order\n")
        topo_indices = sorted([nd.topo_index for nd in bnet.nodes])
        nd_names_topo_ord = [bnet.get_node_with_topo_index(k).name for
                             k in topo_indices]
        f.write('nd_names_topo_ord = ' + str(nd_names_topo_ord) + '\n\n')

        return nd_names_lex_ord, nd_names_topo_ord

if __name__ == "__main__":
    def main():
        print(5)
    main()

