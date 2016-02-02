# PBNT: Python Bayes Network Toolbox
#
# Copyright (c) 2005, Elliot Cohen
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
# * The name "Elliot Cohen" may not be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from numpy import *
from pbnt.Node import *
from pbnt.Utilities import GraphUtilities
from pbnt.Utilities import Utilities
import pbnt.GraphExceptions
import copy

try: set
except NameError:
    import sets
    set = sets.Set

class Graph:
    """ Graph is the parent of all other graph classes.  It defines a very basic undirected graph class.  It essentially is just a list of nodes, and it is the nodes that maintain their own lists of parents and children.
    """

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def add_node(self, node):
        # Check if it is a list of nodes or a single node (arrays are also type=ListType).
        if isinstance(node, list):
            for n in node:
                self.nodes.add(n)
        else:
            self.nodes.add(node)

    def member_of(self, node):
        return node in self.nodes

    def contains(self, nodes):
        return self.nodes.issuperset(nodes)

    def connect_nodes(self, node1, node2):
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)


class DAG(Graph):
    """ Child of Graph class.  It is very similar to Graph class with the addition of a couple of methods aimed at a graph of nodes that are directed.  Currently this class does not ensure that it is acyclic, but it is assumed that the user will not violate this principle.
    """

    def __init__(self, nodes):
        Graph.__init__(self, nodes)
        self.numNodes = len(nodes)
        self.topological_sort()

    def topological_sort(self):
        # Orders nodes such that no node is before any of its parents.
        sorted = set()
        i = 0
        totIter = len(self.nodes)
        while len(self.nodes) > 0:
            if totIter <= 0:
                raise BadGraphStructure("Graph must be acyclic")
            totIter -= 1
            for node in self.nodes:
                if sorted.issuperset(node.parents):
                    sorted.add(node)
                    self.nodes.remove(node)
                    node.index = i
                    i += 1
                    break
        self.nodes = sorted

    def undirect(self):
        for node in self.nodes:
            node.undirect()

    def add_node(self, node):
        Graph.add_node(self, node)
        #FIXME: Could just insert in place rather than resorting entire list
        self.topological_sort()


class BayesNet(DAG):
    """  This is an actual Bayesian Network.  It is essentially a DAG, but it has several extra methods and fields that are used by associated inference and learning algorithms.
    """

    def __init__(self, nodes):
        DAG.__init__(self, nodes)

    def counts(self):
        # Return an array of matrices that can be used as a way to build a set of counts
        # This is also breaking down the abstraction barrier by accessing CPT,
        # should be investigated at a later date.
        return [copy.deepcopy(node.dist) for node in self.nodes]

    def add_counts(self, counts):
        # Update the internal CPTs with the given counts
        for node in self.nodes:
            count = counts[node.index]
            node.dist += count
            node.dist.normalize()

    def update_counts(self, counts, evidence):
        # Update the set of counts with the evidence
        for node in self.nodes:
            count = counts[node.index]
            evIndex = node.evidence_index()
            values = evidence[evIndex]
            index = count.generate_index(evIndex, range(count.nDims))
            count[index] += 1


class MoralGraph(Graph):
    """  A MoralGraph is an undirected graph that is built by connecting all of the parents of a directed graph and dropping the direction of the edges.
    """

    def __init__(self, DAG):
        Graph.__init__(self, DAG.nodes)
        #undirect the nodes
        for node in self.nodes:
            node.undirect()
        #connect all pairs of parents
        for node in self.nodes:
            parents = copy.copy(node.parents)
            for parent in parents:
                for otherParent in parents:
                    if parent != otherParent:
                        self.connect_nodes(parent, otherParent)

    def deep_copy_nodes(self):
        newNodes = list()
        for node in self.nodes:
            newNodes.append(copy.copy(node))
        for (newNode, node) in zip(newNodes, self.nodes):
            # Copy Parent List
            for parent in node.parents:
                newNode.add_parent(newNodes[parent.index])
            # Copy Children List
            for child in node.children:
                newNode.add_child(newNodes[child.index])
            for neighbor in node.neighbors:
                newNode.add_neighbor(newNodes[neighbor.index])
        return newNodes



class MoralDBNGraph(MoralGraph):
    """ This is not finished yet.  The plan is to use this class to create a MoralGraph for use in Dynamic Bayes Nets. The primary difference between doing JunctionTreeInference on a DBN from a static bayes net is that I have to ensure that the forward interface and the backward interface are both contained in a clique of the final join tree.  This can be ensured by making sure that all of the nodes in the two interfaces are connected.  For more details and a justification please see Kevin Murphy's dissertation.
    """

    def __init__(self, DBN):
        MoralGraph.__init__(DBN)
        # Make sure all nodes within the interface are connected
        for interface in [fInterfaceNodes, bInterfaceNodes]:
            #for each interface, connect all nodes within the interface
            for node in interface:
                # OPTIMIZE: we could figure out exactly which nodes to add
                node.neighborSet.add(interface)

class TriangleGraph(Graph):
    """ TriangleGraph is constructed from the MoralGraph.  It is the triangulated graph.  It is constructed by identifying clusters of nodes according to a given heuristic.  There are many heuristics that can be used, and in this implementation the heuristic is implemented in the ClusterBinaryHeap and can therefore be changed independent of this class.  The heap acts as a priority queue.  After the heap has been created, we remove nodes from the heap and use the information to create Cliques.  The Cliques are then added to the graph if they are not contained in a previous Clique.  TODO: Move addClique to this class from GraphUtilities.  Reimplement ClusterBinaryHeap as a built in python priority queue.
    """

    def __init__(self, moral):
        Graph.__init__(self, moral.nodes)
        heap = GraphUtilities.ClusterBinaryHeap()
        # Copy the graph so that we can destroy the copy as we insert it into heap.
        # Deep copy isn't working, need to trace down bug but for now use hack.
        for i, node in enumerate(moral.deep_copy_nodes()):
            heap.insert(node)
        inducedCliques = []
        nodes = list(self.nodes)
        # Want nodes in their index order
        nodes.sort
        for (node, edges) in heap:
            realnode = nodes[node.index]
            for edge in edges:
                # We need to make sure we reference the nodes in the actual graph,
                # not the copied ones that were inserted into the heap.
                node1 = nodes[edge[0].index]
                node2 = nodes[edge[1].index]
                self.connect_nodes(node1, node2)
            clique = Clique(realnode.neighbors.union([realnode]))
            # We only add clique to inducedCliques if is not contained in a previously added clique
            GraphUtilities.addClique(inducedCliques, clique)
        self.cliques = inducedCliques


class JoinTree(Graph):
    """ JoinTree is the final graph that is constructed for JunctionTree Inference.  To create the JoinTree, we first create a forest of n JoinTrees where each tree consists of a single clique (n is the number of cliques).  Then we create a list of all distinct pairs.  Then we insert a sepset between each pair of cliques.  Then we loop n - 1 times.  At each iteration, we choose the next best sepset according to some heuristic.  If we the two cliques connected to the sepset are on different trees, we join them into one larger tree.
    """

    #use constructor from Graph, will take either a single clique or a list of them
    def __init__(self, clique):
        if not isinstance(clique, list):
            clique = [clique]
        Graph.__init__(self, clique)
        self.initialized = False
        self.likelihoods = []

    def init_clique_potentials(self, variables):
        # We currently only handle one tree long forests.
        for v in variables:
            famV = v.parents.union([v])
            for clique in self.nodes:
                if clique.contains(famV):
                    v.clique = clique
                    clique.init_potential(v)
                    break
        self.initialized = True

    def merge(self, sepset, tree):
        cliqueX = sepset.cliqueX
        cliqueY = sepset.cliqueY
        cliqueX.add_neighbor(sepset, cliqueY)
        cliqueY.add_neighbor(sepset, cliqueX)
        for node in tree.nodes:
            self.add_node(node)

    def reinitialize(self, variables):
        for clique in self.nodes:
            clique.reinit_potential()
            # FIXME: the following optimizes each sepset twice, inefficient.
            for sepset in clique.sepsets:
                sepset.reinit_potential()
        self.init_clique_potentials(variables)

    def enter_evidence(self, evidence, nodes):
        """ For all nodes that are not blank, we want to make its family clique consistent with the evidence.  This can be done by setting all values of the clique that are consistent with the evidence to 1 and all other places to 0.
        """
        setNodes = evidence.set_nodes()
        for node in setNodes:
            clique = node.clique
            potentialMask = Potential(clique.potential.nodes, default=0)
            index = potentialMask.generate_index_node([evidence[node]], [node])
            potentialMask[index] = 1
            clique.potential *= potentialMask

class BadGraphStructure:
    """ An exception class used to denote a graph with a malformed structure.  It is currently used to throw an error when a DAG is cyclic.
    """
    def __init__(self, txt):
        self.txt = txt

    def __repr__(self):
        return txt

class BadTreeStructure(BadGraphStructure):
    """ An exception class used to indicate a bad junction tree structure.  It is currently used in Inference to signify when a junction tree is really a forest of trees, which is not an error, but it is not currently supported by this package therefore it is.
    """
    pass
