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

#Python Library packages
import heapq

#Major Packages
from numpy import *
# import numpy.ieeespecial as ieee
import numpy.random as ra

#Local Project Modules
from __init__ import *
from pbnt.Graph import *
from pbnt.Node import *
from pbnt.Distribution import *
from pbnt.Utilities.Utilities import *
from pbnt.Utilities import GraphUtilities
import copy

"""This is the InferenceEngine module.  It defines all inference algorithms.  All of these inference algorithms are implemented as "engines", which means that they wrap around a bayes net in order to create a new inference object that can be treated abstractly.  One reason for this is that abstract inference objects can be used by other methods such as learning algorithms in the same ways regardless of which inference method is actually being used.
"""

class InferenceEngine:
    """ This is the parent class of all inference engines.  It defines several very basic methods that are used by all inference engines.
    """

    def __init__(self, bnet):
        self.bnet = bnet
        self.evidence = Evidence(zip(bnet.nodes, [-1]*len(bnet.nodes)))

    def marginal(self):
        self.action()


class EnumerationEngine(InferenceEngine):
    """ Enumeration Engine uses an unoptimized fully enumerate brute force method to compute the marginal of a query.  It also uses the standard constructor, init_evidence, and change_evidence methods.  In this engine, we use a hack.  We have to check and see if the variable is unobserved.  If it is not, then we know that the probability of that value is automatically 1.  We use this hack, because in order to do it properly, a table of likelihoods that incorporates the evidence would have to be constructed, this is very costly.
    """

    def marginal(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        # Compute the marginal for each node in nodes
        distList = list()
        for node in nodes:
            ns = node.size()
            # Create the return distribution.
            Q = DiscreteDistribution(node)
            if self.evidence[node] == BLANKEVIDENCE:
                 for val in range(ns):
                     prob = self.__enumerate_all(node, val)
                     index = Q.generate_index([val], range(Q.nDims))
                     Q[index] = prob
            else:
                val = self.evidence[node]
                index = Q.generate_index([val], range(Q.nDims))
                Q[index] = 1
            Q.normalize()
            distList.append(Q)
        return distList

    """ The following methods could be functions, but I made them private methods because the are functions that should only be used internally to the class. ADVICE: James, do you think these should remain as private methods or become function calls?
    """

    def __enumerate_all(self, node, value):
        """ We are going to iterate through all values of all non-evidence nodes. For each state of the evidence we sum the probability of that state by the probabilities of all other states.
        """
        oldValue = self.evidence[node]
        # Set the value of the query node to value, since we don't want to iterate over it.
        self.evidence[node] = value
        nonEvidence = self.evidence.empty()
        self.__initialize(nonEvidence)
        # Get the probability of the initial state of all nodes.
        prob = self.__probability(self.evidence)
        while self.__next_state(nonEvidence):
            prob += self.__probability(self.evidence)
        # Restore the state of evidence to its state at the beginning of enumerate_all.
        self.evidence[nonEvidence] = -1
        self.evidence[node] = oldValue
        return prob

    def __initialize(self, nonEvidence):
        self.evidence[nonEvidence] = 0

    def __next_state(self, nonEvidence):
        # Generate the next possible state of the evidence.
        for node in nonEvidence:
            if self.evidence[node] == (node.size() - 1):
                # If the value of the node is its max value, then reset it.
                if node == nonEvidence[-1]:
                    # If we iterated through to the last nonEvidence node, and didn't find a new
                    # value, then we have visited every possible state.
                    return False
                else:
                    self.evidence[node] = 0
                    continue
            else:
                self.evidence[node] += 1
                break
        return True

    def __probability(self, state):
        # Compute the probability of the state of the bayes net given the values of state.
        Q = 1
        for ev in state.items():
            node = ev[0]
            dist = node.dist
            # START HERE, MAYBE MAKE EVIDENCE ITS OWN STRUCTURE
            vals = state[dist.nodes]
            # Generate a slice object to index into dist using vals.
            index = dist.generate_index(vals, range(dist.nDims))
            Q *= node.dist[index]
        return Q

class MCMCEngine(InferenceEngine):
        #implemented as described in Russell and Norvig
        #X is a list of variables
        #N is thenumber of samples
        def marginal (self, X, N):
            if not isinstance(X, list):
                X = [X]
            Nx = [DiscreteDistribution(x) for x in X]
            queryIndex = array([x.index for x in X])
            state = copy(self.evidence)
            nonEvidence = state.empty()
            randMax = array([node.size() for node in nonEvidence])
            #ASSUMPTION: zero is the minimum value
            randMin = zeros([len(nonEvidence)])
            #initialize nonEvidence variables to random values
            state[nonEvidence] = ra.randint(randMin, randMax)
            for i in range(N):
                #record the value of all of the query variables
                # We start with a 100 sample cut as default
                if i > 100:
                    for (node, dist) in zip(X, Nx):
                        index = dist.generate_index([state[node]], range(dist.nDims))
                        dist[index] += 1
                        for node in nonEvidence:
                            val = self.sample_value_given_mb(node, state)
                            #change the state to reflect new value of given variable
                            if not state[node] == val:
                                state[node] = val
            for dist in Nx:
                dist.normalize()
            return Nx


        def sample_value_given_mb(self, node, state):
            MBval = DiscreteDistribution(node)
            children = node.children
            #want to save state
            oldVal = state[node]
            #OPTIMIZE: could vectorize this code
            for value in range(node.size()):
                state[node] = value
                values = state[node.dist.nodes]
                index = node.dist.generate_index(values, range(node.dist.nDims))
                MBindex = MBval.generate_index(value, range(MBval.nDims))
                MBval[MBindex] = node.dist[index]
                for child in children:
                    vals = state[child.dist.nodes]
                    index = child.dist.generate_index(vals, range(child.dist.nDims))
                    MBval[MBindex] *= child.dist[index]
            state[node] = oldVal
            MBval.normalize()
            val = MBval.sample()
            return val


class JunctionTreeEngine(InferenceEngine):
    """ This implementation of the Junction Tree inference algorithm comes from "Belief Networks: A Procedural Guide" By Cecil Huang an Adnan Darwiche (1996).  See also Kevin Murhpy's PhD Dissertation.  Roughly this algorithm decomposes the given bayes net to a moral graph, triangulates the moral graph, and collects it into cliques and joins the cliques into a join tree.  The marginal is then computed from the constructed join tree.
    """

    def __init__ (self, bnet):
        # Still use the built in constructor, but then add on to it
        InferenceEngine.__init__(self, bnet)
        # Create the moral graph
        moralGraph = MoralGraph(self.bnet)
        # Triangulate the graph
        triangulatedGraph = TriangleGraph( moralGraph )
        # Build a join tree and initialize it.
        self.joinTree = self.build_join_tree(triangulatedGraph)

    #def change_evidence(self, nodes, values):
        #""" Override parent's method because in a junction tree we have to perform an update or a retraction based on the changes to the evidence.
        #"""
        ## 0 = no change, 1 = update, 2 = retract
        #isChange = 0
        #changedNodes = []
        #for (node, value) in zip(nodes, values):
            ## Make sure node has actually changed
            #if not self.evidence[node.index] == value:
                #changedNodes += node
                ## Check if node is retracted
                #if not self.evidence[node.index] == -1:
                    #isChange = 2
                    #break
                #else:
                    #isChange = 1

        #if isChange == 1:
            ## Just to avoid import errors
            #assert(1 == 1)

            ## Do a global update
            #for node in changedNodes:
                ## Just to avoid import errors
                #assert(1 == 1)

                ## Update potential X and its likelihood with the new observation
                ## Then do global propagation (if only 1 cluster affected only have
                ## to distribute evidence.
        #elif isChange == 2:
            ## Do a global retraction: Encode the new likelihoods (and do observation entry),
            ## Reinitialize the join tree, do a Global propagation.

            ## Just to avoid import errors
            #assert(1 == 1)

    def marginal(self, query):
        # DELETE: When change_evidence is completed delete this.
        if not self.joinTree.initialized:
            self.joinTree.reinitialize(self.bnet.nodes)

        self.joinTree.enter_evidence(self.evidence, self.bnet.nodes)
        self.global_propagation()
        # DELETE: End delete here

        distributions = []
        if not isinstance(query, list):
            query = [query]
        for node in query:
            Q = DiscreteDistribution(node)
            for value in range(node.size()):
                potential = node.clique.potential
                index = potential.generate_index_node([value], [node])
                distIndex = Q.generate_index([value], range(Q.nDims))
                #FIXME: must be a better way to handle sum problem
                val = potential[index]
                if isinstance(val, ndarray):
                    val = val.sum()
                Q[distIndex] = val
            Q.normalize()
            distributions.append(Q)
        return distributions

    def global_propagation(self):
        self.joinTree.initialized = False
        # Arbitrarily pick a clique to be the root node, could be OPTIMIZED
        startClique = self.joinTree.nodes.pop()
        # Seems very awkward, but with sets there is no
        # way that I know of to get an element without popping it off.
        self.joinTree.nodes.add(startClique)
        GraphUtilities.unmark_all_nodes(self.joinTree)
        # We use 0 to denote that there was no prevCluster
        self.collect_evidence(0, startClique, 0, True)
        GraphUtilities.unmark_all_nodes(self.joinTree)
        self.distribute_evidence(startClique)

    def collect_evidence(self, prevClique, currentClique, sepset, isStart):
        # In this stage we send messages from the outer nodes toward a root node.
        currentClique.visited = 1
        for (neighbor, sep) in zip(currentClique.neighbors, currentClique.sepsets):
            # Do a DFS search of the tree, only visiting unvisited nodes
            if not neighbor.visited:
                self.collect_evidence(currentClique, neighbor, sep, 0)
        if not isStart:
            # After we have found the leaf (or iterated over all neighbors) send a message
            # back toward the root.
            self.pass_message(currentClique, prevClique, sepset)

    def distribute_evidence(self, clique):
        # Send messages from root node out toward leaf nodes
        clique.visited = 1
        for (neighbor, sep) in zip(clique.neighbors, clique.sepsets):
            # Perform DFS passing messages as we go from one node to the next
            if not neighbor.visited:
                self.pass_message(clique, neighbor, sep)
                self.distribute_evidence(neighbor)

    def pass_message(self, fromClique, toClique, sepset):
        # Project the fromCluster onto the sepset, oldSepsetPotential is the sepset's potential
        # before it is affected by the internals of project
        oldSepsetPotential = self.project(fromClique, sepset)
        # Absorb the sepset into the toCluster
        self.absorb(toClique, sepset, oldSepsetPotential)

    def project(self, clique, sepset):
        """ We want to project from the clique to the sepset.  We do this by marginalizing the clique potential into the sepset potential.
        """
        oldSepsetPotential = copy.deepcopy(sepset.potential)
        # OPTIMIZE: Write a new function that does this in place.
        sepset.potential = clique.potential.marginalize(sepset.potential)
        return oldSepsetPotential

    def absorb(self, clique, sepset, oldPotential):
        """ absorb divides the sepset's potential by the old potential.  The result is multiplied by the clique's potential.  Please see c. Huang and A. Darwiche 96.  As with project, this could be optimized by finding the best set of axes to iterate over (either the sepsets, or the clique's axes that are not in the sepset).  The best solution would be to define a multiplication operation on a Potential that hides the details.
        """
        #ABSTRACTION ERROR: We are breaking the abstraction layer here, but can't think of another way to do it without changing __div__
        # Wherever sepset.potential.table is 0, oldPotential is guaranteed to be, so fix it so that we don't divide by 0
        oldPotential[repr(sepset.potential.table == 0)] = 1
        sepset.potential /= oldPotential
        clique.potential *= sepset.potential

    def build_join_tree (self, triangulatedGraph):
        # The Triangulated Graph is really a graph of cliques.
        cliques = triangulatedGraph.cliques
        # We start by creating a forest of trees, one for each clique.
        forest = [JoinTree(clique) for clique in cliques]
        sepsetHeap = self.create_sepset_priority_queue(cliques)
        # Join n - 1 sepsets together forming (hopefully) a single tree.
        for n in range(len(forest) - 1):
            while sepsetHeap:
                sepset = heapq.heappop(sepsetHeap)
                # Find out which tree each clique is from
                joinTreeX = GraphUtilities.getTree(forest, sepset.cliqueX)
                joinTreeY = GraphUtilities.getTree(forest, sepset.cliqueY)
                if not joinTreeX == joinTreeY:
                    # If the cliques are on different trees, then join to make a larger one.
                    joinTreeX.merge(sepset, joinTreeY)
                    forest.remove(joinTreeY)
                    break
        if len(forest) > 1:
            raise BadTreeStructure("Inference on a forest of Junction Trees is not yet supported")
        else:
            tree = forest[0]
        tree.init_clique_potentials(self.bnet.nodes)
        return tree

    def create_sepset_priority_queue(self, cliques):
        """ Create a sepset (with a unique id) for every unique pair of cliques, and insert it into a priority queue.
        """
        sepsetHeap = []
        id = 0
        for i in range(len(cliques) - 1):
            for clique in cliques[i+1:]:
                sepset = Sepset(id, cliques[i], clique)
                id += 1
                heapq.heappush(sepsetHeap, sepset)
        return sepsetHeap


class JunctionTreeDBNEngine(JunctionTreeEngine):
    """ JunctionTreeDBNEngine is the JunctionTreeEngine for dynamic networks.  It is far from done.  This is more of a place holder as of right now.
    """

    def __init__(self, DBN):
        InferenceEngine.__init__(DBN)
        moral = MoralDBNGraph(DBN)
        #triangulate the graph
        triangulatedGraph = TriangleGraph( moralGraph )
        #build a join tree and initialize it
        self.joinTree = self.BuildJoinTree(triangulatedGraph)

