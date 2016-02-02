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
from pbnt.Distribution import *
from pbnt.Utilities import Utilities

try: set
except NameError:
    import sets
    set = sets.Set

class Node:
    """ A Node is the basic element of a graph.  In its most basic form a graph is just a list of nodes.  A Node is a really just a list of neighbors.
    """
    def __init__(self, id, index=-1, name="anonymous"):
        # This defines a list of edges to other nodes in the graph.
        self.neighbors = set()
        self.visited = False
        self.id = id
        # The index of this node within the list of nodes in the overall graph.
        self.index = index
        # Optional name, most usefull for debugging purposes.
        self.name = name

    def __lt__(self, other):
        # Defines a < operator for this class, which allows for easily sorting a list of nodes.
        return self.index < other.index

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, right):
        return self.id == right.id

    def add_neighbor(self, node):
        """ Make node a neighbor if it is not alreadly.  This is a hack, we should be allowing self to be a neighbor of self in some graphs.  This should be enforced at the level of a graph, because that is where the type of the graph would disallow it.
        """
        if (not node in self.neighbors) and (not self == node):
            self.neighbors.add(node)

    def remove_neighbor(self, node):
        # Remove the node from the list of neighbors, effectively deleting that edge from
        # the graph.
        self.neighbors.remove(node)

    def is_neighbor(self, node):
        # Check if node is a member of neighbors.
        return node in self.neighbors

class DirectedNode(Node):
    """ This is the child class of Node.  Instead of mainting a list of neighbors, it maintains a list of parents and children.  Of course since it is the child of Node, it does technically have a list of neighbors (though it should remain empty).
    """
    def __init__(self, id, index=-1, name="anonymous"):
        Node.__init__(self, id, index, name)
        self.parents = set()
        self.children = set()

    def add_parent(self, parent):
        # Same as add_neighbor, but for parents of the node.
        if (not parent in self.parents) and (not self == parent):
            self.parents.add(parent)

    def add_child(self, child):
        # Same as add_parent but for children.
        if (not child in self.children) and (not self == child):
            self.children.add(child)

    def remove_parent(self, parent):
        # Same as remove_neighbor, but for parents.
        self.parents.remove(parent)

    def remove_child(self, child):
        # Same as remove_parent
        self.children.remove(child)

    def undirect(self):
        """ This drops the direction of self's edges.  This doesn't exactly destroy it since self still maintains lists of parents and children.  We could think of this as allowing us to treat self as both directed and undirected simply allowing it to be casted as one at one moment and the other at another moment.
        """
        self.neighbors = self.parents.union(self.children)

class BayesNode(DirectedNode):
    """ BayesNode is a child class of DirectedNode.  Essentially it is a DirectedNode with some added fields that make it more appropriate for a Bayesian Network, such as a field for a distribution and arrays of indices. The arrays are indices of its parents and children; that is the index of its neighbor within the overall bayes net.
    """
    #this is a node for a Bayesian Network, which is a directed node with some extra fields
    def __init__(self, id, numValues, index=-1, name="anonymous"):
        DirectedNode.__init__(self, id, index, name)
        self.numValues = numValues
        # value is the value that this node currently holds.  -1 is currently the "Blank" value, this feels dangerous.
        self.value = -1
        self.clique = -1

    def set_dist(self, dist):
        self.dist = dist

    def size(self):
        return self.numValues

    def __len__(self):
        return self.numValues

    def __copy__(self):
        return BayesNode(self.id, self.numValues, index=self.index, name=self.name)


class Clique(Node):
    """ Clique inherits from Node.  Clique's are clusters which act as a single node within a JoinTree. They are equivalent in JoinTrees to BayesNodes' in Bayesian Networks.  The main difference is that they have "potentials" instead of distributions.  Potentials are in effect the same as a conditional distribution, but unlike conditional distribtions, there isn't as clear a sense that the distribution is over one node and conditioned on a number of others.
    """

    def __init__(self, nodes):
        # Make the name of self the concatenation of the names of the input nodes.
        name = ''
        for node in nodes:
            name += node.name
        Node.__init__(self, name)
        self.nodes = set(nodes)
        # Between every Clique node is a sepset, so this should be as long as
        self.sepsets = set()
        # A Potential is like a conditional distribution, but the probabilities
        # are not explicitly conditioned on other probabilities.
        self.potential = Potential(self.nodes)

    def add_neighbor(self, sepset, node):
        Node.add_neighbor(self, node)
        self.sepsets.add(sepset)

    def init_potential(self, node):
        """ We want to satisfy the formula self.potential = self.potential*P(node|node.parents).
        """
        self.potential *= node.dist

    def reinit_potential( self ):
        self.potential = Potential(self.nodes)

    def contains(self, nodes):
        # Checks if all of nodes is contained in self.nodes
        return self.nodes.issuperset(nodes)


class Sepset(Node):
    """ Sepsets sit between Cliques in a join tree.  They represent the intersection of the variables in the two member Cliques.  They facilitate passing messages between the two cliques.
    """

    def __init__(self, id, cliqueX, cliqueY):
        Node.__init__(self, id)
        # Clique that is connected to one side of self
        self.cliqueX = cliqueX
        # Clique that is connected to the other side of self.
        self.cliqueY = cliqueY
        # The nodes that are in self
        self.nodes = cliqueX.nodes.intersection(cliqueY.nodes)
        # The mass of self (the number of nodes it relates to.
        self.mass = len(self.nodes)
        # The cost, used for breaking ties between mass.  The cost is equal to the
        # product of the node sizes of the nodes in cliqueX + cliqueY.
        costX = product(array([node.size() for node in cliqueX.nodes]))
        costY = product(array([node.size() for node in cliqueY.nodes]))
        self.cost = costX + costY
        self.neighbors = [cliqueX, cliqueY]
        self.potential = Potential(self.nodes)

    def __lt__(self, other):
        # This test is used to order nodes when deciding which sepset has highest priority.
        if self.mass > other.mass:
            return True
        if self.mass == other.mass and self.cost < other.cost:
            return True

        return False

    def reinit_potential(self):
        self.potential = Potential(self.nodes)

if __name__ == "__main__":
    x = [0,1,2,3,4]
    print(x[:-1])
      #Initialize the Cancer Bayes Network
    # s   p
    # \   /
    #   c
    # /   \
    # x   d


    numberOfNodes = 5

    #name the nodes
    pollution = 0
    smoker = 1
    cancer = 2
    xray = 3
    dyspnoea = 4

    pNode = BayesNode(0, 2, name="pollution")
    sNode = BayesNode(1, 2, name="smoker")
    cNode = BayesNode(2, 2, name="cancer")
    xNode = BayesNode(3, 2, name="xray")
    dNode = BayesNode(4, 2, name="dyspnoea")

    #pollution
    pNode.add_child(cNode)

    #smoker
    sNode.add_child(cNode)

    #cancer
    cNode.add_parent(pNode)
    cNode.add_parent(sNode)
    cNode.add_child(xNode)
    cNode.add_child(dNode)

    #xray
    xNode.add_parent(cNode)

    #dyspnoea
    dNode.add_parent(cNode)

    nodes = [pNode, sNode, cNode, xNode, dNode]
    print("index=", cNode.index)
    print("id=",cNode.id)
    print("name=",cNode.name)
    print("neigh=", cNode.neighbors)
    print("parents=", cNode.parents)
    print("children=", cNode.children)
    print("parents type", type(cNode.parents))

#create distributions
  #pollution distribution 
    pDistribution = DiscreteDistribution(pNode)
    index = pDistribution.generate_index([],[])
    pDistribution[index] = 0.1, 0.9
    pNode.set_dist(pDistribution)
    print("p index=", index)
    print("p dist=", pDistribution.table)

    
    #smoker distribution
    sDistribution = DiscreteDistribution(sNode)
    index = sDistribution.generate_index([],[])
    sDistribution[index] = 0.7, 0.3
    sNode.set_dist(sDistribution)
    print("s index=", index)
    print("s dist=", sDistribution.table)


    #cancer
    dist = zeros([pNode.size(), sNode.size(), cNode.size()], dtype=float32)
    # P = High -> 0, S = False -> 0    
    dist[0,0,] = [0.98,0.02]
    dist[0,1,] = [0.95, 0.05]
    dist[1,0,] = [0.999,0.001]
    dist[1,1,] = [0.97, 0.03]
    cDistribution = ConditionalDiscreteDistribution(nodes=[pNode, sNode, cNode], table=dist)
    cNode.set_dist(cDistribution)
    print("c dist=", dist)
    
    #xray
    dist = zeros([cNode.size(), xNode.size()], dtype=float32)
    dist[1,] = [0.1,0.9]
    dist[0,] = [0.8,0.2]
    xDistribution = ConditionalDiscreteDistribution(nodes=[cNode, xNode], table=dist)
    xNode.set_dist(xDistribution)
    print("x dist=", xDistribution.table)
    
    #dyspnoea
    dist = zeros([cNode.size(), dNode.size()], dtype=float32)
    dist[1,] = [0.35,0.65]
    dist[0,] = [0.7, 0.3]
    dDistribution = ConditionalDiscreteDistribution(nodes=[cNode, dNode], table=dist)
    dNode.set_dist(dDistribution)
    print("d dist=", dDistribution.table)
    
