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


def connectNodes( node1, node2 ):
    node1.addNeighbor( node2 )
    node2.addNeighbor( node1 )


def getTree( forest, clique ):
    for tree in forest:
        if clique in tree.nodes:
            return tree


def addClique( cliqueList, clique ):
    add = 1
    for c in cliqueList:
        if c.contains( clique.nodes ):
            add = 0
            break

    if add:
        cliqueList.append( clique )

def unmark_all_nodes( graph ):
    for node in graph.nodes:
        node.visited = False


def missing_edges( node ):
    edges = list()
    for neighbor in node.neighbors:
        for otherNeighbor in node.neighbors:
            if not otherNeighbor == neighbor:
                if not otherNeighbor in neighbor.neighbors:
                    edges.append([neighbor, otherNeighbor])
    return edges


def generateArrayIndex( dimsToIter, axesToIter, constValues, constAxes ):
    if len( axesToIter ) == 0:
        return constValues
    totalNumAxes = len( axesToIter ) + len( constAxes )
    indexList = [array([]) for dim in range( totalNumAxes )]

    indexList = generateArrayIndexHelper( 0, array(dimsToIter), array(axesToIter), indexList )

    nIndices = product(array( dimsToIter ))
    for (val, axis) in zip( constValues, constAxes ):
        indexList[axis] = ones([nIndices]) * val

    return array(indexList)

def generateArrayIndexHelper( val, dims, axes, indexList ):
    #if we have iterated through all of the dimensions
    if len( dims ) == 0:
        return indexList

    #if we have iterated through all of the values
    if val == dims[0]:
        return indexList

    indexList[axes[0]] = concatenate( (indexList[axes[0]], ones([product(dims[1:])]) * val) )
    indexList = generateArrayIndexHelper( 0, dims[1:], axes[1:], indexList )
    return generateArrayIndexHelper( val+1, dims, axes, indexList )


def convertIndex( baseIndex, weights ):
    nAxes = len( baseIndex )
    nIndex = len( baseIndex[0] )
    return sum(baseIndex * reshape(repeat( weights, nIndex ), (nAxes, nIndex)), axis=0)


def generateArrayStrIndex( indices, axes, nDims ):
    indices = array(indices)
    axes = array(axes)
    tmp = zeros([nDims]) + -1
    if len(axes) > 0:
        tmp[axes] = indices
    #tmp = str( tmp ).replace( '-1', ':' )
    indexStr = "["
    for i in range(len(tmp) - 1):
        if tmp[i] == -1:
            indexStr += ':,'
        else:
            indexStr += str(tmp[i])
            indexStr += ','
    #now handle the last element
    if tmp[-1] == -1:
        indexStr += ':]'
    else:
        indexStr += str(tmp[-1])
        indexStr += ']'
    return indexStr

def flatIndex(indices, shape):
    assert(isinstance(indices, ndarray))
    assert(isinstance(shape, types.TupleType))
    flat = 0
    for i in range(len(indices)):
        flat += indices[i] * product(shape[i+1:])

    return flat


class InducedCluster:

    def __init__(self, node):
        self.node = node
        self.edges = missing_edges(self.node)
        self.nEdges = len(self.edges)
        self.weight = self.compute_weight()

    def __lt__( self, other ):
        #less than means that it is better (pick it first)
        if self.nEdges < other.nEdges:
            return True
        if self.nEdges == other.nEdges and self.weight < other.weight:
            return True
        return False

    def recompute( self ):
        self.edges = missing_edges(self.node)
        self.nEdges = len(self.edges)
        self.weight = self.compute_weight()

    def compute_weight( self ):
        return product(array( [node.size() for node in self.node.neighbors] + [self.node.size()] ))

class ClusterBinaryHeap:

        def __init__( self ):
                self.heap = []

        def insert( self, node ):
                iCluster = InducedCluster(node)
                self.heap.append(iCluster)
                self.heap.sort()

        def __iter__(self):
                return self

        def __next__(self):
                if len(self.heap) == 0:
                        raise StopIteration
                cluster = self.heap[0]
                del self.heap[0]
                #find the affected nodes
                tmpClusterList = []
                for node in cluster.node.neighbors:
                        for c in self.heap:
                                if c.node == node:
                                        c.node.neighbors.remove(cluster.node)
                                        tmpClusterList.append(c)
                                        break
                #recompute cluster score of effected clusters
                for c in tmpClusterList:
                        c.recompute()
                #reorder now that edges have changed
                self.heap.sort()
                return (cluster.node, cluster.edges)

        def hasNext( self ):
                return not len( heap ) == 0

