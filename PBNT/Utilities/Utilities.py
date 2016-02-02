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

from __future__ import generators
from numpy import *
import numpy.random as ra
from pbnt.__init__ import BLANKEVIDENCE

from __init__ import *

# Miscellaneous utility functions for use with the rest of the BayesNet Package



class Evidence(dict):
    """ This is the data structure for evidence.  It acts exactly like a dictionary except that it will take lists of keys with the [] notation, rather than just single keys.
    """
    def __setitem__(self, keys, values):
        if not isinstance(keys, list):
            keys = [keys]
            values = [values]
        elif (not isinstance(values, list)) and (not isinstance(values, ndarray)):
            values = [values]*len(keys)
            items = zip(keys,values)
        self.update(items)

    def __getitem__(self, keys):
        if isinstance(keys, list):
            values = []
            for key in keys:
                values.append(self.get(key))
        else:
            values = self.get(keys)
        return values

    def empty(self):
        nonEvidence = []
        for item in self.items():
            if item[1] == BLANKEVIDENCE:
                nonEvidence.append(item[0])
        return nonEvidence

    def set_nodes(self):
        ev = []
        for item in self.items():
            if item[1] != BLANKEVIDENCE:
                ev.append(item[0])
        return ev

    def __copy__(self):
        new = Evidence()
        new.update(self.items())
        return new
        
def myFloatEQ ( a , b ):
# Checks if a and b are equal within a fraction of error.
# The error is that normally introduced by the imprecision of
# Floating point numbers
# now outdated, use allclose

    bHigh = b + 0.000000000100000000
    bLow = b - 0.000000000010000000

    if a < bHigh and a > bLow:
        return True

    return False

# Exactly the same as for sets, except designed for lists
def issubset(L1, L2):
    for item in L1:
        if item not in L2:
            return False
    return True

# Exactly the same as for sets, except designed for lists
def issuperst(L1, L2):
    for item in L2:
        if item not in L1:
            return False
    return True

#returns an array of unique elements given the elements in the input arrays, all arrays must be input as a tuple
#this is VERY UNOPTIMIZED, should be replaced later by a UFUNC in numpy, but we will wait to do that
#arrays assumed to be 1D
def unique( arrayTuple ):
    master = concatenate( arrayTuple )
    uniqueElements = []
    for element in master:
        if not (element in uniqueElements):
            uniqueElements.append( element )
    return array( uniqueElements )


def addToPriorityQueue( queue, element ):
    if len( queue) == 0:
        queue.append( element )
        return queue

    for e in queue:
        if element > e:
            index = queue.index( e )
            queue = queue[0:index] + [element] + queue[index:]
            return queue

    queue.append( element )
    return queue

def intersect( L1, L2 ):
    return [e for e in L1 if e in L2]


def sample(arr):
    #given an array of probabilities return a randomly generated int with
    #probability equal to the values of array
    nPossibleValues = len(arr)
    rnum = ra.random()
    probRange = arr[0]
    i = 0
    for prob in arr[1:]:
        if rnum < probRange:
            break
        else:
            probRange += prob
            i += 1

    return i

def updateCounts(nodes, counts, data):
    assert(isinstance(bnet, Graph))
    assert(isinstance(counts, ndarray))
    assert(isinstance(data, ndarray))
    for node in nodes:
        count = counts[node.index]
        indices = data[concatenate((node.parentIndex, array([node.index])))]
        fIndex = flatIndex(indices, count.shape)
        count.flat[fIndex] += 1

def sequence_generator(iterObjs):
    assert(isinstance(iterObjs, ndarray))
    stop = iterObjs - 1
    value = zeros(len(iterObjs), dtype = int64 )
    value[0] -= 1
    #while True:
    while not alltrue(value == stop):
        for i in range(len(stop)):
            if value[i] == stop[i]:
                value[i] = 0
            else:
                value[i] += 1
                break
            yield value
        #raise StopIteration

if __name__ == "__main__":
    ray = array([2,3,3])
    seq = sequence_generator(ray)
    for s in seq:
        print(s,"\n")
    





