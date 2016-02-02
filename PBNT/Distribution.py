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
import numpy.random as ra
from pbnt.Utilities import GraphUtilities
from pbnt.Utilities import Utilities
import copy

class Potential():
    """ Potentials are very similar to a conditional distribution in that they specify the probability over a set of nodes. The difference is that potentials are not thought of as being centered on the value a one node given other nodes. Therefore, a conditional distribution could be thought of as a special case of a potential.
    """

    def __init__(self, nodes, table=[], default=1):
        #WARNING: prone to bugs where we convert nodes to a
        #list, but table was given and doesn't match newly created list
        self.nodes = list(nodes)
        self.__nodeSet_ = set(nodes)
        self.dims = array([node.size() for node in self.nodes])
        if not isinstance(table, ndarray):
            self.table = zeros(self.dims, dtype= float32) + default
        else:
            self.table = table
            assert(alltrue(shape(table) == self.dims)), "Potential Init Error: Node sizes do not agree with input table"
        self.nDims = len(self.dims)

    def marginalize(self, other):
        """ Return a new potential that is the marginalization of this potential given other.  This identifies the instantiations of self (s1,s2,...,sn) that are consistent with other and sum self(s1) + self(s2) + ... + self(sn).
        """
        new = copy.deepcopy(other)
        intersect = self.__nodeSet_.intersection(new.__nodeSet_)
        newAxes = range(new.nDims)
        sequence = Utilities.sequence_generator(other.dims)
        for seq in sequence:
            index = self.generate_index_node(seq, intersect)
            newIndex = new.generate_index(seq, newAxes)
            val = self[index]
            if isinstance(val, ndarray):
                val = val.sum()
            new[newIndex] = val
        return new

    def normalize(self):
        # Make sure that the last dimension adds to 1 along all other values.
        if self.nDims > 1:
            seq = SequenceGenerator(self.dims[:-1])
            for s in seq:
                index = self.generate_index(s, range(self.nDims - 1))
                c = self.table[index].sum()
                if not c == 0:
                    self.table[index] /= c
        else:
            c = self.table.sum()
            if not c == 0:
                self.table /= c

    def generate_index_node(self, index, nodes):
        """ Generates a list of axes that correspond to nodes, then calls generate_index with the newly generated list of axes.
        """
        assert(self.__nodeSet_.issuperset(nodes))
        axes = [self.nodes.index(node) for node in nodes]
        return self.generate_index(index, axes)

    def generate_index(self, index, axis):
        """ This function works hand in hand with __getitem__.  It takes in a list of indices and a list of axes and generates an index in a format appropriate for __getitem__, which is currently generating strings which are then executed using eval.
        """
        if isinstance(index, (int, float, long)):
            index = [index]
        # assert(len(index) == len(axis))
        tmp = zeros(self.nDims) - 1
        if len(axis) > 0:
            tmp[axis] = index
        indexStr = ""
        for i in tmp:
            if i == -1:
                indexStr += ":,"
            else:
                indexStr += str(i)
                indexStr += ","
        return indexStr[:-1]

    def transpose(self, nodes):
        #FIXME: would like the assertion to be stronger, would like set(nodes) == self.__nodeSet_
        assert(len(nodes) == self.nDims), "Potential Error: Cannot take transpose with a different set of nodes"
        axes = [self.nodes.index(node) for node in nodes]
        self.table.transpose(axes)
        self.nodes = nodes
        self.__nodeSet_ = set(nodes)

    def transpose_copy(self, nodes):
        #FIXME: would like the assertion to be stronger, would like set(nodes) == self.__nodeSet_
        assert(len(nodes) == self.nDims), "Potential Error: Cannot take transpose with a different set of nodes"
        axes = [self.nodes.index(node) for node in nodes]
        return transpose(table, axis=axes)

    """ The following are the overloaded operators of this class. I want these distributions to be treated like tables, even if the underlying representation is not an array or table.  By overloading these, I can treat these classes as if they are just tables with a couple of extra methods specific to the distribution class I am dealing with.  There are two advantages in particular.  First, if I need to improve performance, these classes could be implemented in C by inheriting from the numpy array object and adding the extra methods needed to deal with these objects as distributions.  Second, if I decide to change the underlying array class from numpy to numeric or to something totally different, it wont affect anything else, because everything else with be abstracted away.  This is further guaranteed by generate_index which generates an index for its class given which axes should be set and what the value of those axes are.
    """
    def __eq__(self, other):
        return self.__nodeSet_ == other.__nodeSet_

    def __getitem__(self, index):
        return eval("self.table["+index+"]")

    def __setitem__(self, index, value):
        exec("self.table["+index+"]=" + repr(value))

    def __add__(self, right):
        """ Pointwise addition of elements in self and right.  Assumes that self and right are defined over the same nodes.
        """
        new = copy.deepcopy(self)
        if isinstance(right, (int, float, complex, long)):
            new.table += right
        else:
            assert(self.__nodeSet_ == right.__nodeSet_), \
                  "Attempted to add two Potentials with different sets of nodes"
            right.transpose(new.nodes)
            new.table += right.table
        return new

    def __iadd__(self, right):
        """ Pointwise addition of elements in self and right.  Assumes that self and right are defined over the same nodes.  This operator is called for in place addition +=.
        """
        if isinstance(right, (int, float, complex, long)):
            self.table += right
        else:
            assert(self.__nodeSet_ == right.__nodeSet_), \
                  "Attempted to add two Potentials with different sets of nodes"
            right.transpose(self.nodes)
            self.table += right.table
        return self

    def __mul__(self, right):
        """ A true multiplication of two potentials would be defined as X * Y = Z where the sets of variables z = x U y.  We would then identify the instantiations of x and y that are consistent with z and Z(z) = X(x)Y(y).  We are generally going to be multiplying sepset potentials by clique potentials where the variables of a setpset potential are a subset of the variables of the clique.  Therefore we are going to assume in this operation that right's variables are a subset of self's.
        """
        # right should only be a DiscreteDistribution or a ContinuousDistribution if it is a subset and it should be __imul__
        assert(not isinstance(right, DiscreteDistribution) and not isinstance(right, ConditionalDiscreteDistribution)), \
              "Attempt to Multiply Potential with incompatible type: Discrete or Conditional"
        if isinstance(right, (int, float, complex, long)):
            potential = copy.deepcopy(self)
            potential.table *= right
        else:
            nodeSet = self.__nodeSet_.union(right.__nodeSet_)
            potential = Potential(list(nodeSet))
            selfValues = [potential.nodes.index(node) for node in self.nodes]
            rightValues = [potential.nodes.index(node) for node in right.nodes]
            # Store the following lists so we don't have to recompute them on every iteration
            potAxes = range(potential.nDims)
            selfAxes = range(self.nDims)
            rightAxes = range(right.nDims)
            #OPTIMIZE: Should be able to do this without blindly iterating through dimensions.
            for seq in Utilities.sequence_generator(potential.dims):
                #OPTIMIZE: Could access the table directly, but would break down our abstraction
                potIndex = potential.generate_index(seq, potAxes)
                selfIndex = self.generate_index(seq[selfValues], selfAxes)
                rightIndex = right.generate_index(seq[rightValues], rightAxes)
                potential[potIndex] = self[selfIndex] * right[rightIndex]
        return potential

    def __imul__(self, right):
        """ This is the same operation as __mul__ except that if right.nodes is a subset of self.nodes, we do the multiplication in place, because there is no reason to make a copy, which wastes time and space.
        """
        if isinstance(right, (int, float, complex, long)):
            self.table *= right
        # FIXME: should be right.__nodeSet_ but doesn't work when right is DiscreteDistribution
        elif self.__nodeSet_.issuperset(right.nodes):
            #OPTIMIZE: There must be a way to do this without iterating over every value of table
            selfAxes = [self.nodes.index(node) for node in right.nodes]
            rightAxes = range(right.nDims)
            for seq in Utilities.sequence_generator(right.dims):
                selfIndex = self.generate_index(seq, selfAxes)
                #OPTIMIZE: Could index right.table directly, but this upholds our abstraction barrier
                rightIndex = right.generate_index(seq, rightAxes)
                self[selfIndex] *= right[rightIndex]
        else:
            """ If potential will be over a different set of variables after multiplication, might as well use full __mul__ version, which copies.
            """
            self = self.__mul__(right)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """ ASSUMPTION: This is only defined for potentials over the same set of nodes.  It is a pointwise division of each element within the potential.
        """
        new = copy.deepcopy(self)
        if (isinstance(other, (int, float, complex, long))):
            new.table /= other
        else:
            assert(self.__nodeSet_ == other.__nodeSet_)
            new.transpose(other.nodes)
            new.table /= other.table
        return new

    def __idiv__(self, other):
        """ Same as div, but operates on self in place.
        """
        if (isinstance(other, (int, float, complex, long))):
            self.table /= other
        else:
            assert(self.__nodeSet_ == other.__nodeSet_)
            other.transpose(self.nodes)

            self.table /= other.table
        return self

    def __deepcopy__(self, memo):
        copyTable = copy.deepcopy(self.table)
        return Potential(nodes=self.nodes, table=copyTable)


class DiscreteDistribution(Potential):
    """ The basic class for a distribution, it defines a simple distribution over a set number of values.  This is not to be confused with ConditionalDiscreteDistribution, which is a discrete distribution conditioned on other discrete distributions.
    """

    def __init__(self, node):
        self.node = node
        # FIXME: These should be accomplished through the overloading of __getattr__
        self.nodes = [node]
        self.__nodeSet_ = set([node])
        # END FIXME
        self.table = zeros([node.size()], dtype=float32)
        self.dims = array(shape(self.table))
        self.nDims = 1

    def set_value(self, value, probability):
        self.table[value] = probability

    def size(self):
        return self.node.size()

    def sample(self):
        #Sample a value given the distribution specified in self.table
        rnum = ra.random()
        probRange = 0
        i = -1
        for prob in self.table:
            probRange += prob
            i += 1
            if rnum <= probRange:
                break
        return i

    def __eq__(self, other):
        self.node == other.node


class ConditionalDiscreteDistribution(Potential):
    """ This is very similar to a potential, except that ConditionalDiscreteDistributions are focused on a single variable and its value conditioned on other variables.
    """

    def __init__(self, nodes=[], table=[]):
        Potential.__init__(self, nodes=nodes, table=table)
        self.node = nodes[-1]

    def size(self):
        return self.node.size()

    def __eq__(self, other):
        return isinstance(other, ConditionalDiscreteDistribution) and \
               Potential.__eq__(self, other) and self.node == other.node

    def __deepcopy__(self, memo):
        copyTable = copy.deepcopy(self.table)
        return ConditionalDiscreteDistribution(nodes=self.nodes, table=copyTable)

if __name__ == "__main__":
    x = [0,1,2,3,4]
    print(x[:-1])
