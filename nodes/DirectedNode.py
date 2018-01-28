# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


from nodes.Node import *
# from MyExceptions import *


class DirectedNode(Node):
    """
    A DirectedNode is a Node. Besides maintaining a list of
    neighbors, it maintains a list of parents and children.

    Attributes
    ----------
    children : set[DirectedNode]
    neighbors : set[DirectedNode]
    parents : set[DirectedNode]

    """

    def __init__(self, id_num, name="blank"):
        """
        Constructor

        Parameters
        ----------
        id_num : int
        name : str

        Returns
        -------

        """
        Node.__init__(self, id_num, name)
        self.parents = set()
        self.children = set()

    def add_parent(self, node):
        """
        Add a parent node.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        None

        """
        if self != node:
            self.parents.add(node)
            node.children.add(self)

    def add_parents(self, node_list):
        """
        Add parent nodes from a list of them.

        Parameters
        ----------
        node_list : list(DirectedNode)

        Returns
        -------
        None

        """
        for nd in node_list:
            self.add_parent(nd)

    def remove_parent(self, node):
        """
        Remove a parent node.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        None

        """

        self.parents.remove(node)
        node.children.remove(self)

    def has_parent(self, node):
        """
        Returns answer to question of whether self has the given node as
        parent.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        bool

        """

        return node in self.parents

    def add_child(self, node):
        """
        Add a child node.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        None

        """
        if self != node:
            self.children.add(node)
            node.parents.add(self)

    def add_children(self, node_list):
        """
        Add children nodes from a list of them

        Parameters
        ----------
        node_list : list(DirectedNode)

        Returns
        -------
        None

        """
        for nd in node_list:
            self.add_child(nd)

    def remove_child(self, node):
        """
        Remove a child node.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        None

        """
        self.children.remove(node)
        node.parents.remove(self)

    def has_child(self, node):
        """
        Returns answer to question of whether self has the given node as
        child.

        Parameters
        ----------
        node : DirectedNode

        Returns
        -------
        bool

        """
        return node in self.children

    def undirect(self):
        """
        This makes neighbor set equal to union of parent and children sets.

        Returns
        -------
        None

        """

        self.neighbors = self.parents | self.children

    def get_markov_blanket(self):
        """
        This returns a set called the Markov Blanket of a node,
        which includes the parents, children and children's parents of the
        node.

        Returns
        -------
        set[DirectedNode]

        """
        mb = self.parents | self.children
        for child in self.children:
            mb |= child.parents
        return mb

if __name__ == "__main__":
    def main():
        p1 = DirectedNode(0, "p1")
        p2 = DirectedNode(1, "p2")
        center = DirectedNode(2, "center")
        c1 = DirectedNode(3, "c1")
        c2 = DirectedNode(4, "c2")

        center.add_parent(p1)
        center.add_parent(p2)
        center.add_child(c1)
        center.add_child(c2)

        assert center.has_child(c1)
        assert center.has_child(c2)
        assert center.has_parent(p1)
        assert center.has_parent(p2)
        assert c1.has_parent(center)
        center.remove_parent(p1)
        assert not center.has_parent(p1)

        print(center.name)
    main()






