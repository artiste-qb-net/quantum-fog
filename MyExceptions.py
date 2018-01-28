# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


class BadGraphStructure(Exception):
    """
    An exception class raised when a graph's structure is detected to be
    illegal; for instance, when an alleged DAG is detected to contain cycles.

    Attributes
    ----------
    txt : str

    """

    def __init__(self, txt):
        """
        Constructor

        Parameters
        ----------
        txt : str

        Returns
        -------

        """
        self.txt = txt

    def __repr__(self):
        """

        Returns
        -------
        str

        """
        return self.txt


class UnNormalizablePot(Exception):
    """

    An exception class raised when an attempt to normalize a DiscreteCondPot
    fails because it leads to division by zero.

    Attributes
    ----------
    pa_indices : tuple[int]

    """

    def __init__(self, pa_indices):
        """
        Constructor

        Parameters
        ----------
        pa_indices : tuple[int]
            the indices of the parent state pa(C)=y such that
            pot(C=x|pa(C)=y) = 0 for all x.

        Returns
        -------

        """
        self.pa_indices = pa_indices

    def __repr__(self):
        """

        Returns
        -------
        tuple[int]

        """
        return self.pa_indices

if __name__ == "__main__":
    def main():
        print(5)
    main()

