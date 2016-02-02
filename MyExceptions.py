# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.


class BadGraphStructure(Exception):
    """
    An exception class raised when a graph's structure is detected to be
    illegal. Thrown when an alleged DAG is detected to contain cycles.

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

if __name__ == "__main__":
    print(5)
