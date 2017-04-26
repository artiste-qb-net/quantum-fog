import pandas as pd
import numpy as np
import Utilities as ut


class DataEntropy:
    """
    This class calculates classical (not quantum yet) entropy, conditional
    information (CI), mutual information (MI) and conditional mutual info (
    CMI) from a dataframe (hence, from empirical data, not from the true
    distributions of a bnet).

    logs's in entropies are base e, natural logs

    Error in entropy is ln(n+1) - ln(n) \approx 1/n where n>>1 is the number
    of samples
    """

    @staticmethod
    def ent(df, xcols, verbose=False):
        """
        Returns the entropy H(x) where x is given by the list of columns
        xcols in the dataframe df.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe for which entropy is calculated
        xcols : list[str]
            list of column names in df. The x in H(x)
        verbose : bool
            If True, print extra info in console.

        Returns
        -------
        float

        """
        if len(xcols) == 0 or xcols is None:
            return 0.0
        sample_size = len(df.index)
        df1 = df.copy()
        df1 = df1.groupby(xcols)[xcols].size()
        df1 = df1.apply(lambda x: x/sample_size)
        local_ent = -df1*np.log(df1 + 1E-7)
        all_ent = local_ent.sum()
        if verbose:
            print('\nprobs for ', xcols)
            print(df1)
            print('\nlocal ent for ', xcols)
            print(local_ent)
            print('\ntot ent for ', xcols)
            print(all_ent)
        return all_ent

    @staticmethod
    def cond_info(df, xcols, ycols, verbose=False):
        """
        Returns the conditional information (CI) H(x|y) where x (y,
        resp.) is given by the list of columns xcols (ycols, resp.) in the
        dataframe df.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe for which CI is calculated
        xcols : list[str]
            list of column names in df. The x in H(x:y)
        ycols : list[str]
            list of column names in df. The y in H(x:y)
        verbose : bool
            If True, print extra info in console.

        Returns
        -------
        float

        """
        if xcols is None:
            xcols = []
        if ycols is None:
            ycols = []

        xycols = xcols + ycols
        hxy = DataEntropy.ent(df, xycols)
        hy = DataEntropy.ent(df, ycols)
        info = hxy - hy
        if verbose:
            print('\ncond_info for', xcols, '|', ycols)
            print(info)
        return info

    @staticmethod
    def mut_info(df, xcols, ycols, verbose=False):
        """
        Returns the mutual information (MI) H(x:y) where x (y, resp.) is
        given by the list of columns xcols (ycols, resp.) in the dataframe df.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe for which MI is calculated
        xcols : list[str]
            list of column names in df. The x in H(x:y)
        ycols : list[str]
            list of column names in df. The y in H(x:y)
        verbose : bool
            If True, print extra info in console.

        Returns
        -------
        float

        """
        if xcols is None:
            xcols = []
        if ycols is None:
            ycols = []

        xycols = xcols + ycols
        hxy = DataEntropy.ent(df, xycols)
        hx = DataEntropy.ent(df, xcols)
        hy = DataEntropy.ent(df, ycols)
        info = hx + hy - hxy
        if verbose:
            print('\nmut_info for', xcols, ':', ycols)
            print(info)
        return info

    @staticmethod
    def cond_mut_info(df, xcols, ycols, zcols, verbose=False):
        """
        Returns the conditional mutual information (CMI) H(x:y|z) where x (
        y, z, resp.) is given by the list of columns xcols (ycols, zcols,
        resp.) in the dataframe df.


        Parameters
        ----------
        df : pandas.DataFrame
            dataframe for which CMI is calculated
        xcols : list[str]
            list of column names in df. The x in H(x:y|z)
        ycols : list[str]
            list of column names in df. The y in H(x:y|z)
        zcols : list[str]
            list of column names in df. The z in H(x:y|z)
        verbose : bool
            If True, print extra info in console.

        Returns
        -------
        float

        """
        if xcols is None:
            xcols = []
        if ycols is None:
            ycols = []
        if zcols is None:
            zcols = []

        xzcols = xcols + zcols
        yzcols = ycols + zcols
        xyzcols = xcols + ycols + zcols
        hxyz = DataEntropy.ent(df, xyzcols)
        hz = DataEntropy.ent(df, zcols)
        hxz = DataEntropy.ent(df, xzcols)
        hyz = DataEntropy.ent(df, yzcols)
        info = hxz + hyz - hxyz - hz
        if verbose:
            print('\ncond mut_info for', xcols, ':', ycols, '|', zcols)
            print(info)
        return info


if __name__ == "__main__":
    df = pd.DataFrame({
        'A': [3, 1, 1, 1, 1, 4, 5, 6],
        'B': [1, 4, 1, 1, 4, 7, 8, 3],
        'C': [2, 3, 5, 1, 3, 5, 6, 7],
        'D': [3, 1, 3, 1, 1, 2, 9, 5],
        'E': [5, 2, 1, 2, 2, 9, 1, 3]
    })
    df = df.astype('int')
    print(df)
    print(df.dtypes)
    DataEntropy.ent(df, ['A'], verbose=True)
    DataEntropy.ent(df, ['A', 'B'], verbose=True)
    DataEntropy.cond_info(df, ['A', 'B'], ['C'], verbose=True)
    DataEntropy.mut_info(df, ['A', 'B'], ['C'], verbose=True)
    DataEntropy.cond_mut_info(df, ['A'], ['B', 'E'], ['C'], verbose=True)
