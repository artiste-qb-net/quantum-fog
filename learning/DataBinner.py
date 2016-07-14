import pandas as pd
import numpy as np


class DataBinner:
    """
    This class is all static methods. When the data in a column of a
    dataframe are floats, we want to group those floats into bins and assign
    an int to each bin. We refer to this process as binning. This class is
    mainly just a wrapper for some pandas methods that do most of the heavy
    lifting.

    """

    @staticmethod
    def bin_col(df, col_name, num_bins, do_qtls=True):
        """
        Bins INPLACE the column called col_name in the dataframe df. By
        inplace we mean that df is changed: its column col_name is replaced
        by a binned version. The function returns bin_edges, a list of the
        edges of the bins, and bin_to_mean, a dictionary mapping bin number
        to the mean value of the points inside the bin. This is mainly a
        wrapper for the Pandas functions cut() and qcut().

        Parameters
        ----------
        df : pandas.DataFrame
        col_name : str
            name of the column that you wish to bin
        num_bins : int
            number of bins
        do_qtls : bool
            do quantiles. If True, will bin into quantiles, if False will
            use equal length bins.

        Returns
        -------
        (list(float), list(float))

        """
        col_name_ = col_name + '_'
        # print('---df---', df, df.dtypes)
        df_pair = pd.DataFrame(df[col_name], dtype=float)

        # add small amount of noise so qcut() doesn't get
        # quantiles of zero width
        df_noise = pd.DataFrame(
            {col_name: np.random.rand(len(df.index))*1E-7})
        df_pair[col_name] = df_pair[col_name] + df_noise[col_name]

        # print('rand df_pair', df_pair)
        # add extra column to df_pair, hence its name because it has 2 columns
        df_pair[col_name_] = df_pair[col_name]
        # print('df_pair=\n', df_pair)
        if do_qtls:
            binner = pd.qcut
        else:
            binner = pd.cut
        # print(df_pair, df_pair.dtypes)
        df_pair[col_name_], bin_edges = binner(
            df_pair[col_name_], num_bins, labels=False, retbins=True)
        # print('======df_pair\n', df_pair)
        # print('bin_edges', bin_edges)
        df[col_name] = df_pair[col_name_]
        # print('====df', df)
        means_df = df_pair.groupby([col_name_])[col_name].mean()
        # print('means_df\n', means_df)
        bin_to_mean = dict(means_df)
        # print(bin_to_mean)

        return bin_edges, bin_to_mean

if __name__ == "__main__":
    df = pd.DataFrame({
        'A': [3, 1, 3, 1, 1, 4, 5, 6],
        'B': [1, 4, 1, 4, 4, 7, 8, 3],
        'C': [2, 3, 2, 3, 3, 5, 6, 7],
        'D': [3, 1, 3, 1, 1, 2, 9, 5],
        'E': [5, 2, 5, 2, 2, 9, 1, 3]
    })
    df['B'] = df['B']*10 + .2
    print('input df=\n', df)

    bin_edges, bin_to_mean = DataBinner.bin_col(df, 'B', 3)
    print('after binning B, df=\n', df)
    print('bin_edges=\n', bin_edges)
    print('bin_to_mean=\n', bin_to_mean)
