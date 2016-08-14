# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np
from MyConstants import *
import itertools as it
import cmath


def cartesian_product(shape_list):
    """
    Given a list of ints [k1, k2,...], it generates the elements of the
    cartesian product of range(k1) times range(k2) times ...

    Parameters
    ----------
    shape_list : list[int]

    Returns
    -------
    itertools.product

    """
    x = (range(shape_list[k]) for k in range(len(shape_list)))
    return it.product(*x)


def fix(in_str, bad_chs, sub):
    """
    This replaces in 'in_str' each character of 'bad_chs' by a 'sub'

    Parameters
    ----------
    in_str : str
    bad_chs : str
    sub : str

    Returns
    -------
    str

    """
    for c in bad_chs:
        in_str = in_str.replace(c, sub)
    return in_str

if __name__ == "__main__":
    # both work
    # ray = np.array([2, 3, 4])
    ray = [2, 3, 4]
    seq = cartesian_product(ray)
    for s in seq:
        print(s)

