# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np
from MyConstants import *
import itertools as it


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


def debug_assert(*args):
    """
    Wrapper for assert() so that assert() can be removed after debugging.

    Parameters
    ----------
    args :

    Returns
    -------

    """
    if DEBUG_ON:
        assert args


if __name__ == "__main__":
    # both work
    # ray = np.array([2, 3, 3])
    ray = [2, 3, 3]
    seq = cartesian_product(ray)
    for s in seq:
        print(s)

