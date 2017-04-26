# Most of the code in this file comes from PBNT by Elliot Cohen. See
# separate file in this project with PBNT license.

import numpy as np
import itertools as it
import cmath
from fractions import Fraction


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


def log_print(x):
    """
    Prints file name of log_print() call, then file line of log_print()
    call, then x

    Parameters
    ----------
    x : object

    Returns
    -------
    None

    """
    from inspect import getframeinfo, stack
    caller = getframeinfo(stack()[1][0])
    print(caller.filename, "line=", caller.lineno, ":\n", x)


def formatted_number_str(num, num_format):
    """
    Returns formatted string for num

    Parameters
    ----------
    num : float|complex
    num_format : str

    Returns
    -------
    str

    """
    if num_format == 'Fraction':
        return str(Fraction.from_float(num).limit_denominator(100))
    elif num_format == 'Percentage':
        return "{:.2%}".format(num)
    elif num_format == 'Float':
        return str(num)
    else:
        return num_format.format(num)

def is_sq_arr(arr):
    """
    Returns True iff arr is a square array.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    bool

    """
    shp = arr.shape
    return shp == (shp[0], shp[0])

def is_herm(arr):
    """
    Returns True iff arr is a Hermitian matrix.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    bool

    """
    assert is_sq_arr(arr)
    return np.linalg.norm(arr - arr.T.conj()) < 1e-6


if __name__ == "__main__":
    # both work
    # ray = np.array([2, 3, 4])
    ray = [2, 3, 4]
    seq = cartesian_product(ray)
    for s in seq:
        print(s)

