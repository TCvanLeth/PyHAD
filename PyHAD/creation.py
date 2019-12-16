# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:02:09 2016

@author: tcvanleth

adapted functions from dask
"""

from __future__ import absolute_import, division, print_function
from functools import partial

from dask.array.core import Array, normalize_chunks
from dask.base import tokenize
import numpy as np

from phad import common


def linspace(start, stop, num=50, chunks=None, dtype=None, endpoint=True):
    """
    Return `num` evenly spaced values over the closed interval [`start`,
    `stop`].

    TODO: implement the `endpoint`, `restep`, and `dtype` keyword args

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The last value of the sequence.
    num : int, optional
        Number of samples to include in the returned dask array, including the
        endpoints.
    chunks :  int
        The number of samples on each block. Note that the last block will have
        fewer samples if `num % blocksize != 0`

    Returns
    -------
    samples : dask array

    """
    num = int(num)
    if endpoint == False:
        num = num + 1
    if chunks is None:
        raise ValueError("Must supply a chunks= keyword argument")

    chunks = normalize_chunks(chunks, (num,))

    range_ = stop - start

    space = float(range_) / (num - 1)

    name = 'linspace-' + tokenize((start, stop, num, chunks, dtype, endpoint))

    dsk = {}
    blockstart = start

    for i, bs in enumerate(chunks[0]):
        blockstop = blockstart + ((bs - 1) * space)
        task = (partial(np.linspace, dtype=dtype), blockstart, blockstop, bs)
        blockstart = blockstart + (space * bs)
        dsk[(name, i)] = task

    return Array(dsk, name, chunks, dtype=dtype)


def arange(*args, **kwargs):
    """
    Return evenly spaced values from `start` to `stop` with step size `step`.

    The values are half-open [start, stop), so including start and excluding
    stop. This is basically the same as python's range function but for dask
    arrays.

    Parameters
    ----------
    start : int, optional
        The starting value of the sequence. The default is 0.
    stop : int
        The end of the interval, this value is excluded from the interval.
    step : int, optional
        The spacing between the values. The default is 1 when not specified.
        The last value of the sequence.
    chunks :  int
        The number of samples on each block. Note that the last block will have
        fewer samples if ``len(array) % chunks != 0``.

    Returns
    -------
    samples : dask array

    """
    args = tuple(common.to_nptype(x) for x in args)

    if len(args) == 1:
        start = 0
        stop = args[0]
        step = np.int_(1)
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = np.int_(1)
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError('''
        arange takes 3 positional arguments: arange([start], stop, [step])
        ''')
    if step is None:
        step = np.int_(1)
    if start is None:
        start = np.int_(0)

    if 'chunks' not in kwargs:
        raise ValueError("Must supply a chunks= keyword argument")
    chunks = kwargs['chunks']

    dtype = kwargs.get('dtype', None)
    if dtype is None:
        dtype = np.arange(0, 1, step).dtype

    range_ = stop - start
    if hasattr(step, 'dtype') and step.dtype.kind == 'm':
        num = int(abs(range_ / step))
    else:
        num = int(np.round(abs(range_ / step)))

    chunks = normalize_chunks(chunks, (num,))

    name = 'arange-' + tokenize((start, stop, step, chunks, num))
    dsk = {}
    elem_count = 0

    # Correct arange for non-integer steps.
    def fix_arange(start, stop, step, dtype):
        x0 = int(np.round(start / step))
        x1 = int(np.round(stop / step))
        return (np.arange(x0, x1) * step).astype(dtype)

    for i, bs in enumerate(chunks[0]):
        blockstart = start + (elem_count * step)
        blockstop = start + ((elem_count + bs) * step)
        task = (fix_arange, blockstart, blockstop, step, dtype)
        dsk[(name, i)] = task
        elem_count += bs

    return Array(dsk, name, chunks, dtype=dtype)
