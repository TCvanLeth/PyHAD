#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:43:08 2016

@author: tcvanleth
"""


## dask.array.core.py


def insert_to_ooc(out, arr, lock=True, region=None):
    if lock is True:
        lock = Lock()

    def store(x, index, lock, region):
        if lock:
            lock.acquire()
        try:
            if region is None:
                out[index] = np.asanyarray(x)
            else:
                out[fuse_slice(region, index)] = np.asanyarray(x)
        finally:
            if lock:
                lock.release()

        return None

    slices = slices_from_chunks(arr.chunks)

    name = 'store-%s' % tokenize(arr.name, out) # <--
    dsk = dict(((name,) + t[1:], (store, t, slc, lock, region))
               for t, slc in zip(core.flatten(arr._keys()), slices))
    return dsk


def histogram(a, bins=None, range=None, normed=False, weights=None, density=None):
    """
    Blocked variant of numpy.histogram.

    Follows the signature of numpy.histogram exactly with the following
    exceptions:

    - Either an iterable specifying the ``bins`` or the number of ``bins``
      and a ``range`` argument is required as computing ``min`` and ``max``
      over blocked arrays is an expensive operation that must be performed
      explicitly.

    - ``weights`` must be a dask.array.Array with the same block structure
      as ``a``.

    Examples
    --------
    Using number of bins and range:

    >>> import dask.array as da
    >>> import numpy as np
    >>> x = da.from_array(np.arange(10000), chunks=10)
    >>> h, bins = da.histogram(x, bins=10, range=[0, 10000])
    >>> bins
    array([     0.,   1000.,   2000.,   3000.,   4000.,   5000.,   6000.,
             7000.,   8000.,   9000.,  10000.])
    >>> h.compute()
    array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

    Explicitly specifying the bins:

    >>> h, bins = da.histogram(x, bins=np.array([0, 5000, 10000]))
    >>> bins
    array([    0,  5000, 10000])
    >>> h.compute()
    array([5000, 5000])
    """
    if bins is None or (range is None and bins is None):
        raise ValueError('dask.array.histogram requires either bins '
                         'or bins and range to be defined.')

    if weights is not None and weights.chunks != a.chunks:
        raise ValueError('Input array and weights must have the same '
                         'chunked structure')

    if not np.iterable(bins):
        bin_token = bins
        mn, mx = range
        if mn == mx:
            mn -= 0.5
            mx += 0.5

        bins = np.linspace(mn, mx, bins + 1, endpoint=True)
    else:
        bin_token = bins
    token = tokenize(a, bin_token, range, normed, weights, density)

    nchunks = len(list(core.flatten(a._keys())))
    chunks = ((1,) * nchunks, (len(bins) - 1,))

    name = 'histogram-sum-' + token

    # Map the histogram to all bins
    def block_hist(x, weights=None):
        return np.histogram(x, bins, [bins.min(), bins.max()], # <--
                                      weights=weights)[0][np.newaxis]

    if weights is None:
        dsk = dict(((name, i, 0), (block_hist, k))
                   for i, k in enumerate(core.flatten(a._keys())))
        dtype = np.histogram([])[0].dtype
    else:
        a_keys = core.flatten(a._keys())
        w_keys = core.flatten(weights._keys())
        dsk = dict(((name, i, 0), (block_hist, k, w))
                   for i, (k, w) in enumerate(zip(a_keys, w_keys)))
        dtype = weights.dtype

    all_dsk = sharedict.merge(a.dask, (name, dsk))
    if weights is not None:
        all_dsk.update(weights.dask)

    mapped = Array(all_dsk, name, chunks, dtype=dtype)
    n = mapped.sum(axis=0)

    # We need to replicate normed and density options from numpy
    if density is not None:
        if density:
            db = from_array(np.diff(bins).astype(float), chunks=n.chunks)
            return n / db / n.sum(), bins
        else:
            return n, bins
    else:
        # deprecated, will be removed from Numpy 2.0
        if normed:
            db = from_array(np.diff(bins).astype(float), chunks=n.chunks)
            return n / (n * db).sum(), bins
        else:
            return n, bins