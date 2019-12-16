# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:47:29 2016

@author: tcvanleth

adapt numpy ufuncs for use with hierarchical containers
"""

import builtins as builtin
from functools import wraps
import importlib
from argparse import Namespace

import dask.array as da
from dask.array.core import map_blocks
import numba
import numpy as np
from numpy import ma
import scipy.ndimage
import scipy.special

from phad import common as com
from phad import indexing
from phad import resampling

def dafunc(func):
    func = basecompat(func)
    func = hierarch(func)
    return func


def sufunc(**kwargs):
    def ufunc(func):
        """
        wrap a function to provide the ufunc hooks
        """
        func = daskcompat(func, **kwargs)
        func = basecompat(func)
        func = hierarch(func)
        return func
    return ufunc


ufunc = sufunc()


def hierarch(func):
    """
    wrap a function to provide the ufunc hooks
    """
    @wraps(func)
    def nfunc(*args, **kwargs):
        for i in range(len(args)):
            iarg = args[i]
            if hasattr(iarg, '__numpy_ufunc__'):
                out = iarg.__numpy_ufunc__(nfunc, '__call__', i, args, **kwargs)
                if out is not NotImplemented:
                    return out
        return func(*args, **kwargs)
    return nfunc


def basecompat(func):
    @wraps(func)
    def nfunc(*args, **kwargs):
        args = tuple(x if hasattr(x, '__call__')
                     else com.to_nptype(x) for x in args)
        return func(*args, **kwargs)
    return nfunc


def daskcompat(func, dtypes=None):
    @wraps(func)
    def nfunc(*args, **kwargs):
        if builtin.any(hasattr(iarg, 'chunks') for iarg in args):
            if dtypes is not None:
                kwargs['dtype'] = np.dtype(dtypes[str(args[0].dtype)])
            else:
                kwargs['dtype'] = args[0].dtype
            return map_blocks(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    return nfunc


def extend_dims(array, xdims):
    newchunks = tuple((i,) for i in xdims)
    name = array.name
    chunks = array.chunks + newchunks
    dsk = array.dask
    blocks = np.meshgrid(*(np.arange(i) for i in array.numblocks))
    blocks = zip(*(x.flatten() for x in blocks))

    for iblock in blocks:
        dsk[(name,)+iblock+(0,)*len(xdims)] = dsk.pop((name,)+iblock)
    return da.Array(dsk, name, chunks, dtype=array.dtype)


###############################################################################
# implement the proper hooks in existing numpy ufuncs
da_names = ['log', 'log10', 'exp', 'sqrt',
            'cos', 'sin', 'arctan2', 'min', 'max',
            'real', 'imag', 'floor', 'ceil', 'round', 'around',
            'maximum', 'minimum', 'mean', 'sum', 'std', 'any', 'all',
            'percentile', 'argmax', 'argmin', 'cumsum', 'cumprod',
            'nansum', 'nanmean', 'nanmin', 'nanmax', 'nanstd',
            'nancumsum', 'nancumprod', 'deg2rad', 'stack', 'dot',
            'isnan', 'isinf', 'isclose', 'histogram', 'where', 'matmul',
            'diag']

ufunc_names = ['divide', 'abs', 'diagonal', 'polyfit']
char_names = ['split']
special_names = ['gamma']

common_names = ['isnull', 'nan_equal', 'array_equal']


thismodule = importlib.import_module(__name__)
for name in da_names:
    setattr(thismodule, name, dafunc(getattr(da, name)))

for name in ufunc_names:
    setattr(thismodule, name, ufunc(getattr(np, name)))

char = Namespace()
for name in char_names:
    setattr(char, name, ufunc(getattr(np.char, name)))

for name in special_names:
    setattr(thismodule, name, ufunc(getattr(scipy.special, name)))

for name in common_names:
    setattr(thismodule, name, ufunc(getattr(com, name)))
setattr(thismodule, name, ufunc(ma.array))


###############################################################################
# new unversal functions
searchsorted = hierarch(indexing.searchsorted)


@sufunc(dtypes={'int64': 'int64', 'int32':'int32'})
def bincount(x, weight, axis=None):
    """
    da.bincount for multidimensional arrays.
    number of occurances are counted only over one axis.
    """
    if isinstance(axis, tuple):
        axis = axis[0]
    axes = tuple(range(axis)) + tuple(range(axis + 1, x.ndim)) + (axis,)
    x = x.transpose(axes)
    weight = weight.transpose(axes)
    shp = np.maximum(weight.shape, x.shape)

    n = len(axes)
    for i in range(n - 1, 0, -1):
        idx = np.arange(0, np.prod(shp[i - 1:]), np.prod(shp[i:]))
        sli = (None,) * (i - 1) + (slice(None),) + (None,) * (n - i)
        x = x + idx[sli]

    res = np.bincount(x.ravel(), minlength=np.prod(shp), weights=weight.ravel())
    return res.reshape(shp).transpose(np.argsort(axes))


@ufunc
def in1d(ar1, ar2, **kwargs):
    shp  = ar1.shape
    ar1 = ar1.flatten()
    return np.in1d(ar1, ar2, **kwargs).reshape(shp)


def rolling(func):
    def nfunc(a, *args, window=10):
        depth = {i:window[i] if i in window else 1 for i in range(a.ndim)}
        size = tuple(window[i] if i in window else 1 for i in range(a.ndim))
        boundary = {i:'reflect' for i in range(a.ndim)}
        origin = tuple(x // 2 - 1 + x % 2 for x in size)

        # Divide and conquer in case our window is too large for the chunksize.
        # NB: This is not guaranteed to get exact results for every filter function.
#        if depth > a.chunks * 0.1:
#            stride = depth * 10 / a.chunks
#            a = resampling.reshape(a, nstep, mstep, a.chunks, aligned)
#            a = func(a, axis=i).rechunk(a.chunks)
#            depth = depth / stride
#            size = size / stride

        b = da.ghost.ghost(a, depth, boundary)
        filt = scipy.ndimage.filters.generic_filter
        c = da.map_blocks(filt, b, func, size=size, dtype=a.dtype, extra_arguments=args)
        return da.ghost.trim_internal(c, depth)
    return nfunc


@dafunc
def rollmedian(a, window=10):
    """
    rolling median (onesided window)
    """
    depth = {i:window[i] if i in window else 1 for i in range(a.ndim)}
    size = tuple(window[i] if i in window else 1 for i in range(a.ndim))
    boundary = {i:'reflect' for i in range(a.ndim)}
    origin = tuple(x // 2 - 1 + x % 2 for x in size)

    b = da.ghost.ghost(a, depth, boundary)
    func = scipy.ndimage.filters.median_filter
    c = da.map_blocks(func, b, size=size, dtype=a.dtype)
    return da.ghost.trim_internal(c, depth)


rollmean = dafunc(rolling(np.mean))
nanrollmean = dafunc(rolling(np.nanmean))
rollstd = dafunc(rolling(np.std))
nanrollstd = dafunc(rolling(np.nanstd))
rollpercentile = dafunc(rolling(np.percentile))
nanrollpercentile = dafunc(rolling(np.nanpercentile))
rollmax = dafunc(rolling(np.max))
nanrollmax = dafunc(rolling(np.nanmax))
rollmin = dafunc(rolling(np.min))
nanrollmin = dafunc(rolling(np.nanmin))
rollsum = dafunc(rolling(np.sum))
nanrollsum = dafunc(rolling(np.nansum))
rollany = dafunc(rolling(np.any))
nanrollany = dafunc(rolling(np.any))
nanrollmedian = dafunc(rolling(np.nanmedian))

@dafunc
def rollall(a, n, axis=None):
    """ test whether the n next array elements along a given axis evaluate to
    True in a rolling window.
    """
    depth = {i:n if i == axis else 0 for i in range(a.ndim)}
    boundary = {i:'reflect' for i in range(a.ndim)}

    def func(x):
        xsum = np.cumsum(x, axis=axis)
        out = np.zeros_like(x)
        out[:-n] = (xsum[n:] - xsum[:-n])//n
        return out

    a = da.ghost.ghost(a, depth, boundary)
    c = da.map_blocks(func, a)
    c = da.ghost.trim_internal(c, depth)
    return c


@ufunc
def hystgate(begin, end, axis=None):
    """ turns gate on with one condition and off with another.
    """
    nr = begin.shape[axis]
    event = np.zeros((nr), bool)
    event[0] == begin[0]

    for i in range(1, nr):
        event[i] = (event[i-1] or begin[i]) and not end[i]

    event[1:] = (event[:-1] | begin[1:]) & ~end[1:]
    return event


@dafunc
def count(a):
    """counts number of distinct "on" switches in a boolean 1d array
    and assigns them cumulatively
    """
    a = a.astype(int)
    count = da.cumsum(da.insert(da.maximum(a[1:]-a[:-1], 0), a[0], 0, axis=0), axis=0)
    return da.where(a == True, count, 0)


@ufunc
def nandiv(a, b):
    """ division ignoring divide by zero and substituting zero.
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


@ufunc
def nansqrt(a):
    """ square root substituting zero for invalid (negative) input
    NOTE: mainly useful when no negative input is expected, but numerical
    precision could result in small negative values around zero.
    """
    return np.sqrt(a, out=np.zeros_like(a), where=a>=0)


@ufunc
def nanlog10(a):
    """
    base 10 logarithm ignoring divide by zero and substituting zero.
    """
    return np.log10(a, out=np.zeros_like(a), where=a>0)


@ufunc
def nanlog(a):
    """
    base e logarithm ignoring divide by zero and substituting zero.
    """
    return np.log(a, out=np.zeros_like(a), where=a>0)


@dafunc
def average(a, weights, **kwargs):
    """
    compute the weighted average
    """
    avg = da.sum(a * weights, **kwargs)
    tot = da.sum(weights, **kwargs)
    res = avg / tot
    return res


@dafunc
def nanaverage(a, weights, **kwargs):
    """
    compute the weighted average with nans ignored
    """
    avg = da.nansum(a * weights, **kwargs)
    tot = da.nansum(weights, **kwargs)
    return nandiv(avg, tot)


@dafunc
def nanpercentile(a, q, axis=None, **kwargs):
    """
    compute the qth percentile from the values in array a
    """
    return da.percentile(a, q, where=~da.isnan(a), axis=None, **kwargs)


@ufunc
def median(a, axis=None, **kwargs):
    """
    compute the median
    """
    return np.percentile(a, 50, axis=None, **kwargs)


@dafunc
def nanmedian(a, axis=None, **kwargs):
    """
    compute the median ignoring nan values
    """
    return nanpercentile(a, 50, axis=None, **kwargs)

@dafunc
def cov(*args, axis=None, **kwargs):
    """
    covariance
    """
    if axis is None:
        args = [x.flatten() for x in args]
        axis = 0

    X = da.stack(args, axis=-1).rechunk(com.CHUNKSIZE)
    cond = da.any(da.isnan(X), axis=-1)
    X = da.where(cond[..., None], np.nan, X)

    X -= da.nanmean(X, axis=axis, keepdims=True)
    X = da.where(da.isnan(X), 0, X)
    return X.swapaxes(axis,-1) @ X.swapaxes(axis,-2).conj() / (X.shape[axis] - 1)


@dafunc
def corr(*args, axis=None, **kwargs):
    """
    Pearson's correlation coefficient
    """
    c = cov(*args, axis=axis, **kwargs)
    std = da.sqrt(np.diagonal(c, axis1=-1, axis2=-2))[..., None]
    return c / std / std.swapaxes(-1,-2)


@dafunc
def interm(x, y, axis=None):
    n = da.nansum((x > 0.1) & (y > 0.1) & ~da.isnan(x) & ~da.isnan(y), axis=axis)
    o = da.nansum(((x > 0.1) | (y > 0.1)) & ~da.isnan(x) & ~da.isnan(y), axis=axis)
    return n/o


@hierarch
def broadcast(*args):
    return args


@sufunc(dtypes={'complex128':'float64', 'complex64': 'float32'})
@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


@ufunc
def splitn(a, sep, n):
    return np.array([x[n] if len(x) > n else 'NA' for x in np.char.split(a, sep)])