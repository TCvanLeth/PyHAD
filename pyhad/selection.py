#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:41:55 2016

@author: tcvanleth
"""

import dask.array as da
import numpy as np


def select(old, locs):
    if not isinstance(locs, tuple):
        locs = (locs,)

    for i, loc in enumerate(locs):
        if isinstance(loc, slice) or isinstance(loc, int):
            if loc == slice(None):
                continue
            slicer = [slice(None)] * old.ndim
            slicer[i] = loc
            old = old[tuple(slicer)]
            if all(ishape > 0 for ishape in old.shape):
                old = old#.rechunk(max(old.chunks[i]))
        elif hasattr(loc, 'chunks'):
            if loc.size == 0:
                old = da.from_array(np.array([], dtype=old.dtype), chunks=1)
            else:
                old = selectsingle(old, loc, i)
        else:
            print(locs)
            raise Exception
    return old


def selectsingle(old, loc, axis):
    name1 = 'tempselect_' + str(axis) + '_' + da.core.tokenize(old.name, loc.name)
    name = 'select_' + str(axis) + '_' + da.core.tokenize(old.name, loc.name)

    ndim = old.ndim
    nblocks = old.numblocks
    mblocks = loc.numblocks
    nstep = tuple(max(x) for x in old.chunks)[axis]
    mstep = tuple(max(x) for x in loc.chunks)[0]

    def func(iold, iloc, i, j):
        if iloc.dtype == bool:
            iloc = np.nonzero(iloc)[0] + j * mstep
        nanloc = np.where(iloc == -9223372036854775808)[0]
        iloc[nanloc] = i * nstep
        iloc = iloc[(iloc >= i * nstep) & (iloc < (i + 1) * nstep)]
        new = iold.take(iloc - i * nstep, axis=axis)
        slicer = (*(slice(None),)*axis, nanloc, ...)
        new[slicer] = np.full_like(new[slicer], np.nan)
        return new

    def func2(*parts):
        return np.concatenate(parts, axis=axis)

    dask = {}
    def recurse(to_pos, fr_pos, ndims):
        if ndims < ndim:
            if ndims == axis:
                for i in range(nblocks[ndims]):
                    for j in range(mblocks[0]):
                        fr_pos2 = fr_pos + (i,)
                        to_pos2 = to_pos + (j,)
                        recurse(to_pos2, fr_pos2, ndims + 1)
            else:
                for i in range(nblocks[ndims]):
                    fr_pos2 = fr_pos + (i,)
                    to_pos2 = to_pos + (i,)
                    recurse(to_pos2, fr_pos2, ndims + 1)
        else:
            i = fr_pos[axis]
            j = to_pos[axis]
            dask[(name1, *to_pos, i)] = (func, (old.name, *fr_pos), (loc.name, j), i, j)
            if i == nblocks[axis] - 1:
                dask[(name, *to_pos)] = (func2, *((name1, *to_pos, ii) for ii in range(nblocks[axis])))

    dask.update(old.dask)
    dask.update(loc.dask)
    recurse((), (), 0)

    chunks = list(old.chunks)
    chunks[axis] = loc.chunks[0]
    chunks = tuple(chunks)
    return da.Array(dask, name, chunks, old.dtype)
