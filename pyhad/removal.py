#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:18:57 2017

@author: tcvanleth
"""
import dask.array as da
import numpy as np


def remove(old, locs):
    if not isinstance(locs, tuple):
        locs = (locs,)

    for i, loc in enumerate(locs):
        if isinstance(loc, slice):
            if loc == slice(None):
                continue
            slicer = [slice(None)] * old.ndim
            slicer[i] = loc
            old = old[tuple(slicer)]
        elif hasattr(loc, 'chunks'):
            if loc.size == 0:
                continue
            else:
                old = removesingle(old, loc, i)
        else:
            raise Exception
    return old


def removesingle(old, loc, axis):
    name = 'select_' + str(axis) + '_' + da.core.tokenize(old.name, loc.name)

    ndim = old.ndim
    nblocks = old.numblocks
    mblocks = loc.numblocks
    nstep = tuple(max(x) for x in old.chunks)[axis]

    def func(iold, iloc, i):
        if iloc.dtype == bool:
            iloc = np.nonzero(iloc)[0]
        iloc = iloc[(iloc >= i * nstep) & (iloc < (i+1) * nstep)]
        new = iold.delete(iloc, axis=axis)
        return new

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
            dask[(name, *to_pos)] = (func, (old.name, *fr_pos), (loc.name, j), i)

    dask.update(old.dask)
    dask.update(loc.dask)
    recurse((), (), 0)

    chunks = list(old.chunks)
    chunks[axis] = loc.chunks[0]
    chunks = tuple(chunks)
    return da.Array(dask, name, chunks, old.dtype)
