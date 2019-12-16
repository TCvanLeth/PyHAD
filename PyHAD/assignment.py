# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:46:40 2016

@author: tcvanleth
"""
from itertools import product

import dask.array as da
import numpy as np


def assign(old, locs, new):
    if hasattr(locs, 'dtype') and locs.dtype == bool and locs.shape == old.shape:
        return da.where(locs, new, old)
    if all(ishape <= ichunk[0] for ishape, ichunk in zip(new.shape, new.chunks)):
        if all(ishape <= ichunk[0] for ishape, ichunk in zip(old.shape, old.chunks)):
            return assignsingle(old, locs, new)
        else:
            return assignblock(old, locs, new)
    if any(hasattr(x, 'chunks') and x.chunks[0][0] < x.shape[0] for x in locs):
        return assignirregular(old, locs, new)
    else:
        new = assignblocks(old, locs, new)
        return new


def assignsingle(old, locs, new):
    names = tuple(iloc.name if hasattr(iloc, 'name') else iloc for iloc in locs)
    name = 'assign_' + da.core.tokenize(old.name, new.name, *names)

    ndim = old.ndim
    mdim = new.ndim
    new = new[(...,) + (None,) * (ndim - mdim)]

    def func(iold, inew, *ilocs):
        iold2 = iold.copy()
        iold2[ilocs] = inew
        return iold2

    to_pos = (0,) * old.ndim
    fr_pos = (0,) * new.ndim
    iold = (old.name, *to_pos)
    inew = (new.name, *fr_pos)

    dask = {}
    newlocs = []
    for iloc in locs:
        if hasattr(iloc, 'dask'):
            loc_pos = (0,) * iloc.ndim
            newlocs.append((iloc.name, *loc_pos))
            dask.update(iloc.dask)
        else:
            newlocs.append(iloc)

    dask.update({(name, *to_pos): (func, iold, inew, *tuple(newlocs))})
    dask.update(old.dask)
    dask.update(new.dask)
    return da.Array(dask, name, old.chunks, old.dtype)


def assignblock(old, locs, new):
    name = 'assign_' + da.core.tokenize(old.name, new.name, locs)

    ndim = old.ndim
    mdim = new.ndim
    new = new[(...,) + (None,) * (ndim - mdim)]
    nblocks = old.numblocks
    nstep = tuple(max(x) for x in old.chunks)
    mshape = new.shape

    fr_slicer, to_slicer = get_slice(locs, nstep, nblocks, mshape)
    fr_pos = (0,) * mdim

    def func(iold, inew, to_slices, fr_slices):
        iold2 = iold.copy()
        iold2[to_slices] = inew[fr_slices]
        return iold2

    def recurse(to_pos, to_slices, fr_slices, ndims):
        if ndims < ndim:
            for i in range(nblocks[ndims]):
                to_pos2 = to_pos + (i,)
                to_slices2 = to_slices + to_slicer[ndims][i]
                fr_slices2 = fr_slices + fr_slicer[ndims][i]
                recurse(to_pos2, to_slices2, fr_slices2, ndims + 1)
        else:
            iold = (old.name, *to_pos)
            inew = (new.name, *fr_pos)
            if any(x.start == 0 and x.stop == 0 for x in to_slices):
                dask[(name, *to_pos)] = dask[iold]
            else:
                dask[(name, *to_pos)] = (func, iold, inew, to_slices, fr_slices)

    dask = {}
    dask.update(old.dask)
    dask.update(new.dask)
    recurse((), (), (), 0)
    return da.Array(dask, name, old.chunks, old.dtype)


def assignblocks(old, locs, new):
    name = 'assign_' + da.core.tokenize(old.name, new.name, locs)

    ndim = old.ndim
    mdim = new.ndim
    new = new[(...,) + (None,) * (ndim - mdim)]
    nblocks = old.numblocks
    mblocks = new.numblocks
    nstep = tuple(max(x) for x in old.chunks)
    mstep = tuple(max(x) for x in new.chunks)
    mshape = new.shape

    fr_slicer, to_slicer = get_blockslice(locs, nstep, mstep, nblocks, mblocks,
                                          mshape)

    def assign(iold, inew, to_slices, fr_slices):
        iold2 = iold.copy()
        iold2[to_slices] = inew[fr_slices]
        return iold2

    dask = dict.merge(old.dask, new.dask)

    to_pos = tuple(product(*tuple(range(x) for x in nblocks)))
    fr_pos = tuple(product(*tuple(range(x) for x in mblocks)))
    for ito in to_pos:
        name1 = None
        for ifr in fr_pos:
            to_slices = tuple(x[ito[k], ifr[k]] for k, x in enumerate(to_slicer))
            fr_slices = tuple(x[ito[k], ifr[k]] for k, x in enumerate(fr_slicer) if x is not None)
            if (any(isinstance(i, slice) and i == slice(0, 0) for i in fr_slices) or
                any(isinstance(i, slice) and i == slice(0, 0) for i in to_slices)):
                continue
            if name1 is None:
                iold = (old.name, *ito)
            else:
                iold = (name1, *ito)
            inew = (new.name, *ifr)
            name1 = 'assign1_' + da.core.tokenize(iold, inew)
            dask.update({(name1, *ito): (assign, iold, inew, to_slices, fr_slices)})

        if name1 is None:
            dask.update({(name, *ito): dask[(old.name, *ito)]})
        else:
            dask.update({(name, *ito): dask[(name1, *ito)]})

    return da.Array(dask, name, old.chunks, old.dtype)


def assignirregular(old, locs, new):
    name = 'assign_' + da.core.tokenize(old.name, new.name, locs)

    ndim = old.ndim
    mdim = new.ndim
    new = new[(...,) + (None,) * (ndim - mdim)]
    nblocks = old.numblocks
    mblocks = new.numblocks
    nstep = tuple(max(x) for x in old.chunks)
    mstep = tuple(max(x) for x in new.chunks)
    mshape = new.shape

    def assign(iold, inew, to_pos, *iloc):
        iloc = list(iloc)
        for n, x in enumerate(iloc):
            if hasattr(x, 'ndim'):
                iloc[n] = iloc[n] - to_pos[n] * nstep[n]

        cond = [None]*len(iloc)
        for n, x in enumerate(iloc):
            if hasattr(x, 'ndim'):
                cond[n] = (x >= 0) & (x < nstep[n])
                if all(~cond[n]):
                    return iold
                iloc[n] = iloc[n][cond[n]]
            else:
                cond[n] = x

        inew = inew[cond]
        del cond

        iold2 = iold.copy()
        del iold
        iold2[iloc] = inew
        return iold2

    def recurse(to_pos, fr_pos, ndims, mdims, name1=None):
        if ndims < ndim:
            for i in range(nblocks[ndims]):
                to_pos2 = to_pos + (i,)
                recurse(to_pos2, fr_pos, ndims+1, mdims)
        elif mdims < ndim:
            for j in range(mblocks[mdims]):
                fr_pos2 = fr_pos + (j,)
                name1 = recurse(to_pos, fr_pos2, ndims, mdims+1, name1)
            if mdims == 0:
                if name1 is None:
                    dask[(name, *to_pos)] = dask[(old.name, *to_pos)]
                else:
                    dask[(name, *to_pos)] = dask[(name1, *to_pos)]
                    del dask[(name1, *to_pos)]
            return name1
        else:
            if name1 is None:
                iold = (old.name, *to_pos)
            else:
                iold = (name1, *to_pos)
            iloc = tuple(x if isinstance(x, slice) else (x.name, fr_pos[axis]) for axis, x in enumerate(locs))
            inew = (new.name, *fr_pos)
            name2 = 'assign1_' + da.core.tokenize(iold, inew)
            dask[(name2, *to_pos)] = (assign, iold, inew, to_pos, *iloc)
            return name2

    dask = {}
    dask.update(old.dask)
    dask.update(new.dask)
    for x in locs:
        if hasattr(x, 'dask'):
            dask.update(x.dask)
    recurse((), (), 0, 0)
    return da.Array(dask, name, old.chunks, old.dtype)


def get_index(locs, nstep, nblocks, mshape):
    ndims = len(nblocks)
    fr_indexer = []
    to_indexer = []
    for k in range(ndims):
        x = locs[k]
        nblock = nblocks[k]
        step = nstep[k]
        m = np.arange(mshape[k])
        n = np.arange(nblock)

        fr_index = np.zeros((nblock), object)

        cond = (n * step < x[..., None]) & ((n + 1) * step > x[..., None])
        to_index = (x[..., None] - n * step)[cond]
        if mshape[k] == 1:
            fr_index[...] = np.asarray([0])
        else:
            for i in range(nblock):
                fr_index[i] = m[cond[..., i]]

        fr_indexer.append(fr_index)
        to_indexer.append(to_index)
    return fr_indexer, to_indexer


def get_slice(locs, nsteps, nblocks, mshape):
    ndims = len(nblocks)
    fr_slicer = []
    to_slicer = []
    for k in range(ndims):
        nblock = nblocks[k]
        nstep = nsteps[k]
        if isinstance(locs[k], slice):
            x0 = locs[k].start if locs[k].start is not None else 0
            x1 = locs[k].stop if locs[k].stop is not None else nstep * nblock
        else:
            x0 = locs[k]
            x1 = locs[k] + 1

        fr_slice = np.zeros((nblock), object)
        to_slice = np.zeros((nblock), object)
        for i in range(nblock):
            if x0 < (i + 1) * nstep and x1 > i * nstep:
                if i * nstep < x0:
                    to_start = x0 - i * nstep
                    fr_start = None
                else:
                    to_start = None
                    fr_start = i * nstep - x0

                if (i + 1) * nstep > x1:
                    to_stop = x1 - i * nstep
                    fr_stop = None
                else:
                    to_stop = None
                    fr_stop = (i + 1) * nstep - x0

                to_slice[i] = (slice(to_start, to_stop),)
                if mshape[k] == 1:
                    fr_slice[i] = ()
                else:
                    fr_slice[i] = (slice(fr_start, fr_stop),)
            else:
                to_slice[i] = (slice(0, 0),)
                fr_slice[i] = (slice(0, 0),)
        fr_slicer.append(fr_slice)
        to_slicer.append(to_slice)
    return fr_slicer, to_slicer


def get_blockslice(locs, nsteps, mstep, nblocks, mblocks, mshape):
    ndims = len(nblocks)
    fr_slicer = [None] * ndims
    to_slicer = [None] * ndims
    for k in range(ndims):
        mblock = mblocks[k]
        nblock = nblocks[k]
        nstep = nsteps[k]

        fr_slicer[k] = np.zeros((nblock, mblock), object)
        to_slicer[k] = np.zeros((nblock, mblock), object)
        if nblock == 1 and mblock == 1:
            fr_slicer[k][0, 0] = slice(None)
            try:
                to_slicer[k][0, 0] = locs[k].compute()
            except AttributeError:
                to_slicer[k][0, 0] = locs[k]
        else:
            if isinstance(locs[k], slice):
                x0 = locs[k].start if locs[k].start is not None else 0
                x1 = locs[k].stop if locs[k].stop is not None else nstep * nblock
            else:
                x0 = locs[k]
                x1 = x0 + 1

            for j in range(mblock):
                x0_b = x0 + j * mstep[k]
                x1_b = min(x0 + (j + 1) * mstep[k], x1)
                for i in range(nblock):
                    if x0_b < (i + 1) * nstep and x1_b > i * nstep:
                        if i * nstep < x0_b:
                            to_start = x0_b - i * nstep
                            fr_start = None
                        else:
                            to_start = None
                            fr_start = i * nstep - x0_b

                        if (i + 1) * nstep > x1_b:
                            to_stop = x1_b - i * nstep
                            fr_stop = None
                        else:
                            to_stop = None
                            fr_stop = (i + 1) * nstep - x0_b

                        to_slicer[k][i, j] = slice(to_start, to_stop)
                        if mshape[k] == 1:
                            fr_slicer[k][i, j] = None
                        else:
                            fr_slicer[k][i, j] = slice(fr_start, fr_stop)
                    else:
                        to_slicer[k][i, j] = slice(0, 0)
                        fr_slicer[k][i, j] = slice(0, 0)
    return fr_slicer, to_slicer
