# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:06:40 2016

@author: tcvanleth
"""

import dask.array as da
from numba import jit
import numpy as np

from phad.selection import select


def searchdaskuniform(a0, step, n_a, v, how=None, atol=None):
    index = (v - a0) / step
    if how == 'nearest':
        indexer = da.maximum(da.minimum(da.around(index), n_a - 1), 0)
    elif how == 'bfill':
        indexer = da.maximum(da.ceil(index), 0)
    elif how =='ffill':
        indexer = da.minimum(da.floor(index), n_a - 1)
    elif how is None:
        indexer = da.ceil(index)
        indexer = da.where(indexer != index, n_a, indexer)

    if atol is not None:
        indexer = da.where((da.absolute(indexer - index) * step > atol) |
                           (indexer < 0) | (indexer >= n_a), n_a, indexer)
    else:
        indexer = da.where((indexer < 0) | (indexer >= n_a), n_a, indexer)
    return indexer.astype(int)


def searchdask(a, v, how=None, atol=None):
    n_a = a.shape[0]
    searchfunc, args = presearch(a, v)

    if how == 'nearest':
        l_index = da.maximum(searchfunc(*args, side='right') - 1, 0)
        r_index = da.minimum(searchfunc(*args), n_a - 1)
        cond = 2 * v < (select(a, r_index) + select(a, l_index))
        indexer = da.maximum(da.where(cond, l_index, r_index), 0)
    elif how == 'bfill':
        indexer = searchfunc(*args)
    elif how == 'ffill':
        indexer = searchfunc(*args, side='right') - 1
        indexer = da.where(indexer == -1, n_a, indexer)
    elif how is None:
        l_index = searchfunc(*args)
        r_index = searchfunc(*args, side='right')
        indexer = da.where(l_index == r_index, n_a, l_index)
    else:
        return NotImplementedError

    if atol is not None:
        a2 = da.concatenate([a, [atol + da.max(v) + 1]])
        indexer = da.where(da.absolute(select(a2, indexer) - v) > atol, n_a, indexer)
    return indexer


def searchsorted(a, v, side='left'):
    searchfunc, args = presearch(a, v)
    return searchfunc(*args, side=side)


def presearch(a, v):
    if a.ndim > 1:
        a = a.squeeze()

    n_a = a.shape[0]
    step_a = tuple(max(x) for x in a.chunks)[0]
    if not any(ichunk[0] < ishape for ichunk, ishape in zip(v.chunks, v.shape)):
        if a.chunks[0][0] >= n_a:
            args = (a, v)
            searchfunc = searchsingle
        else:
            a_block = a[np.arange(0, n_a, step_a)].rechunk(len(a.chunks[0]))
            b_indxrs = da.maximum(searchsingle(a_block, v) - 1, 0).compute()
            if v.size == 1:
                args = (a, v, b_indxrs[0], step_a)
                searchfunc = searchblock
            else:
                firsts = np.insert(np.diff(b_indxrs).nonzero()[0] + 1, 0, 0)
                b_indxrs = b_indxrs[firsts]
                firsts = np.append(firsts, len(v))
                slicers = [slice(firsts[i], firsts[i+1])
                           for i in range(len(firsts) - 1)]
                args = (a, v, b_indxrs, step_a, slicers)
                searchfunc = searchblocks
    else:
        step_v = tuple(max(x) for x in v.chunks)[0]

        a_block = a[np.arange(0, n_a, step_a)].rechunk(len(a.chunks[0]))
        v_block = v[np.arange(0, v.shape[0], step_v)].rechunk(len(v.chunks[0]))
        b_indxrs = da.maximum(searchsingle(a_block, v_block) - 1, 0).compute()
        b_indxrs = np.append(b_indxrs, len(a_block) - 1)
        args = (a, v, b_indxrs, step_a, a_block.compute())
        searchfunc = searchindexblocks
    return searchfunc, args


def searchsingle(a, v, side='left'):
    name = 'search-' + side + '-' + da.core.tokenize([a.name, v.name])

    def func2(ja, iv):
        return ja.searchsorted(iv, side=side)

    v_pos = (0,) * v.ndim
    dask = {(name, *v_pos): (func2, (a.name, 0), (v.name, *v_pos))}
    dask.update(a.dask)
    dask.update(v.dask)

    chunks = v.chunks
    dtype = np.int64
    return da.Array(dask, name, chunks, dtype)


def searchblock(a, v, b_indxr, step, side='left'):
    name = 'search-' + side + '-' + da.core.tokenize([a.name, v.name])

    def func2(ja, iv, j):
        if j == -1:
            return np.int_(0)
        index = ja.searchsorted(iv, side=side)
        index += j * step
        return index

    dask = {(name, 0): (func2, (a.name, b_indxr), (v.name, 0), b_indxr)}
    dask.update(a.dask)
    dask.update(v.dask)

    chunks = ((1,), )
    dtype = np.int64
    return da.Array(dask, name, chunks, dtype)


def searchblocks(a, v, b_indxrs, step, slicers, side='left'):
    name2 = 'index-' + side + '-' + da.core.tokenize([a.name, v.name])
    name = 'search-' + side + '-' + da.core.tokenize(name2)

    def func2a(ja, iv, j, sli):
        ijv = iv[sli]
        index = ja.searchsorted(ijv, side=side)
        index += j * step
        return index

    def func2b(iv, sli):
        ijv = iv[sli]
        return np.zeros(len(ijv), dtype=int)
    
    dask = dict.merge(a.dask, v.dask)
    dask.update({(name2, j): (func2a, (a.name, j), (v.name, 0), j, sli)
            if j != -1
            else (func2b, (v.name, 0), sli)
            for j, sli in zip(b_indxrs, slicers)})
    dask.update({(name, 0): (np.concatenate, [(name2, j)
                for j in b_indxrs])})

    chunks = v.chunks
    dtype = np.int64
    return da.Array(dask, name, chunks, dtype)


def searchindexblocks(a, v, bb_indxrs, step, a_block, side='left'):
    n = v.numblocks[0]

    name1 = 'concat-' + side + '-' + da.core.tokenize([a.name, v.name])
    name2 = 'index_' + side + '-' + da.core.tokenize(name1)
    name = 'search-' + side + '-' + da.core.tokenize(name2)

    def func1(iv, i):
        a_block1 = a_block[bb_indxrs[i]: bb_indxrs[i+1]+1]
        m = len(a)
        n = len(a_block1)
        slicers = np.insert(iv.searchsorted(a_block1), [0, n], [0, m])
        return [slice(slicers[j], slicers[j+1]) for j in range(len(slicers)-1)]

    def func2a(ja, iv, j, sli):
        ijv = iv[sli[j]]
        index = ja.searchsorted(ijv, side=side)
        index += j * step
        return index

    def func2b(iv, j, sli):
        ijv = iv[sli[j]]
        return np.zeros(len(ijv), dtype=int)

    def func3(indices, i):
        index = np.concatenate(indices)
        return index + bb_indxrs[i] * step

    dask = {}
    dask.update({(name1, i): (func1, (v.name, i), i)
                for i in range(n)})
    dask.update({(name2, i, j): (func2a, (a.name, j), (v.name, i), j, (name1, i))
                if j != -1
                else (func2b, (v.name, i), j, (name1, i))
                for i in range(n)
                for j in range(bb_indxrs[i], bb_indxrs[i+1]+1)})
    dask.update({(name, i): (func3, [(name2, i, j)
                for j in range(bb_indxrs[i], bb_indxrs[i+1]+1)], i)
                for i in range(n)})
    dask.update(a.dask)
    dask.update(v.dask)

    chunks = v.chunks
    dtype = np.int64
    return da.Array(dask, name, chunks, dtype)


###############################################################################
def searchuniform(a0, step, n_a, v, how='bfill'):
    indexer = (v - a0) / step
    if how == 'nearest':
        indexer = np.around(indexer)
    elif how == 'bfill':
        indexer = np.floor(indexer)
    elif how =='ffill':
        indexer = np.ceil(indexer)
    indexer = np.minimum(np.maximum(indexer.astype(int), 0), n_a - 1)
    return indexer


def searchirregular(a, v, how='bfill'):
    args = (a, v)
    n_a = a.shape[0]
    if how == 'nearest':
        l_index = np.searchsorted(*args, side='right') - 1
        r_index = np.searchsorted(*args)
        a = np.concatenate([a, np.asarray([np.inf])])
        cond = 2*v < (select(a, r_index) + select(a, l_index))
        indexer = np.maximum(np.where(cond, l_index, r_index), 0)
    elif how == 'bfill':
        indexer = np.searchsorted(*args)
    elif how == 'ffill':
        indexer = np.maximum(np.searchsorted(*args, side='right') - 1, 0)
    indexer = np.minimum(indexer, n_a - 1)
    return indexer

def dask_is_increasing(arr):
    n = arr.numblocks[0]
    s = -(-len(arr)//n)

    inc = is_increasing(arr[:s].compute())
    if not inc:
        return False
    for i in range(1, n):
        if arr[i*s] <= arr[i*s-1]:
            return False
        inc = is_increasing(arr[i*s:(i+1)*s].compute())
        if not inc:
            return False
    return True

@jit
def is_increasing(arr):
    n = len(arr)
    prev = arr[0]
    for i in range(0, n):
        cur = arr[i]
        if cur <= prev:
            return False
        prev = cur
    return True


@jit
def binsearch(a, v, how):
    n = len(a)
    indices = np.zeros(len(v), int)
    index = 0
    for j, jv in enumerate(v):
        left = index
        right = n-1
        while left <= right:
            i = left + (right-left)//2
            ai = a[i]
            if ai < jv:
                left = i+1
            elif ai > jv:
                right = i-1
            else:
                index = i
                break
        else:
            index = i
        indices[j] = index
    return indices


###############################################################################
def ffill(a, v, limit=None):
    fill_count = 0
    nleft = len(a)
    nright = len(v)
    indexer = np.empty(nright, dtype=np.int64)
    indexer.fill(-1)

    if limit is None:
        lim = nright
    else:
        if limit < 0:
            raise ValueError('Limit must be non-negative')
        lim = limit

    if nleft == 0 or nright == 0 or v[-1] < a[0]:
        return indexer

    i = j = 0
    cur = a[i]
    while j < nright and v[j] < cur:
        j += 1

    while True:
        if j == nright:
            break
        if i == nleft - 1:
            while j < nright:
                if v[j] == cur:
                    indexer[j] = i
                elif v[j] > cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                j += 1
            break

        next = a[i + 1]
        while j < nright and cur <= v[j] < next:
            if v[j] == cur:
                indexer[j] = i
            elif fill_count < lim:
                indexer[j] = i
                fill_count += 1
            j += 1

        fill_count = 0
        i += 1
        cur = next
    return indexer


def bfill(a, v, limit=None):
    fill_count = 0
    nleft = len(a)
    nright = len(v)
    indexer = np.empty(nright, dtype=np.int64)
    indexer.fill(-1)

    if limit is None:
        lim = nright
    else:
        if limit < 0:
            raise ValueError('Limit must be non-negative')
        lim = limit

    if nleft == 0 or nright == 0 or v[0] > a[nleft - 1]:
        return indexer

    i = nleft - 1
    j = nright - 1
    cur = a[i]
    while j >= 0 and v[j] > cur:
        j -= 1

    while True:
        if j < 0:
            break
        if i == 0:
            while j >= 0:
                if v[j] == cur:
                    indexer[j] = i
                elif v[j] < cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                j -= 1
            break

        prev = a[i - 1]
        while j >= 0 and prev < v[j] <= cur:
            if v[j] == cur:
                indexer[j] = i
            elif v[j] < cur and fill_count < lim:
                indexer[j] = i
                fill_count += 1
            j -= 1

        fill_count = 0
        i -= 1
        cur = prev
    return indexer


def get_indexer_algo(values, vals, dtype):
    n = len(values)
    m = len(vals)
    locs = np.empty(m, dtype=np.int64)
    if np.issubdtype(dtype, int):
        table = kh_init_int64()
        kh_resize_int64(table, n)
        with nogil:
            for i in range(n):
                k = kh_put_int64(table, values[i])
                table.vals[k] = i
            for i in range(m):
                k = kh_get_int64(table, vals[i])
                if k != table.n_buckets:
                    locs[i] = table.vals[k]
                else:
                    locs[i] = -1
    elif np.issubdtype(dtype, float):
        table = kh_init_float64()
        kh_resize_float64(table, n)
        with nogil:
            for i in range(n):
                k = kh_put_float64(table, values[i])
                table.vals[k] = i
            for i in range(m):
                k = kh_get_float64(table, vals[i])
                if k != table.n_buckets:
                    locs[i] = table.vals[k]
                else:
                    locs[i] = -1
    return np.asarray(locs)


def get_slice_bound(values, label):
    if label is None:
        return None
    if np.issubdtype(label.dtype, float) and np.issubdtype(values.dtype, int):
        ckey = int(label)
        if ckey == label:
            label = ckey
    return values.searchsorted(label)