#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:14:44 2019

@author: tcvanleth
"""
import dask.array as da
import phad.common as com


def reshape(a, nstep, mstep, chunksize, aligned=0):
    shape = a.shape
    dtype = a.dtype

    if aligned != 0:
        newshape = shape[:-1] + (aligned,)
        fill = da.full(newshape, com.get_fill(dtype), dtype=dtype,
                       chunks=chunksize)
        a = da.concatenate((fill, a), axis=-1)
        shape = a.shape

    fill_len = (mstep - shape[-1] % mstep) % mstep
    if fill_len != 0:
        newshape = shape[:-1] + (fill_len,)
        fill = da.full(newshape, com.get_fill(dtype), dtype=dtype,
                       chunks=chunksize)
        a = da.concatenate((a, fill), axis=-1)

    newshape = shape[:-1] + (nstep, mstep)
    a = a.reshape(newshape).rechunk(chunksize)
    return a


