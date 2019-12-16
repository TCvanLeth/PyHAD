# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:38:46 2016

@author: tcvanleth
"""
from collections import OrderedDict

import dask.array as da

import phad.common as com


def align(datlist):
    """
    align a list of channels along there indexes.

    NOTE: currently only aligns along 'uniform' indexes. Other indexes are
    checked for equality instead.
    """
    if len(datlist) == 1:
        return datlist

    steps = get_highest_res(datlist)
    datlist = upsample(datlist, steps)
    slices = intersect(datlist)
    datlist = sel(datlist, slices)
    return datlist


def get_highest_res(datlist):
    dd = com.defaultdict(list)
    for idat in datlist:
        if hasattr(idat, 'indexes'):
            for iindex in idat.indexes.values():
                dd[iindex.name].append(iindex)

    steps = {}
    for iID, iindex in dd.items():
        if iindex[0].step is not None:
            steps[iID] = min([i.step for i in iindex])
        elif not var_equal(*tuple(iindex)):
            raise IndexError('index "%s" is irregular and not equal across \
                              all inputs' % iID)
    return steps


def get_highest_res2(datlist):
    dd = com.defaultdict(OrderedDict)
    for i, idat in datlist.items():
        if hasattr(idat, 'indexes'):
            for iindex in idat.indexes.values():
                dd[iindex.name][i] = iindex

    steps = {}
    for iID, iindex in dd.items():
        if next(iter(iindex.values())).step is not None:
            steps[iID] = min([i.step for i in iindex.values()])
    return steps


def upsample(datlist, steps):
    newdat = []
    for idat in datlist:
        if hasattr(idat, 'indexes'):
            newdat.append(idat.upsample(**steps))
        else:
            newdat.append(idat)
    return newdat


def upsample2(datlist, steps):
    newdat = OrderedDict()
    for i, idat in datlist.items():
        if hasattr(idat, 'indexes'):
            newdat[i] = idat.upsample(**steps)
        else:
            newdat[i] = idat
    return newdat


def intersect(datlist):
    dd = com.defaultdict(list)
    for idat in datlist:
        if hasattr(idat, 'indexes'):
            for iindex in idat.indexes.values():
                dd[iindex.name].append(iindex)

    slices = {}
    for iID, iindex in dd.items():
        if iindex[0].step is not None:
            slices[iID] = intersect_index(iindex)
        elif not var_equal(*tuple(iindex)):
            raise IndexError('index %s is irregular and not equal across \
                              all inputs' % iID)
    return slices


def intersect_index(indexes):
    x0 = max(x.start for x in indexes)
    x1 = min(x.stop for x in indexes)
    return slice(x0, x1)


def sel(datlist, slices):
    newdat = []
    for idat in datlist:
        if hasattr(idat, 'indexes'):
            newdat.append(idat.sel(**slices))
        else:
            newdat.append(idat)
    return newdat

def var_equal(*args):
    for iarg in args[1:]:
        if iarg is args[0]:
            continue
        if iarg.shape != args[0].shape:
            return False
        if iarg.dtype.kind == 'f':
            if da.any(~da.isclose(iarg._values, args[0]._values)):
                return False
        else:
            if da.any(iarg._values != args[0]._values):
                return False
    return True
