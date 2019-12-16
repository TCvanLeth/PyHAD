# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:50:00 2015

@author: T.C. van Leth

functions and classes for dealing with the serialised WURex database
using hdf5.

TO-DO: support discontinuous time span
"""

import datetime as dt
import gc
from pathlib import Path
import shutil

import dask.array as da
import h5py
import numpy as np
import yaml

from phad import common


# hdf5 store
def to_hdf(data, outpath):
    """
    write container object into hdf5 file
    """
    outpath = Path(outpath)

    # make sure target directory exists
    outdir = outpath.parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    outpath = outpath.with_suffix('.h5')
    tmppath = outpath.with_suffix('.h5.temp')
    if tmppath.exists():
        tmppath.unlink()

    with h5py.File(tmppath, mode='a') as outh:
        sources, targets = data._create_store(outh)
        da.store(sources, targets)

    # replace old file with temp file
    if outpath.exists():
        outpath.unlink()
    tmppath.rename(outpath)


def to_yaml(data, outpath):
    """
    write container object into yaml file
    """
    outpath = Path(outpath)

    # make sure target directory exists
    outdir = outpath.parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    outpath = outpath.with_suffix('.yaml')
    tmppath = outpath.with_suffix('.yaml.temp')
    if tmppath.exists():
        tmppath.unlink()
    if outpath.exists():
        shutil.copyfile(outpath, tmppath)

    # serialize to temporary file
    with open(tmppath, mode='a') as outh:
        store = {}
        data.to_store(store)
        yaml.dump(store, stream=outh)

    # replace old file with temp file
    if outpath.exists():
        outpath.unlink()
    tmppath.rename(outpath)


def from_hdf(cls, inpath, **kwargs):
    """
    read hdf5 file into container object
    """
    inpath = Path(inpath)

    inpath = inpath.with_suffix('.h5')
    if not inpath.exists():
        raise IOError('path "%s" does not exist!' % inpath)

    inh = h5py.File(inpath, mode='r')
    data = cls.from_store(inh, **kwargs)
    data.handles |= {inh}
    return data


def from_yaml(cls, inpath, **kwargs):
    """
    read yaml file into container object
    """
    inpath = Path(inpath)

    inpath = inpath.with_suffix('.yaml')
    if not inpath.exists():
        raise IOError('path "%s" does not exist!' % inpath)

    with open(inpath, 'r') as inh:
        store = yaml.safe_load(inh)
    return cls.from_yaml(store, **kwargs)


def closeall():
    for obj in gc.get_objects():   # Browse through ALL objects
        if type(obj) == h5py.File:   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

def isscale(leaf):
    attrs = leaf.attrs
    return ('CLASS' in attrs and
            attrs['CLASS'] == b'DIMENSION_SCALE')


def isindex(leaf):
    attrs = leaf.attrs
    return ('CLASS' in attrs and
            attrs['CLASS'] == b'DIMENSION_SCALE' and
            attrs['array_id'] == leaf.dims[0].label)

def iscoord(leaf):
    attrs = leaf.attrs
    return ('CLASS' in attrs and
            attrs['CLASS'] == b'DIMENSION_SCALE' and not
            attrs['array_id'] == leaf.dims[0].label)

def isvar(leaf):
    attrs = leaf.attrs
    return 'CLASS' not in attrs or attrs['CLASS'] != b'DIMENSION_SCALE'


###############################################################################
# timescale manipulation
def select_time(data, days):
    if days is not None and len(data):
        tstep = common.str_to_td64(data.attrs['temporal_resolution'])
        regtime = []
        for iday in days:
            start = iday
            end = iday + np.timedelta64(1, 'D')
            regtime += [np.arange(start, end, tstep)]
        if regtime:
            regtime = np.hstack(regtime)
        data = data.sel(time=regtime)
    return data


def itersearch(grph, key):
    if key in grph:
        return grph[key]
    else:
        for subh in grph.values():
            found = itersearch(subh, key)
            if found is not None:
                return found


def checktimes(times):
    # when no selection is given, select all dates from start of experiment
    if times is None:
        today = np.datetime64(dt.date.today())
        timespan = (np.datetime64('2014-01-01'), today)
        # TO-DO: flexible begin time
    else:
        timespan = times

    # convert input daterange to appropriate types
    if isinstance(timespan, tuple) and len(timespan) == 2:
        days = np.arange(timespan[0], timespan[1], np.timedelta64(1, 'D'))
    elif isinstance(timespan, slice):
        days = np.arange(timespan.start, timespan.stop, np.timedelta64(1, 'D'))
    elif isinstance(timespan, (list, np.ndarray)):
        days = timespan
    else:
        raise TypeError('unknown type for timespan!')
    return days


###############################################################################
## what about partially complete days?!
#def checkdates(level, setID, period, proID=None, daily=False, station=None):
#    path = getpath(level, setID, proID=proID)
#
#    # get the first and last day of the requested period
#    sday = np.datetime64(period[0], 'ns')
#    eday = np.datetime64(period[1], 'ns')
#    dates = np.arange(sday, eday, np.timedelta64(1, 'D'))
#
#    # search files
#    if daily is True:
#        cond = np.zeros(len(dates), dtype='bool_')
#        for i, thisday in enumerate(dates):
#            inpath = makename(setID, thisday, path)
#
#            if os.path.exists(inpath+'.h5'):
#                if station is None:
#                    cond[i] = True
#                else:
#                    with h5py.File(inpath+'.h5', mode='r') as inh:
#                        if station in inh:
#                            cond[i] = True
#        return dates[~cond]
#
#    if daily is False:
#        # look for the date in a hdf5 file
#        if os.path.exists(path+'.h5'):
#            with h5py.File(path+'.h5', mode='r') as inh:
#                if station is not None:
#                    if station in inh:
#                        inh = inh[station]
#                    else:
#                        return dates
#
#                time = itersearch(inh, 'time')
#                time = np.datetime64(time[:], time.attrs['unit'])
#                cond = np.in1d(dates, time)
#                return dates[~cond]
#        else:
#            return dates
