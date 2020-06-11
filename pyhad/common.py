# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:52:41 2015

@author: T.C. van Leth
"""
from collections import OrderedDict
from enum import Enum
import datetime as dt
import inspect
import logging
import numexpr
import psutil

import dask.array as da
import numpy as np


CHUNKSIZE = 200000


def chunk(times, chunk=30, tstep=0.01):
    """
    seperate input times into appropriately sized chunks
    """
    mem = psutil.virtual_memory().available
    mem = mem/(20*1024*1024*1024)  # dimensionless memory
    chunk = int(chunk * mem)  # scale chunksize with available memory

    today = dt.datetime.today()
    if isinstance(times, dt.datetime):
        span = (times, today)
    else:
        assert isinstance(times, tuple)
        span = times
    itime = (span[0], span[0])
    while True:
        itime = (itime[1], itime[1] + dt.timedelta(days=chunk))
        itime2 = (itime[0], itime[1] - dt.timedelta(seconds=tstep))
        if itime[0] >= today or itime[0] >= span[1]:
            break
        yield itime2


def where(condition):
    """
    """
    ch = condition.chunks[0]
    n = len(ch)
    a = 'a'+da.core.tokenize(condition)
    b = 'where'+da.core.tokenize(condition)
    def func(x):
        try:
            return next(ix for ix in x if len(ix)>0)
        except StopIteration:
            return np.int_(-1)

    parts = [(a, i) for i in range(n)]
    dask = {(a, i): (lambda x, y: np.where(x)[0]+y*ch[y], (condition.name, i), i) for i in range(n)}
    dask.update({(b, 0): (func, parts)})
    dask.update(condition.dask)
    dtype = int
    chunks = ((1,),)
    return da.Array(dask, b, chunks, dtype)


###############################################################################
def get_fill(dtype):
    """
    define fill values for different data types
    """
    if np.issubdtype(dtype, np.floating):
        fill_value = np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        fill_value = np.nan + np.nan*1j
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64('NaT')
    elif np.issubdtype(dtype, np.timedelta64):
        fill_value = np.timedelta64('NaT')
    elif np.issubdtype(dtype, np.str_):
        fill_value = 'NA'
    elif np.issubdtype(dtype, np.signedinteger):
        fill_value = -9999
    else:
        raise NotImplementedError(dtype)
    return fill_value


def isnull(array, **kwargs):
    array = to_nptype(array)
    if array is None:
        return True
    dtype = array.dtype
    fill_value = get_fill(dtype)
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        if isinstance(array, da.Array):
            return da.isnan(array)
        return np.isnan(array)
    else:
        return array == fill_value


def null_array(shape, dtype):
    if np.issubdtype(dtype, str):
        dtype = np.dtype((str, 64))
    array = np.empty(shape, dtype=dtype)
    array.fill(get_fill(dtype))
    return array


def nan_equal(a, b, **kwargs):
    if np.issubdtype(a.dtype, '<M8'):
        return np.all(a == b)
    return np.all(numexpr.evaluate('(a==b)|((a!=a)&(b!=b))'))


def array_equal(a, b, **kwargs):
    if a is b:
        return True
    elif a.shape != b.shape:
        return False
    else:
        return nan_equal(a, b)


def isint(label):
    if isinstance(label, int):
        return True
    elif isinstance(label, slice):
        return (isinstance(label.stop, (int, type(None))) and
                isinstance(label.stop, (int, type(None))) and
                isinstance(label.step, (int, type(None))))
    else:
        False


def np_to_base_type(data):
    if is_dict_like(data):
        newdat = type(data)()
        for key, value in data.items():
            newdat[key] = np_to_base_type(value)
    elif isinstance(data, list):
        newdat = type(data)(np_to_base_type(i) for i in data)
    elif hasattr(data, 'dtype'):
        newdat = data.tolist()
    else:
        newdat = data
    return newdat


def to_nptype(data):
    if hasattr(data, 'dtype'):
        return data
    if isinstance(data, tuple):
        return tuple(to_nptype(x) for x in data)
    elif isinstance(data, list):
        return np.asarray(data)
    elif isinstance(data, bool):
        return np.bool_(data)
    elif isinstance(data, int):
        return np.int_(data)
    elif isinstance(data, float):
        return np.float_(data)
    elif isinstance(data, complex):
        return np.complex_(data)
    elif isinstance(data, str):
        return np.str_(data)
    elif isinstance(data, dt.datetime):
        return np.datetime64(data)
    elif isinstance(data, dt.timedelta):
        return np.timedelta64(data)
    elif data is None:
        return data
    else:
        raise ValueError('unknown type %s with value %s' % (type(data), data))


def conform_type(data):
    newdat = to_nptype(data)
    if newdat.dtype.kind == 'U':
        newdat = newdat.astype('<U64')
    return newdat


def isbasetype(data):
    basetypes = (int, float, str, bool)
    return isinstance(data, basetypes)


###############################################################################
class Error(Exception):
    pass


class defaultdict(dict):
    def __init__(self, factory, *args, **kwargs):
        dict.__init__(self)
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def __missing__(self, key):
        self[key] = self.factory(*self.args, **self.kwargs)
        return self[key]


def standardlogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)


def indent(string, spaces):
    return "\n".join((spaces * " ") + i for i in string.splitlines())


def is_dict_like(value):
    return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def is_scalar(value):
    """np.isscalar only works on primitive numeric types and (bizarrely)
    excludes 0-d ndarrays; this version does more comprehensive checks
    """
    if hasattr(value, 'ndim'):
        return value.ndim == 0
    return (np.isscalar(value) or
            isinstance(value, (dt.datetime, dt.date, dt.timedelta)) or
            value is None)


def prep_merger(datlist):
    assert(isinstance(datlist, (OrderedDict, list)))
    if isinstance(datlist, list):
        datlist = OrderedDict({k: v for (k, v) in enumerate(datlist)
                               if v is not None})
    return datlist


def get_kwarg_names(function):
    """ Return a list of keyword argument names function accepts. """
    sig = inspect.signature(function)
    keywords = []
    for param in sig.parameters.values():
        if(param.kind == inspect.Parameter.KEYWORD_ONLY or
            (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                param.default != inspect.Parameter.empty)):
            keywords.append(param.name)
    return keywords


def splitkwargs(func, kwargs):
    a = {k: v for k, v in kwargs.items() if k in get_kwarg_names(func)}
    b = {k: v for k, v in kwargs.items() if k not in a}
    return a, b


###############################################################################
# common time manipulations
class UTC(dt.tzinfo):
    def tzname(self):
        return "UTC"

    def utcoffset(self, idt):
        return dt.timedelta(0)


def dt64_to_dt(dt64):
    ns = 1e-9
    return dt.datetime.utcfromtimestamp(dt64.astype('uint64') * ns)


def dt64_to_posix(time):
    us = 1e-6
    return time.astype('uint64') * us

def time_to_micros(time):
    seconds = time.hour * 60 * 60 + 60 * time.minute + time.second
    return 1000000 * seconds + time.microsecond


def posix_to_dt64(time):
    us = 1e-6
    return (time / us).astype('<M8[us]')


def dt_to_posix(time):
    epoch = dt.datetime(1970, 1, 1, tzinfo=UTC())
    return (time-epoch).total_seconds()


def roundtime(time, step):
    seconds = (time-time.min).seconds
    rounding = (seconds+step/2)//step*step
    return time+dt.timedelta(0, rounding-seconds, -dt.microsecond)


def str_to_td64(string):
    """
    convert pandas style frequency string into timedelta64
    """
    if isinstance(string, np.timedelta64):
        return string
    if isinstance(string, (str, np.str_)):
        tail = string.lstrip('0123456789')
    elif isinstance(string, (bytes, np.bytes_)):
        tail = string.lstrip(b'0123456789')
    return np.timedelta64(string[:-len(tail)], tail)

def str_to_latex(string):
    """
    convert pandas style frequency string into latex compatible unit expression
    """
    if isinstance(string, (str, np.str_)):
        tail = string.lstrip('0123456789')
    elif isinstance(string, (bytes, np.bytes_)):
        tail = string.lstrip(b'0123456789')

    if tail == 'm':
        unit = 'min'
    else:
        unit = tail
    return '$%s\,\mathrm{%s}$' % (string[:-len(tail)], unit)

def str_to_sec(string):
    """
    convert pandas style frequency string into number of seconds
    """
    td64 = str_to_td64(string)
    return td64/np.timedelta64(1, 's')


def time_component(string):
    if string == 'date':
        return '1D'
    elif string == 'hour':
        return '1h'
    elif string == 'minute':
        return '1m'
    elif string == 'second':
        return '1s'


def dt64_to_slice(time):
    if time.dtype.kind == 'M' and time.dtype != '<M8[us]':
        return slice(time, (time+1))
    return time


# candidates for deprecation
###############################################################################
#def smerge(data, ID_tag=None):
#    """
#    merge sequence of Dataset derived objects into one dataset
#    """
#    assert isinstance(data[0], cb.Channel)
#
#    if ID_tag is not None:
#        mdat = data[0]
#        ID = mdat.attrs[ID_tag]
#        for vID in mdat.data_vars.keys():
#            mdat = mdat.rename({vID: ID+'_'+vID})
#        for idat in data[1:]:
#            ID = idat.attrs[ID_tag]
#            for vID in idat.data_vars.keys():
#                idat = idat.rename({vID: ID+'_'+vID})
#            mdat = mdat.merge(idat)
#    else:
#        mdat = data[0]
#        for idat in data[1:]:
#            mdat = mdat.merge(idat)
#
#    mdat.attrs = merge_attrs(data)
#    return mdat