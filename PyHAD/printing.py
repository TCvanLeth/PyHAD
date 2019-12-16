# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:26:00 2016

@author: tcvanleth
"""
import numpy as np

from phad import common as com


def get_printvals(values, size, shape, maxlen=80):
    max_items = max(int(np.ceil(maxlen / 2.0)), 1)

    if isinstance(values, np.ma.MaskedArray):
        values = values.compressed()
    if max_items < size:
        cum_items = np.cumprod(shape[::-1])
        n_steps = np.argmax(cum_items >= max_items)
        stop = int(np.ceil(float(max_items) / np.r_[1, cum_items][n_steps]))
        indexer = ((0, ) * (len(shape) - 1 - n_steps) + (slice(stop), ) +
                   (slice(None), ) * n_steps)
        values = values[indexer]

    if len(values.shape) == 0:
        values = values[None]
    elif not 0 in values.shape:
        values = values.flatten()[:max_items]
    return values


def format_var(values, maxlen=80):
    """
    """
    end_padding = '...'
    pprint_items = format_items(values)
    if len(pprint_items) == 0:
        return '[]'
    cum_len = np.cumsum([len(s) + 1 for s in pprint_items]) - 1
    if (cum_len > maxlen).any():
        count = max(np.argmax((cum_len + len(end_padding)) > maxlen), 1)
        pprint_items = pprint_items[:count] + [end_padding]
    return ' '.join(pprint_items)


def format_items(x):
    """
    Returns a succinct summaries of all items in a sequence as
    strings
    """
    if x.dtype.kind == 'M':
        outstr = [format_timestamp(ix) for ix in x]
    elif x.dtype.kind == 'm':
        day_part = x[~com.isnull(x)].astype('timedelta64[D]')
        if (day_part == np.timedelta64(0, 'D')).all():
            td_format = 'time'
        elif (x == day_part).all():
            td_format = 'date'
        else:
            td_format = 'datetime'

        outstr = [format_timedelta(ix, td_format=td_format) for ix in x]
    elif x.dtype.kind in ('U', 'S'):
        outstr = [repr(ix) for ix in x]
    elif x.dtype.kind in ('f', 'c'):
        outstr = ['{0:.4}'.format(ix) for ix in x]
    else:
        outstr = [str(ix) for ix in x]
    return outstr


def format_timestamp(t):
    """
    turn datetime64 object into a nicely formatted string
    """
    datetime_str = str(t)
    if datetime_str == 'NaT':
        return datetime_str
    str_list = datetime_str.split('.')
    if str_list[1] == '000000':
        str_list = str_list[0].split('T')
        if str_list[1] == '00:00:00':
            return str_list[0]
        return '%sT%s' % tuple(str_list)
    return '%s.%s' % tuple(str_list)


def format_timedelta(t, td_format=None):
    """
    Cast given object to a Timestamp and return a nicely formatted
    string
    """
    timedelta_str = str(t)
    return timedelta_str


def pprint(x, maxlen=80):
    """
    Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = str(x)
    if len(s) > maxlen:
        return s[:(maxlen - 3)] + '...'
    else:
        return s + ' ' * (maxlen - len(s))