# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:12:08 2016

@author: tcvanleth
"""
from datetime import datetime
import string

import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as pl
from scipy import stats

from phad import analysis
from phad import common
from phad import ufuncs as uf


def plot(x, y, style='o', dim=None, log=False, semilogx=False, semilogy=False,
         ax=None, aligned=False, fig=None, lloc='best', **kwargs):
    xdim = x.dims
    ydim = y.dims
    if xdim != ydim:
        err = 'unequal dimensions: %s and %s' % (str(xdim), str(ydim))
        raise Exception(err)
    if len(xdim) > 1 and dim is None:
        err = 'more than 1 dimension: %s' % str(xdim)
        raise Exception(err)

    labels, kwargs = get_labels(x, y, **kwargs)
    if x.attrs['quantity'] == y.attrs['quantity']:
        kwargs['figsize'] = pl.figaspect(1)
    elif x.attrs['quantity'] == 'time':
        kwargs['figsize'] = (15, 5)

    if aligned == False:
        x, y = uf.broadcast(x, y)
    fig_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.figure)}
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplot)}
    plot_kwargs = {k: v for k, v in kwargs.items()
                   if k not in {**subp_kwargs, **fig_kwargs}}

    if x.dtype.kind == 'M':
        x = x.astype(datetime)
    if y.dtype.kind == 'M':
        y = y.astype(datetime)

    if ax is None:
        if fig is None:
            fig = pl.figure(**fig_kwargs)
        ax = fig.add_subplot(111, **subp_kwargs)

    if log or semilogx:
        x = uf.log(x)

    if log or semilogy:
        y = uf.log(y)

    if dim is not None:
        x = x.split(dim)
        y = y.split(dim)
        for ix, iy in zip(x, y):
            line, = ax.plot(ix[1].values, iy[1].values, style, label=ix[0],
                            **plot_kwargs)
    else:
        line, = ax.plot(x.values, y.values, style, **plot_kwargs)

    # layout
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if log:
        def f(x, pos):
            return '{:0.2e}'.format(uf.exp(x))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))

    ax.legend(loc=lloc)
    pl.tight_layout()
    return fig, ax


def barplot(x, y, width=None, ax=None, fig=None, aligned=False, **kwargs):
    labels, kwargs = get_labels(x, y, **kwargs)
    if aligned == False:
        x, y = uf.broadcast(x, y)
    fig_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.figure)}
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplot)}
    plot_kwargs = {k: v for k, v in kwargs.items()
                   if k not in subp_kwargs}

    if x.dtype.kind == 'M':
        x = x.astype(datetime)
    if y.dtype.kind == 'M':
        y = y.astype(datetime)

    if ax is None:
        if fig is None:
            fig = pl.figure(**fig_kwargs)
        ax = fig.add_subplot(111, **subp_kwargs)

    x = x.values
    if width is None:
        width = 0.5 * x[1] - x[0]
    else:
        try:
            width = width.values
        except AttributeError:
            pass

    ax.bar(x, y.values, align='center',  width=width, **plot_kwargs)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.grid=True
    pl.tight_layout()
    return fig, ax


def pixplot(x, y, z, log=False, interp='nearest', aligned=False, **kwargs):
    labels, kwargs = get_labels(x, y, z, **kwargs)
    if aligned == False:
        x, y = uf.broadcast(x, y)
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplots)}
    imsh_kwargs = {k: v for k, v in kwargs.items()
                   if k not in subp_kwargs}

    if x.dtype.kind == 'M':
        xlims = mdates.date2num([x.min().values.astype(datetime).item(),
                                 x.max().values.astype(datetime).item()])
    else:
        xlims = (min(x).values.item(), max(x).values.item())

    if y.dtype.kind == 'M':
        ylims = mdates.date2num([y.min().values.astype(datetime).item(),
                                 y.max().values.astype(datetime).item()])
    else:
        ylims = [y.min().values.item(), y.max().values.item()]

    fig, ax = pl.subplots(**subp_kwargs)

    if log:
        z = uf.log10(z)
    img = ax.imshow(z.values, interpolation=interp,
                    extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                           origin='lower', aspect='auto', **imsh_kwargs)

    # layout
    if x.dtype.kind == 'M':
        ax.xaxis_date()
        fig.autofmt_xdate()
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(labels[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.grid = True
    pl.tight_layout()
    return fig, ax


def contourplot(x, y, z, log=False, levels=100, aligned=False, maxz=None,
                minz=None, zlog=False, ax=None, fig=None, **kwargs):
    labels, kwargs = get_labels(x, y, z, **kwargs)

    if aligned == False:
        x, y = uf.broadcast(x, y)
    fig_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.figure)}
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplots)}


    if x.dtype.kind == 'M':
        x = x.astype(datetime)
    if y.dtype.kind == 'M':
        y = y.astype(datetime)

    if ax is None:
        if fig is None:
            fig = pl.figure(**fig_kwargs)
        ax = fig.add_subplot(111, **subp_kwargs)
    else:
        fig = ax.figure
        if ax.get_xlabel() == labels[1] or ax.get_ylabel() == labels[0]:
            tmp = x.copy()
            x = y.copy()
            y = tmp
            tmp = labels[0]
            labels[0] = labels[1]
            labels[1] = tmp
            z = z.T

    if maxz is None:
        minz = 0
        maxz = uf.nanmax(z).values
    if zlog:
        levels = np.logspace(minz, maxz, levels)
    else:
        levels = np.linspace(minz, maxz, levels)

    z = z.values
    z = np.ma.masked_array(z, mask=np.isnan(z))
    img = ax.contour(x.values, y.values, z, levels=levels, **kwargs)

    # layout
    if log:
        def f(x, pos):
            return '{:0.2e}'.format(uf.exp(x))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))
    if x.dtype.kind == 'M':
        fig.autofmt_xdate()
    cbar = fig.colorbar(img, ax=ax, format='%.1e')
    cbar.set_label(labels[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.grid = True
    pl.tight_layout()
    return fig, ax


def meshplot(x, y, z, log=False, levels=100, aligned=False, **kwargs):
    labels, kwargs = get_labels(x, y, z, **kwargs)
    if aligned == False:
        x, y = uf.broadcast(x, y)
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplots)}
    cont_kwargs = {k: v for k, v in kwargs.items()
                   if k not in subp_kwargs}

    if x.dtype.kind == 'M':
        x = x.astype(datetime)
    if y.dtype.kind == 'M':
        y = y.astype(datetime)

    fig, ax = pl.subplots(**subp_kwargs)
    levels = np.linspace(0, uf.nanmax(z).values, levels)
    img = ax.pcolormesh(x.values, y.values, z.values, **cont_kwargs)

    # layout
    if x.dtype.kind == 'M':
        fig.autofmt_xdate()
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(labels[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.grid = True
    pl.tight_layout()
    return fig, ax


def fillplot(x, y, alpha=0.1, color='k', ax=None, **kwargs):
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    y = np.insert(y.values, 0, False)
    y = np.append(y, False)
    x = x.values
    x = np.insert(x, 0, x[0])
    start = x[y[1:] &  ~y[:-1]]
    end = x[~y[1:] & y[:-1]]
    for istart, iend in zip(start, end):
        ax.axvspan(istart, iend, alpha=alpha, color=color, **kwargs)
    #ax.legend()
    return fig, ax


def bandplot(x, y1, y2, ax=None, color=None, boundaries=False, **kwargs):
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if boundaries:
        ax.plot(x.values, y1.values, '--', color=color, alpha=0.5, **kwargs)
        ax.plot(x.values, y2.values, '--', color=color, alpha=0.5, **kwargs)
    ax.fill_between(x.values, y1.values, y2.values, color=color, alpha=0.2, **kwargs)
    return fig, ax

###############################################################################
def histplot(a, bins=10, range=None, **kwargs):
    if range is None:
        range = [uf.nanmin(a).values, uf.nanmax(a).values]
    x, y, w = analysis.histogram(a, bins=bins, range=range, **kwargs)
    return y.barplot(width=w)


#def hist2dplot(x, y, **kwargs):
#    histkwargs, kwargs = common.splitkwargs(analysis.histogram2d, kwargs)
#    x, y, z, wx, wy = analysis.histogram2d(x, y, **histkwargs)
#    return z.pixplot(**kwargs)


def kdeplot(x, y, **kwargs):
    z = analysis.kde(x, y, **kwargs)
    return z.contplot(**kwargs)


def fourierplot():
    pass


def spectrogram():
    pass


def plotlinregress(x, y, style='-', ax=None, fig=None, pos=(0.7, 0.8), color='C0',
                   xlim=None, prob=0.95, interval=False, **kwargs):
    a0, a1, r, stderr, stderr_res, stderr_a1, cov, n = analysis.linregress2(x, y)

    if xlim is None:
        xlim = (float(uf.nanmin(x).values), float(uf.nanmax(x).values))

    ax.plot([xlim[0], xlim[1]], [a0 + a1 * xlim[0], a0 + a1 * xlim[1]], style,
            color='#262626')

    if interval:
        xx = np.linspace(*xlim)
        stderr_mean = stderr_res * (1 + (xx - uf.nanmean(x))**2 / cov[0, 0])**0.5 / n**0.5
        y_hat = xx * a1 + a0
        crit = stats.t.ppf((1 - prob) / 2, n - 2)
        conf = (y_hat + stderr_mean * crit, y_hat - stderr_mean * crit)
        ax.plot(xx, conf[0], ':', color='#262626')
        ax.plot(xx, conf[1], ':', color='#262626')
    ax.text(*pos, '$r = %.3f$\n$y=%.3f+%.3f x$\n$RSE=%.3f$' % (r, a0, a1, stderr_res),
            transform=ax.transAxes, bbox={'facecolor':'none', 'edgecolor':color})
    if x.attrs['unit'] == y.attrs['unit']:
        ax.plot(xlim, xlim, '--', color='#262626')
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_aspect(1)
    return a0, a1, r, stderr


def plotpowerlaw(x, y, style='-', ax=None, fig=None, pos=(0.7, 0.8), color='C0',
                   xlim=None, prob=0.95, interval=False, **kwargs):
    a0, a1, R2 = analysis.powerlaw(x, y)

    if xlim is None:
        xlim = (float(uf.nanmin(x).values), float(uf.nanmax(x).values))

    xpred = np.linspace(*xlim)
    ypred = a0 * xpred**a1
    ax.plot(xpred, ypred, style, color='#262626')

    ax.text(*pos, '$R^2 = %.3f$\n$y = %.3f x %.3f$' % (R2, a0, a1),
            transform=ax.transAxes, bbox={'facecolor':'none', 'edgecolor':color})
    if x.attrs['unit'] == y.attrs['unit']:
        ax.plot(xlim, xlim, '--', color='#262626')
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_aspect(1)
    return a0, a1, R2


def connect(time, axes):
    xy = (time, 0)
    con = ConnectionPatch(xyA=xy, xyB=(0.5, 1), coordsA='data', coordsB='axes fraction',
                          axesA=axes[0], axesB=axes[1], color='#999999', zorder=100)
    axes[0].axvline(time, color='#999999', linestyle='--')
    axes[1].add_artist(con)


def sublabels(axes, pos='ul', include=None, c='k'):
    if pos == 'ul':
        xpos = 0.01
        ypos = 0.9
    elif pos == 'ur':
        xpos = 0.9
        ypos = 0.9
    elif pos == 'll':
        xpos = 0.01
        ypos = 0.01
    elif pos == 'lr':
        xpos = 0.9
        ypos = 0.01

    if include is not None:
        for n, ax in enumerate(axes):
            addtext = include[n]
            ax.text(xpos, ypos, '('+string.ascii_lowercase[n]+')' + addtext,
                    transform=ax.transAxes, size=16, weight='bold', zorder=100,
                    color=c)
    else:
        for n, ax in enumerate(axes):
            ax.text(xpos, ypos, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes,
                    size=16, weight='bold', zorder=100, color=c)


###############################################################################
def hist2dplot(x, y, log=False, xbins=100, ybins=100, **kwargs):
    xdim = x.dims
    ydim = y.dims
    if xdim != ydim:
        err = 'unequal dimensions: %s and %s' % (str(xdim), str(ydim))
        raise Exception(err)

    labels, kwargs = get_labels(x, y, **kwargs)
    subp_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.subplots)}
    hist_kwargs = {k: v for k, v in kwargs.items()
                   if k in common.get_kwarg_names(pl.hist2d)}

    if x.dtype.kind == 'M':
        x = x.astype(datetime)
    if y.dtype.kind == 'M':
        y = y.astype(datetime)

    if log:
        x = uf.log(x)
        y = uf.log(y)

    fig, ax = pl.subplots(**subp_kwargs)

    H, xedges, yedges, img = ax.hist2d(x.values, y.values, (xbins, ybins), **hist_kwargs)

    # layout
    if log:
        def f(x, pos):
            return '{:0.2e}'.format(uf.exp(x))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(f))
    if x.dtype.kind == 'M':
        fig.autofmt_xdate()
    fig.colorbar(img, ax=ax)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.grid = True
    pl.tight_layout()
    return fig, ax


def lineplot(ax, array, label, style='-', **kwargs):
#    array = array[~uf.isnan(array)]
    line, = ax.plot(array['time'].values, array.values, style, label=label,
                    **kwargs)
    ax.set_ylabel(array.attrs['quantity']+' ('+array.attrs['unit']+')')


def timeseries(array):
    fig, ax = pl.subplots()
    array = array[~uf.isnan(array)]
    line, = ax.plot(array['time'].values, array.values, label=array.name)
    ax.set_ylabel(array.attrs['quantity']+' ('+array.attrs['unit']+')')
    ax.legend()
    ax.set_xlabel('time')
    pl.grid=True
    return fig, ax

def cumplot(ax, x, y, label):
    x = uf.cumsum(x)
    y = uf.cumsum(y)
    mask = (~uf.isnan(x) & ~uf.isnan(y) & ~
            uf.isinf(x) & ~uf.isinf(y))
    line, = ax.plot(x.values[mask.values], y.values[mask.values], label=label)
    ax.set_ylabel(y.attrs['quantity']+' ('+y.attrs['unit']+')')
    ax.set_xlabel(x.attrs['quantity']+' ('+x.attrs['unit']+')')


###############################################################################
def get_labels(*var, **kwargs):
    names = []
    for x in var:
        names.append('$' + x.name.replace(' ', '\ ') + '$')

    units = []
    for x in var:
        if (x.dtype.kind == 'M') or (x.attrs['unit'] == 'NA') or (x.attrs['unit'] == ''):
            units.append('')
        else:
            units.append('[$\mathrm{' + x.attrs['unit'] + '}$]')

    labels = ['x', 'y', 'z']
    if var[0].attrs['quantity'] == var[1].attrs['quantity']:
        labels[0] = r'%s %s' % (names[0], units[0])
        labels[1] = r'%s %s' % (names[1], units[1])
        #kwargs['figsize'] = pl.figaspect(1)
    else:
        labels[0] = r'%s %s' % (var[0].attrs['quantity'], units[0])
        labels[1] = r'%s %s' % (var[1].attrs['quantity'], units[1])

    if len(var) == 3:
        labels[2] = '%s %s' % (names[2], units[2])
    return labels, kwargs
