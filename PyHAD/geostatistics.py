#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:37:42 2017

@author: tcvanleth
"""

from collections import OrderedDict
import importlib
import itertools

import dask.array as da
import numpy as np
from matplotlib.colors import to_hex
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

from phad import analysis, common, geometry, ufuncs


thismodule = importlib.import_module(__name__)


colors  = {'#1f77b4': '#aec7e8',
           '#ff7f0e': '#ffbb78',
           '#2ca02c': '#98df8a',
           '#d62728': '#ff9896',
           '#9467bd': '#c5b0d5',
           '#8c564b': '#c49c94',
           '#e377c2': '#f7b6d2',
           '#7f7f7f': '#c7c7c7',
           '#bcbd22': '#dbdb8d',
           '#17becf': '#9edae5',
           '#262626': '#646464'}


def sciformat(x, fmt="%1.2e"):
    tup = (fmt % x).split('e')
    significand = tup[0].rstrip('.')
    sign = tup[1][0].replace('+', '')
    exponent = tup[1][1:].lstrip('0')
    if exponent:
        exponent = '10^{%s%s}' % (sign, exponent)
    if significand and exponent:
        return  r'%s{\times}%s' % (significand, exponent)
    else:
        return  r'%s%s' % (significand, exponent)


def distance(a, b, direction=None):
    """
    distance between two geolocated stations (zerodimensional)
    """
    x1 = a.attrs['site_longitude']
    y1 = a.attrs['site_latitude']
    try:
        z1 = a.attrs['site_altitude']
    except KeyError:
        z1 = np.nan
    x2 = b.attrs['site_longitude']
    y2 = b.attrs['site_latitude']
    try:
        z2 = b.attrs['site_altitude']
    except KeyError:
        z2 = np.nan
    l = geometry.haversines(x1, y1, z1, x2, y2, z2).compute()
    if direction is None:
        return np.sqrt(l @ l)
    else:
        return np.abs(l @ direction) / np.linalg.norm(direction)


def interm(x, y, axis=None):
    n = np.nansum((x > 0.1) & (y > 0.1) & ~ufuncs.isnan(x) & ~ufuncs.isnan(y), axis=axis)
    o = np.nansum(((x > 0.1) | (y > 0.1)) & ~ufuncs.isnan(x) & ~ufuncs.isnan(y), axis=axis)
    return n/o


def spatialstat(Rs, stat=None, func='exp2', ylabel=None, p0=None, ax=None,
                fig=None, Lmax=None, direction=None, log=False, fmt='o',
                bootstrap=False, num=100, plotdata=True, color=None, plot=True,
                style='-'):
    """
    plot the correlogram for data Rs and fit the function func.
    """
    # make combinations from stations
    if isinstance(Rs, list):
        combis = sum([list(itertools.combinations(tuple(x.values()), 2))
                     for x in Rs], [])
    else:
        combis = list(itertools.combinations(tuple(Rs.values()), 2))

    # determine distances and correlation coefficients
    L = np.zeros(len(combis))
    rpred = np.zeros(len(combis))
    rconf = np.ones(len(combis))
    for i, combi in enumerate(combis):
        L[i] = distance(*combi, direction) / 1000
        if bootstrap:
            r = analysis.bootstrap(*combi, func=stat)[:, 0, 1]
            rpred[i] = r.mean()
            rconf[i] = r.std()
        else:
            if stat == interm:
                rpred[i] = stat(*combi)
            else:
                rpred[i] = stat(*combi, dim='time').values[0, 1]

    #plotting
    if plot:
        if ax is None:
            fig, ax = pl.subplots()
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            color = to_hex(color)
        lcolor = colors[color]
        if plotdata:
            if bootstrap:
                ax.errorbar(L, rpred, yerr=rconf, fmt=fmt, color=lcolor)
            else:
                ax.plot(L, rpred, fmt, color=lcolor, alpha=0.8)
        ax.set_xlabel('$d\ \mathrm{[km]}$')
        ax.set_ylabel(ylabel)
        if log:
            ax.set_xscale('log')

    if func is None:
        return L, rpred, None, None, fig, ax

    # define correlogram regression curve
    bounds = (-np.inf, np.inf)
    if func == 'exp1':
        func = lambda x, d0: np.exp(-(x / d0))
        jac = lambda x, d0: [x / d0**2 * np.exp(-(x / d0))]
        if p0 is None:
            p0 = [('d_0', 3),]
    elif func == 'exp2':
        func = lambda x, d0, s0: np.exp(-(x / d0)**s0)
        def jac(x, d0, s0):
            v = np.exp(-(x/d0)**s0)
            return np.stack((s0*(x/d0)**s0*v/d0, -(x/d0)**s0*np.log(x/d0)*v)).T
        if p0 is None:
            p0 = [('d_0', 20), ('s_0', 0.7)]
    elif func == 'exp3':
        func = lambda x, d0, s0, n0: n0 * np.exp(-(x / d0)**s0)
        def jac(x, d0, s0, n0):
            v = np.exp(-(x/d0)**s0)
            return np.stack(((n0*s0*(x/d0)**s0*v/d0),
                             -n0*(x/d0)**s0*np.log(x/d0)*v, v)).T
        bounds = (np.array([0, 0, 0]), np.array([1e6, 10, 1]))
        if p0 is None:
            p0 = [('d_0', 20), ('s_0', 1), ('n_0', 0)]
    elif func == 'gau2':
        func = lambda x, d0, s0: np.exp(-0.5*(x / d0)**s0)
        def jac(x, d0, s0):
            v = np.exp(-0.5*(x/d0)**s0)
            return np.stack((0.5*s0*(x/d0)**s0*v/d0, -0.5*(x/d0)**s0*np.log(x/d0)*v)).T
        if p0 is None:
            p0 = [('d_0', 10), ('s_0', 1)]
    elif func == 'sph2':
        def func(x, d0, s0):
            return s0 * np.minimum(1 - (1.5 * x / d0 + 0.5 * (x / d0)**3), 1)
        if p0 is None:
            p0 = [('d_0', 3), ('s_0', 1)]
    elif func == 'sph3':
        def func(x, d0, s0, n0):
            return n0 + s0 * np.minimum(1 - (1.5 * x / d0 + 0.5 * (x / d0)**3), 1)
        if p0 is None:
            p0 = [('d_0', 3), ('s_0', 1), ('n_0', 0)]
    p0 = OrderedDict(p0)

    # nonlinear least-squares regression
    cond = ~da.isnan(rpred)
    L = L[cond]
    rpred = rpred[cond]
    rconf = rconf[cond]
    popt, pcov = curve_fit(func, L, rpred, p0=list(p0.values()),
                           jac=jac, bounds=bounds)

    # goodness of fit
    SS_res = np.sum((func(L, *popt) - rpred)**2)
    SS_tot = np.sum((rpred - np.mean(rpred))**2)
    R2 = 1 - SS_res / SS_tot

    if plot is False:
        return L, rpred, popt, pcov, fig, ax

    #evaluate fit function
    if Lmax is None:
        Lmax = L.max()
    if Lmax < L.max():
        x = np.linspace(L.min(), Lmax, num)
        x_be = np.linspace(0, L.min(), num)
        x_af = np.linspace(Lmax, Lmax, 1)
    else:
        x = np.linspace(L.min(), L.max(), num)
        x_be = np.linspace(0, L.min(), num)
        x_af = np.linspace(L.max(), Lmax, num)

    y = func(x, *popt)
    y_be = func(x_be, *popt)
    y_af = func(x_af, *popt)

    # numeric confidence band
    dist = np.random.multivariate_normal(popt, pcov, 1000).T
    ensemble = func(x[None, :], *tuple(dist[..., None]))
    p975 = np.percentile(ensemble, 97.5, axis=0)
    p025 = np.percentile(ensemble, 2.5, axis=0)

    ensemble_be = func(x_be[None, :], *tuple(dist[..., None]))
    p975_be = np.percentile(ensemble_be, 97.5, axis=0)
    p025_be = np.percentile(ensemble_be, 2.5, axis=0)

    ensemble_af = func(x_af[None, :], *tuple(dist[..., None]))
    p975_af = np.percentile(ensemble_af, 97.5, axis=0)
    p025_af = np.percentile(ensemble_af, 2.5, axis=0)

    std = np.sqrt(np.diag(pcov))

    # plot regression curve
    text = '\n'.join('$%s=%s\pm%s$' % (x[0], sciformat(x[1]), sciformat(x[2]))
                     for x in zip(p0.keys(), popt, std))
    text += '\n $R^2=%.2f$' % R2
    ax.plot(x, y, style, color=color, label=text, zorder=10)
    ax.plot(x_be, y_be, '--', color=color, zorder=10)
    ax.plot(x_af, y_af, '--', color=color, zorder=10)
    ax.fill_between(x, p025, p975, color=color, alpha=0.1)
    ax.fill_between(x_be, p025_be, p975_be, color=color, alpha=0.1)
    ax.fill_between(x_af, p025_af, p975_af, color=color, alpha=0.1)
    ax.legend()
    return L, rpred, popt, pcov, fig, ax


def correlogram(Rs, func='exp2', **kwargs):
    ylabel = r'$\rho\ \mathrm{[-]}$'
    return spatialstat(Rs, stat=ufuncs.corr, ylabel=ylabel, func=func, **kwargs)


def semivariogram(Rs, func='sph2', **kwargs):
    ylabel = r'$SEMIVAR$'
    return spatialstat(Rs, stat=analysis.semivar, ylabel=ylabel, func=func, **kwargs)


def intermittency(Rs, func='exp2', **kwargs):
    ylabel = r'$I \mathrm{[-]}$'
    return spatialstat(Rs, stat=interm, ylabel=ylabel, func=func, **kwargs)


def time_space(data, intervals, ax=None, fig=None, fig2=None, ax2=None,
               moving=False, statistic='correlogram', wetonly=False, how='mean',
               xmin=None, xmax=None, d0func='pow', s0func='pow', color=None,
               **kwargs):
    """
    plot correlograms of the data for all intervals and plot the relation
    between temporal aggregation and decorrelation distance.
    """
    sfunc = getattr(thismodule, statistic)

    #determine decorrelation distances
    d0 = np.full(len(intervals), np.nan)
    s0 = np.full(len(intervals), np.nan)
    d0_sigma = np.full(len(intervals), np.nan)
    s0_sigma = np.full(len(intervals), np.nan)
    dt = np.zeros(len(intervals))
    L = [None] * len(intervals)
    r2 = [None] * len(intervals)
    if ax2 is None:
        ax2 = [None]*len(intervals)
        fig2 = [None]*len(intervals)

    for i, interval in enumerate(intervals):
        dt[i] = common.str_to_sec(interval)
        p0 = [('d_0', 0.5*dt[i]**0.55), ('s_0', 0.44*dt[i]**0.07)]
        if isinstance(data, list):
            dat = []
            for idat in data:
                try:
                    if moving:
                        dat.append(ufuncs.rollmean(idat, window={'time':interval}))
                    else:
                        dat.append(idat.resample(interval, how=how))
                except ValueError:
                    continue
            if wetonly:
                dat = [idat[idat != 0] for idat in dat]
        else:
            try:
                if moving:
                    dat = ufuncs.rollmean(data, window={'time':interval})
                else:
                    dat = data.resample(interval, how=how)
            except ValueError:
                d0[i] = np.nan
                s0[i] = np.nan
                continue
            if wetonly:
                dat = dat[dat != 0]
        if isinstance(fig2, list):
            fig2i = fig2[i]
        else:
            fig2i = fig2
        if ax2[i] is None:
            L[i], r2[i], popt, pcov, fig2i, ax2[i] = sfunc(dat, p0=p0, plot=False, **kwargs)
        else:
            L[i], r2[i], popt, pcov,  _, _ = sfunc(dat, p0=p0, plot=False,
                                                   ax=ax2[i], fig=fig2i, **kwargs)
        d0[i] = popt[0]
        s0[i] = popt[1]
        d0_sigma[i] = np.sqrt(np.diag(pcov))[0]
        s0_sigma[i] = np.sqrt(np.diag(pcov))[1]

    p0_2 = [('a', 0.44), ('b', 0.07)]
    fig, ax[0] = fitplot(dt, d0, d0_sigma, '$d_0\ \mathrm{[km]}$', ax=ax[0],
           xmax=xmax, xmin=xmin, func=d0func, color=color)
    fig, ax[1] = fitplot(dt, s0, s0_sigma, '$s_0\ \mathrm{[-]}$', p0=p0_2,
           scale='linlog', ax=ax[1], xmax=xmax, xmin=xmin, func=s0func,
           color=color)
    return L, r2, fig, ax, fig2, ax2


def fitplot(x_obs, y_obs, sigma, ylabel, func='pow', p0=None, ax=None, fig=None,
            scale='loglog', num=100, xmin=None, xmax=None, color=None):
    #plotting
    if ax is None:
        fig, ax = pl.subplots()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    lcolor = colors[color]
    ax.errorbar(x_obs, y_obs, yerr=sigma, fmt='o', color=lcolor)
    ax.set_xlabel('$\Delta t\ \mathrm{[s]}$')
    ax.set_ylabel(ylabel)
    if scale == 'loglog':
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif scale == 'linlog':
        ax.set_xscale('log')
    elif scale == 'linlin':
        pass

    if func is None:
        return fig, ax

    # define regression curve
    if func == 'pow':
        func = lambda x, a, b: a*x**b
        if p0 is None:
            p0 = [('a', 500), ('b', 0.55)]
    elif func == 'para':
        func = lambda x, a, b: a*x**2 + b
        if p0 is None:
            p0 = [('c', 1e-3), ('d', 3e4)]
    p0 = OrderedDict(p0)

    # non-linear least-squares regression
    cond = ~np.isnan(y_obs)
    x_obs = x_obs[cond]
    y_obs = y_obs[cond]
    sigma = sigma[cond]
    popt, pcov = curve_fit(func, x_obs, y_obs, p0=list(p0.values()), sigma=sigma)

    # goodness of fit
    SS_res = np.sum((func(x_obs, *popt) - y_obs)**2)
    SS_tot = np.sum((y_obs - np.mean(y_obs))**2)
    R2 = 1 - SS_res / SS_tot

    #evaluate fit function
    if xmax is None:
        xmax = np.nanmax(x_obs)
    if xmin is None:
        xmin = np.nanmin(x_obs)
    if scale == 'loglog' or scale == 'linlog':
        x_pred = np.logspace(np.log10(np.nanmin(x_obs)), np.log10(xmax), num)
        x_be = np.logspace(np.log10(xmin), np.log10(np.nanmin(x_obs)), num)
        x_af = np.logspace(np.log10(np.nanmax(x_obs)), np.log10(xmax), num)
    else:
        x_pred = np.linspace(np.nanmin(x_obs), np.nanmax(x_obs), num)
        x_be = np.linspace(0, np.nanmin(x_obs), num)
        x_af = np.linspace(np.nanmax(x_obs), xmax, num)
    y_pred = func(x_pred, *popt)
    y_be = func(x_be, *popt)
    y_af = func(x_af, *popt)

    # numeric confidence band
    dist = np.random.multivariate_normal(popt, pcov, 1000).T
    ensemble = func(x_pred[None, :], *tuple(dist[..., None]))
    p975 = np.percentile(ensemble, 97.5, axis=0)
    p025 = np.percentile(ensemble, 2.5, axis=0)

    ensemble_be = func(x_be[None, :], *tuple(dist[..., None]))
    p975_be = np.percentile(ensemble_be, 97.5, axis=0)
    p025_be = np.percentile(ensemble_be, 2.5, axis=0)

    ensemble_af = func(x_af[None, :], *tuple(dist[..., None]))
    p975_af = np.percentile(ensemble_af, 97.5, axis=0)
    p025_af = np.percentile(ensemble_af, 2.5, axis=0)

    std = np.sqrt(np.diag(pcov))

    #plot regression curve
    text = '\n'.join('$%s=%s\pm%s$' % (x[0], sciformat(x[1]), sciformat(x[2]))
                     for x in zip(p0.keys(), popt, std))
    text += '\n $R^2=%.2f$' % R2
    ax.plot(x_pred, y_pred, '-', color=color, label=text, zorder=10)
    ax.plot(x_be, y_be, '--', color=color, zorder=10)
    ax.plot(x_af, y_af, '--', color=color, zorder=10)
    ax.fill_between(x_pred, p025, p975, color=color, alpha=0.1)
    ax.fill_between(x_be, p025_be, p975_be, color=color, alpha=0.1)
    ax.fill_between(x_af, p025_af, p975_af, color=color, alpha=0.1)
    ax.legend()
    return fig, ax


def seasonality(data, intervals, statistic='correlogram', **kwargs):
    data = data.groupby('time.day')

    sfunc = getattr(thismodule, statistic)

    #determine decorrelation distances
    d0 = np.full(len(intervals), np.nan)
    sigma = np.full(len(intervals), np.nan)
    dt = np.zeros(len(intervals))
    L = [None] * len(intervals)
    r2 = [None] * len(intervals)
    for i, interval in enumerate(intervals):
        dt[i] = common.str_to_sec(interval)
        if isinstance(data, list):
            dat = []
            for idat in data:
                try:
                    dat.append(idat.resample(interval))
                except ValueError:
                    continue
            stations = sum([list(itertools.combinations(tuple(idat.values()), 2))
                        for idat in dat], [])
        else:
            try:
                dat = data.resample(interval)
            except ValueError:
                d0[i] = np.nan
                continue
            stations = list(itertools.combinations(tuple(dat.values()), 2))
        L[i], r2[i], popt, pcov, fig2i, ax2 = sfunc(stations, **kwargs)
        d0[i] = popt[0]
        sigma[i] = np.sqrt(np.diag(pcov))[0]