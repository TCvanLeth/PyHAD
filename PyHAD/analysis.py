#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:26:13 2017

@author: tcvanleth
"""
import dask.array as da
import numpy as np
from scipy import stats, optimize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import phad.ufuncs as uf
from phad.variable import Index


def semivar(*args, **kwargs):
    """
    semivariance
    """
    args = uf.broadcast(*args)
    X = da.stack([x.task.flatten() for x in args])
    out = 0.5 * da.nanmean((X[0] - X[1])**2)
    return out


def bootstrap(*args, func='corr', size=100):
    if func == 'corr':
        func = uf.corr

    args = uf.broadcast(*args)
    args = tuple(x.task if hasattr(x, 'task') else x for x in args)
    args = tuple(x.compute() if hasattr(x, 'compute') else x for x in args)

    indices = np.random.randint(len(args[0]), size=(size, len(args[0])))
    return func(*tuple(x[indices] for x in args), axis=1)


def linregress(x, y, **kwargs):
    """
    linear least-squares regression
    """
    x, y = uf.broadcast(x, y)
    x = x.task.flatten().compute()
    y = y.task.flatten().compute()
    cond = ~(np.isnan(x) | np.isnan(y))
    x = x[cond]
    y = y[cond]
    return stats.linregress(x, y)


def linregress2(x, y, prob=0.95, **kwargs):
    """
    linear least-squares regression
    """
    c = uf.cov(x, y).task
    r = uf.corr(x, y).task[0, 1]
    a1 = (c[0, 1] / c[0, 0])

    cond = uf.isnan(x) | uf.isnan(y)
    x = uf.where(cond, np.nan, x)
    y = uf.where(cond, np.nan, y)

    n = uf.nansum(~uf.isnan(x)).task
    a0 = uf.nanmean(y) - uf.nanmean(x) * a1
    stderr = uf.sqrt((1 - r**2) * c[1, 1] / c[0, 0] / (n - 2))
    stderr_res = uf.nanstd(y - a0 - x * a1) * ((n - 1) / (n - 2))**0.5
    stderr_a1 = stderr_res / uf.nanstd(x) * n**-0.5
    return da.compute(a0.task, a1, r, stderr, stderr_res.task, stderr_a1.task, c, n)


#@uf.hierarch
def powerlaw(x, y):
    """
    non-linear powerlaw fit
    """
    x = x.task.flatten()
    y = y.task.flatten()
    cond = (x > 0) & (y > 0)
    x = x[cond].compute()
    y = y[cond].compute()

    def fitfunc(p, x): return p[0] * x**p[1]

    def costfunc(p, x, y):
        return fitfunc(p, x) - y

    def Dfun(p, x, y):
        v = x**p[1]
        return [v, p[0] * v * np.log(x)]

    par, cov, infodict, mesg, ier = optimize.leastsq(costfunc, [1., 1.],
                                                     args=(x, y), Dfun=Dfun,
                                                     col_deriv=True,
                                                     full_output=True)
    ss_err = (infodict['fvec']**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    R2 = 1 - (ss_err/ss_tot)
    return np.append(par, [R2])


@uf.hierarch
def powerlaw2(x, y):
    """
    linear fit on logarithms
    """
    cond = (x > 0) & (y > 0)
    x = uf.log10(x[cond]).compute()
    y = uf.log10(y[cond]).compute()
    par = np.asarray(linregress(x, y)[:2])
    par[0] = np.exp(par[0])
    return par


def kde(x, y, bandwidth=1.0, log=False, num=200, **kwargs):
    """
    kernel density estimation
    """
    typ = type(x)
    xdim = x.dims
    ydim = y.dims
    if xdim != ydim:
        err = 'unequal dimensions: %s and %s' % (str(xdim), str(ydim))
        raise Exception(err)

    # prepare training data
    if log is True:
        x, y = uf.log(x), uf.log(y)
    x_m, y_m = uf.nanmean(x), uf.nanmean(y)
    x_s, y_s = uf.nanstd(x), uf.nanstd(y)
    x = (x - x_m) / x_s
    y = (y - y_m) / y_s

    # estimate  bandwidth with cross validation
    mask = (~uf.isnan(y) & ~uf.isnan(x) & ~uf.isinf(y) & ~uf.isinf(x))
    train = np.vstack((x.values[mask.values], y.values[mask.values])).T
    if bandwidth is None:
        gcv = GridSearchCV(KernelDensity(),
                           {'bandwidth': np.linspace(0.05, 1.0)}, cv=5)
        gcv.fit(train)
        bandwidth = gcv.best_params_['bandwidth']
    kde = KernelDensity(bandwidth=bandwidth).fit(train)

    # prepare sample grid
    x_g = np.linspace(uf.nanmin(x).values, uf.nanmax(x).values, num=num)
    y_g = np.linspace(uf.nanmin(y).values, uf.nanmax(y).values, num=num)

    x_grid, y_grid = np.meshgrid(x_g, y_g)
    grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    z = np.exp(kde.score_samples(grid)).reshape(x_grid.shape).T

    x = Index(x_g * x_s + x_m, name=x.name, attrs=x.attrs)
    y = Index(y_g * y_s + y_m, name=y.name, attrs=y.attrs)
    return typ(z, coords=[x, y], name='density',
                   attrs={'bandwidth': bandwidth})


def histogram(a, **kwargs):
    y, bins = da.histogram(a.task, **kwargs)
    w = bins[1:] - bins[:-1]
    x = (bins[1:] + bins[:-1]) * 0.5

    x = Index(x, a.name, a.attrs)
    y = type(a)(y, coords=[x], name='number')
    w = type(a)(w, coords=[x], name='binwidth', attrs=a.attrs)
    return x, y, w


def histogram2d(a, b, **kwargs):
    aa = a.values
    bb = b.values
    cond = ~np.isnan(a) & ~np.isnan(b)
    aa = aa[cond]
    bb = bb[cond]

    z, xbins, ybins = np.histogram2d(aa, bb, **kwargs)
    wx = xbins[1:] - xbins[:-1]
    x = (xbins[1:] + xbins[:-1]) * 0.5
    wy = ybins[1:] - ybins[:-1]
    y = (ybins[1:] + ybins[:-1]) * 0.5

    x = Index(x, a.name, a.attrs)
    y = Index(y, b.name, b.attrs)
    z = type(a)(z, coords=[x, y], name='number')
    wx = type(a)(wx, coords=[x], name='binwidth')
    wy = type(b)(wy, coords=[y], name='binwidth')
    return x, y, z, wx, wy


###############################################################################
# main functions
def wavelet(data):
    pass


def pca():
    pass


def subanalysis():
    pass
