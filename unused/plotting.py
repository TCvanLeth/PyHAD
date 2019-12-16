# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:03:13 2015

@author: T.C. van Leth

This module provides some convenient shortcut functions for common plots
"""
import datetime as dt
import os

import numpy as np

import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib import cm

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from scipy.interpolate import InterpolatedUnivariateSpline

import harray as ha
from harray import ufuncs as uf
from mwlink import settings


def timeseries(data, band=None):
    fig, ax = pl.subplots()
    for ID, idat in data.iteritems():
        cond = ~np.isnan(idat)
        line, = ax.plot(idat['time'][cond], idat[cond], label=ID)
        ax.set_ylabel(idat.attrs['quantity']+' ('+idat.attrs['units']+')')

    if band:
        for ID, iband in band.iteritems():
            cond = ~np.isnan(iband)
            fill_between

    ax.legend()
    ax.set_xlabel('time')
    pl.grid=True
    return fig

def timeseries2(data, output=None, var=None):
    """
    produce timeseries from data.
    """
    if not var:
        # unpack data
        if not isinstance(data, dict):
            data = {'': data}
        for idat in data.keys():
            subdat = data[idat]
            if isinstance(subdat, ha.Array):
                subdat = ha.Channel(variables={subdat.name: subdat})
            elif isinstance(subdat, list):
                dic = {}
                for i in range(len(subdat)):
                    if subdat[i].name in dic.keys():
                        iname = subdat[i].name+'_'+str(i)
                    else:
                        iname = subdat[i].name
                    dic.update({iname: subdat[i]})
                subdat = ha.Channel(variables=dic)
            elif isinstance(subdat, ha.Channel):
                pass
            else:
                raise Exception

            # prepare figure
            fig, ax = pl.subplots()
            for jdat in subdat.data_vars.keys():

                # plot
                cond = ~uf.isnan(subdat[jdat])
                line, = ax.plot(subdat[jdat]['time'][cond], subdat[jdat][cond],
                                label=jdat)

            # layout
            ax.legend()
            ax.set_title(idat)
            ax.set_xlabel('time')
            ax.set_ylabel(subdat.data_vars.values()[0].attrs['unit'])
            pl.grid = True
    else:
        # prepare figure
        fig, ax = pl.subplots()
        for ilink in data.iterkeys():
            for ichan in data[ilink].iterkeys():
                x = data[ilink][ichan][var]

                # plot
                cond = ~uf.isnan(x)
                line, = ax.plot(x['time'][cond], x[cond],
                                label=ilink+'/'+ichan)

        # layout
        unit = data.values()[0].values()[0][var].attrs['unit']
        ax.legend()
        ax.set_title(var)
        ax.set_xlabel('time')
        ax.set_ylabel(var+' ('+unit+')')
        pl.grid = True
    return fig


def k_R(dsddat):

    # prepare data
    h = dsddat['htype']
    k = dsddat['k_38_H'][h == 'liquid']
    R = dsddat['R'][h == 'liquid']
    mask = (~uf.isnan(uf.log(k)) & ~uf.isnan(uf.log(R)) &
            ~uf.isinf(uf.log(k)) & ~uf.isinf(uf.log(R)))

    # make plot
    fig, ax = pl.subplots()
    H, xedges, yedges, img = pl.hist2d(uf.log(R)[mask],
                                       uf.log(k)[mask], 1000, cmap=cm.hot)

    # layout
    ax.set_ylabel('attenuation (dB)')
    ax.set_xlabel('rain intensity (mm/hr)')
    ax.set_title('parsivel derived k_R relation')
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:0.2e}'.format(uf.exp(x))))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{:0.2e}'.format(uf.exp(y))))
    fig.colorbar(img, ax=ax)
    pl.tight_layout()
    pl.grid = True


# kde_plots
def kde(x, y, bandwidth=None, log=False):
    # unpack data
    xname = x.name
    yname = y.name
    # x_unit = x.attrs['unit']
    # y_unit = y.attrs['unit']

    # prepare training data
    if log is True:
        x, y = uf.log(x), uf.log(y)
    mask = (~uf.isnan(y) & ~uf.isnan(x) & ~uf.isinf(y) & ~uf.isinf(x))
    x, y = x[mask], y[mask]
    x_m, y_m = uf.mean(x), uf.mean(y)
    x_s, y_s = uf.std(x), uf.std(y)
    x, y = (x-x_m)/x_s, (y-y_m)/y_s
    train = uf.vstack((x[mask], y[mask])).T

    # prepare sample grid
    x_g = np.linspace(uf.min(x), uf.max(x), num=200)
    y_g = np.linspace(uf.min(y), uf.max(y), num=200)
    x_grid, y_grid = np.meshgrid(x_g, y_g)
    grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    # estimate  bandwidth with cross validation
    if bandwidth is None:
        gcv = GridSearchCV(KernelDensity(),
                           {'bandwidth': np.linspace(0.05, 1.0)}, cv=5)
        gcv.fit(train)
        bandwidth = gcv.best_params_['bandwidth']
        print(gcv.best_params_)

    # compute kde
    kde = KernelDensity(bandwidth=bandwidth).fit(train)
    Z = uf.exp(kde.score_samples(grid)).reshape(x_grid.shape)
    x_grid, y_grid = x_grid*x_s+x_m, y_grid*y_s+y_m

    # pack data
    xdim = ha.Index(xname, x_g * x_s + x_m)
    ydim = ha.Index(yname, y_g * y_s + y_m)

    return ha.Array(Z, coords=[xdim, ydim], name='density',
                    attrs={'bandwidth': bandwidth})


# fourier-series
def freqplot(data, log=False, var=None):
    if not var:
        # unpack data
        if not isinstance(data, dict):
            data = {'': data}
        for idat in data.keys():
            fig, ax = pl.subplots()
            subdat = data[idat]
            if isinstance(subdat, ha.Array):
                subdat = subdat.to_dataset(name=subdat.name)
            elif isinstance(subdat, list):
                dic = {}
                for i in range(len(subdat)):
                    if subdat[i].name in dic.keys():
                        iname = subdat[i].name+'_'+str(i)
                    else:
                        iname = subdat[i].name
                    dic.update({iname: subdat[i]})
                subdat = ha.Channel(variables=dic)
            elif isinstance(subdat, ha.Channel):
                pass
            else:
                raise Exception

            for jdat in subdat.keys():
                # fill gaps with spline interpolation
                series = subdat[jdat].dropna('time')
                time = series['time'].values.astype('d')
                spline = InterpolatedUnivariateSpline(time, series.values, k=3)
                series = spline(time)

                # normalize data
                std = uf.std(series)
                mean = uf.mean(series)
                series = (series-mean)/std

                # fast fourier transform
                n = len(series)
                freq = np.fft.rfftfreq(n, d=30)
                wind = np.hanning(n)
                spec = uf.abs(np.fft.rfft(series*wind))**2
                dens = spec/uf.sum(spec)
                if log is True:
                    dens = uf.log(dens)

                # make plot
                ax.plot(freq, dens, label=jdat)

            # layout
            ax.set_title(idat)
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('power density')
            ax.legend()
    else:
        # prepare figure
        fig, ax = pl.subplots()
        for ilink in data.iterkeys():
            for ichan in data[ilink].iterkeys():
                x = data[ilink][ichan][var]

                # fill gaps with spline interpolation
                series = x.dropna('time')
                time = series['time'].values.astype('d')
                if len(time) > 3:
                    spline = InterpolatedUnivariateSpline(time, series.values)
                    series = spline(time)

                    # normalize data
                    std = uf.std(series)
                    mean = uf.mean(series)
                    series = (series-mean)/std

                    # fast fourier transform
                    n = len(series)
                    freq = np.fft.rfftfreq(n, d=30)
                    wind = np.hanning(n)
                    spec = uf.abs(np.fft.rfft(series*wind))**2
                    dens = spec/uf.sum(spec)
                    if log is True:
                        dens = uf.log(dens)

                    # make plot
                    ax.plot(freq, dens, label=ilink+'/'+ichan)

        # layout
        ax.set_title(var+' fourier transform')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('spectral density')
        ax.legend()


def spectra(data):
    """
    show spectrogram of attenuation data
    """

    m = len(data.columns)/2
    for iname in data.keys():
        fig, ax = pl.subplots()
        i = 0
        pxx, freq, time, cax = ax.specgram(data[iname], Fs=1/30.)

        # layout
        ax.set_title(iname)
        fig.colorbar(cax, ax=ax)
        i += 1
        pl.tight_layout()
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('time')
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:0.2f}'.format(np.exp(x))))

# error maps

# velocity-diameter plot

# histograms/kde of bulk variables

# cumulative plot

# cross-correlation plot
def cross(x, y):
    # prepare data
    mask = (~uf.isnan(y) & ~uf.isnan(x) & ~uf.isinf(y) & ~uf.isinf(x))
    x, y = x[mask], y[mask]
    x_m, y_m = uf.mean(x), uf.mean(y)
    x_s, y_s = uf.std(x), uf.std(y)
    x, y = (x-x_m)/x_s, (y-y_m)/y_s

    # calculate cross correlation
    cross = np.correlate(x, y, 'same')/len(x)

    # prepare time axis
    lag = np.arange(uf.floor(-len(x)*0.5+1), uf.floor(len(x)*0.5)+1)*30

    # plot
    fig, ax = pl.subplots()
    ax.plot(lag, cross)

    # layout
    pl.tight_layout
    ax.set_title('cross correlation')
    ax.set_xlabel('time')
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(dt.timedelta(seconds=x))))

    # print lag/lead
    opt_lag = dt.timedelta(seconds=lag[uf.argmax(cross)])
    print('the lag of x wrt y is: ', opt_lag)


def save_plot(fig, outname, link_ID, pro_ID):
    subdir = os.path.join(settings.plotpath, link_ID, pro_ID)
    path = os.path.join(subdir, outname+'.png')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    fig.savefig(path)
