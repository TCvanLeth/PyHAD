# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:30:07 2015

@author: T.C. van Leth
"""

import dask.array as da

from phad import constants as cst
from phad import ufuncs as uf


@uf.dafunc
def haversines(x1, x2, y1, y2, z1=None, z2=None):

    x1, x2 = da.deg2rad(x1), da.deg2rad(x2)
    y1, y2 = da.deg2rad(y1), da.deg2rad(y2)

    x = (x2 - x1) * da.cos((y1 + y2) * 0.5) * cst.r_earth
    y = (y2 - y1) * cst.r_earth * da.ones_like(x1) * da.ones_like(x2)

    if z1 is None or z2 is None:
        return da.stack((x, y), axis=-1)
    else:
        z1 = da.where(da.isnan(z1), 0, z1)
        z2 = da.where(da.isnan(z2), 0, z2)
        z = (z2 - z1) * da.ones_like(x)
        return da.stack((x, y, z), axis=-1)


def length(x1, x2, y1, y2, z1=None, z2=None):
    path = haversines(x1, x2, y1, y2, z1, z2)
    return uf.sqrt((path**2).sum(dim='dim0'))
