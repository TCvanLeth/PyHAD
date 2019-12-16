# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:37:05 2016

@author: tcvanleth
"""

from collections import OrderedDict

import dask.array as da

from phad import common as com
from phad.variable import Variable


class Variables(OrderedDict):
    def __repr__(self, **kwargs):
        values = self.get_printvals()
        return self.pformat(*da.compute(values), **kwargs)

    def get_printvals(self):
        v_short = {}
        for k, v in self.items():
            v_short[k] = v.get_printvals()
        return v_short

    def pformat(self, values, title=None, col_width=None):
        if not title:
            title = 'variables'
        summary = ['%s:' % title]
        if self:
            if not col_width:
                max_name_length = max(len(str(vID)) for vID in self)
                col_width = max(max_name_length, 7) + 6
            for k, v in self.items():
                summary += [v.pformat(values[k], col_width)]
        else:
            summary += ['    *empty*']
        return ''.join(summary)

    @ property
    def dims(self):
        return tuple({i for ivar in self.values() for i in ivar.dims})

    def merge(self, other):
        newvars = self.copy()
        for key, value in other.items():
            if key in newvars:
                if newvars[key].identical(value):
                    continue
                raise KeyError('duplicate key: %s' % key)
            newvars[key] = value
        return newvars

    def setname(self, name, **kwargs):
        for key in tuple(self.keys()):
            ivar = self.pop(key)
            ivar.setname(name, **kwargs)
            if key is None:
                self[None] = ivar
            else:
                self[ivar.name] = ivar

    def rename(self, name_dict):
        for oldname, newname in name_dict.items():
            oldvar = self.pop(oldname, None)
            if oldvar is not None:
                oldvar.name = newname
                self[newname] = oldvar

    def replace(self, var_dict):
        for oldname, newvar in var_dict.items():
            oldvar = self.pop(oldname, None)
            if oldvar is not None:
                if hasattr(newvar, 'name'):
                    self[newvar.name] = newvar
                else:
                    oldvar.name = newvar
                    self[newvar] = oldvar

    def swap_dims(self, name_dict):
        outvars = type(self)()
        for vID, ivar in self.items():
            outvars[vID] = ivar.swap_dims(name_dict)
        return outvars

    def setattrs(self, **kwargs):
        for ivar in self.values():
            ivar.setattrs(**kwargs)

    def delattrs(self, *args):
        for ivar in self.values():
            ivar.delattrs(*args)

    def getattrs(self, key):
        params = type(self)()
        for vID, ivar in self.items():
            try:
                params[vID] = ivar.getattrs(key)
            except KeyError:
                pass # <-- no dict entry will be created when attribute doesn't exist
        return params

    def select(self, **kwargs):
        data = self
        for key, values in kwargs.items():
            select = type(self)()
            if not values:
                continue
            for vID, ivar in data.items():
                if ivar.meta(key, values):
                    select[vID] = ivar
            data = select
        return data

    def idrop(self, *keys, **indexers):
        newvars = self.copy()
        for vID, ivar in newvars.items():
            if vID in keys:
                del self[vID]
            else:
                newvars[vID] = ivar.idrop(**indexers)
        return newvars

    def isel(self, **indexers):
        newvars = type(self)()
        for vID, ivar in self.items():
            newvar = ivar.isel(**indexers)
            if not any(x == 0 for x in newvar.shape) or vID is None:
                newvars[vID] = newvar
        return newvars

    def apply_index(self, locs):
        outvars = type(self)()
        for vID, ivar in self.items():
            outvars[vID] = ivar.apply_index(locs)
        return outvars

    def mask(self, mask, *args, **kwargs):
        return self._do('mask', mask, *args, **kwargs)

    def rechunk(self, chunks):
        return self._do('rechunk', chunks)

    def astype(self, dtype):
        new = self._do('astype', dtype)
        return new

    def conform(self):
        return self._do('conform')

    def _do(self, method, *args, **kwargs):
        newvars = type(self)()
        for vID, ivar in self.items():
            newvar = getattr(ivar, method)(*args, **kwargs)
            if vID is None:
                newvars[None] = newvar
            else:
                newvars[newvar.name] = newvar
        return newvars

    def apply(self, func, *args, **kwargs):
        outvars = type(self)()
        for vID, ivar in self.items():
            outvar = func(ivar, *args, **kwargs)
            if vID is None:
                outvars[None] = outvar
            else:
                outvars[outvar.name] = outvar
        return self._wrap_output(outvars)

    @staticmethod
    def _make_op(method):
        def func(self, *args, **kwargs):
            return self._invoke_method(method, (self, *args), **kwargs)
        return func

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        return self._invoke_method(method, inputs, obj=ufunc, **kwargs)

    def _invoke_method(self, method, inputs, obj=None, **kwargs):
        outvars = com.defaultdict(type(self))
        for vID in self.keys():
            args = self._match(inputs, vID)
            if obj is None:
                obj2, args = args[0], args[1:]
            else:
                obj2 = obj

            outvar = getattr(obj2, method)(*args, **kwargs)
            if not isinstance(outvar, tuple):
                outvar = (outvar,)
            for i, x in enumerate(outvar):
                if vID is None:
                    outvars[i][None] = x
                else:
                    outvars[i][x.name] = x
        if len(outvars) == 1:
            return outvars[0]
        return tuple(outvars.values())

    def _match(self, inputs, vID):
        invars = []
        for idat in inputs:
            if hasattr(idat, 'keys') and len(idat) == 1 and None in idat:
                invars += [idat[None]]
            elif hasattr(idat, 'keys'):
                if vID in idat:
                    invars += [idat[vID]]
            else:
                invars += [idat]
        return tuple(invars)

    def _wrap_output(self, newvars):
        if newvars and isinstance(next(iter(newvars.values())), tuple):
            outvars = com.defaultdict(type(self))
            for vID, ivar in newvars.items():
                for i, x in enumerate(ivar):
                    outvars[i][vID] = x
            return tuple(outvars.values())
        else:
            return newvars

    def copy(self, deep=False):
        newvars = type(self)()
        if deep:
            for vID, ivar in self.items():
                newvars[vID] = ivar.copy(deep=True)
        else:
            for vID, ivar in self.items():
                newvars[vID] = ivar.copy()
        return newvars

    def transpose(self, *dims):
        newvars = type(self)()
        for vID, ivar in self.items():
            newvars[vID] = ivar.transpose(*dims)
        return newvars

    def equals(self, other):
        for k in self:
            if k not in other or not self[k].equals(other[k]):
                return False
        for k in other:
            if k not in self:
                return False
        return True

    def identical(self, other):
        for k in self:
            if k not in other or not self[k].identical(other[k]):
                return False
        for k in other:
            if k not in self:
                return False
        return True


    def extend(self, index, how=None, dim=None):
        newvars = type(self)()
        for vID, ivar in self.items():
            if dim is None or dim in ivar.dims:
                newvars[vID] = ivar.extend(index, how=how, dim=dim)
            else:
                newvars[vID] = ivar
        return newvars

    def ortho(self, dim1, dim2, rename=False, aligned=0):
        iID = dim1.name
        iID2 = dim2.name
        nstep = len(dim1)
        mstep = len(dim2)

        newvars = type(self)()
        for vID, ivar in self.items():
            if iID in ivar.dims:
                newvars[vID] = ivar.ortho(nstep, mstep, iID, iID2, rename, aligned)
            else:
                newvars[vID] = ivar
        return newvars

    def inline(self, index, iID2):
        iID = index.name
        nstep = len(index)

        newvars = type(self)()
        for vID, ivar in self.items():
            if iID in ivar.dims and iID2 in ivar.dims:
                newvars[vID] = ivar.inline(nstep, iID, iID2)
            else:
                newvars[vID] = ivar
        return newvars

    def split(self, dim, n):
        varslist = [type(self)() for i in range(n)]
        for vID, ivar in self.items():
            ilist = ivar.split(dim, n)
            for i, item in enumerate(ilist):
                varslist[i][vID] = item
        return varslist

    def _create_store(self, outh, **kwargs):
        sources = []
        targets = []
        for jname, jvar in self.items():
            if jvar.name in outh:
                continue
            source, target = jvar._create_store(outh, **kwargs)
            sources.append(source)
            targets.append(target)
        return sources, targets

    @classmethod
    def merged(cls, datlist, slicers, length, xdim=None, **kwargs):
        datlist = com.prep_merger(datlist)

        dd = com.defaultdict(OrderedDict)
        for i, idat in datlist.items():
            if hasattr(idat, 'variables'):
                for ivar in idat.variables.values():
                    vID = ivar.name
                    dd[vID][i] = ivar

        variables = cls()
        for vID, ivar in dd.items():
            subcls = type(next(iter(ivar.values())))
            variables[vID] = subcls.merged(ivar, slicers, length, xdim=xdim,
                                           **kwargs)
        return variables


class Coords(Variables):
    @classmethod
    def merged(cls, datlist, slicers, length, xdim=None, **kwargs):
        datlist = com.prep_merger(datlist)

        dd = com.defaultdict(OrderedDict)
        for i, idat in datlist.items():
            if hasattr(idat, 'coords'):
                for cID, icoo in idat.coords.items():
                    dd[cID][i] = icoo

        coords = cls()
        for cID, icoo in dd.items():
            subcls = type(next(iter(icoo.values())))
            coords[cID] = subcls.merged(icoo, slicers, length, **kwargs)

        if xdim is not None:
            for vID in xdim.variables:
                coords[vID] = xdim.variables[vID]
        return coords


class Indexes(Variables):
    @property
    def step(self):
        return {dim: index.step for dim, index in self.items()}

    def get_indexer(self, labels, **kwargs):
        locs = {}
        for dID, label in labels.items():
            if dID in self:
                locs[dID] = self[dID].get_indexer(label, **kwargs)
        return locs

    def join(self, other):
        joined = type(self)()
        for iID, iindex in self.items():
            if iID in other and not iindex.equals(other[iID]):
                joined[iID] = iindex or other[iID]
        return joined

    def __getitem__(self, key):
        if '.' in key:
            split_key = key.split('.', 1)
            if len(split_key) != 2:
                raise KeyError(key)
            ref_name, var_name = split_key
            ref_var = self[ref_name]
            data = getattr(ref_var, var_name)
            return Variable(ref_var.dims, data)
        else:
            return super(Indexes, self).__getitem__(key)

    @classmethod
    def merged(cls, datlist, xdim=None, **kwargs):
        """
        take the union of all dimensions of all datasets in datlist
        """
        datlist = com.prep_merger(datlist)
        dd = com.defaultdict(OrderedDict)
        for i, idat in datlist.items():
            if hasattr(idat, 'indexes'):
                for iindex in idat.indexes.values():
                    dd[iindex.name][i] = iindex

        indexes = cls()
        slicers = {}
        shape = {}
        for iID, iindex in dd.items():
            subcls = type(next(iter(iindex.values())))
            indexes[iID], slicers[iID] = subcls.merged(iindex, **kwargs)
            shape[iID] = len(indexes[iID])

        if xdim is not None:
            xID, xindex = next(iter(xdim.indexes.items()))
            indexes[xID] = xindex
            shape[xID] = len(indexes[xID])
        return indexes, slicers, shape


def inject_ops(cls):
    other_methods = {'item', 'argsort', 'searchsorted'}
    simple_ops = {'lt', 'le', 'ge', 'gt', 'eq', 'ne', 'neg', 'pos', 'abs',
                  'invert'}
    num_binary_ops = {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                      'pow', 'and', 'xor', 'or', 'matmul'}
    np_unary_methods = {'clip', 'conj', 'conj', 'real', 'imag', 'all', 'any'}
    nan_reduce_methods = {'argmax', 'argmin', 'max', 'min', 'mean', 'prod',
                          'sum', 'std', 'var', 'median'}

    for name in num_binary_ops | simple_ops:
        name = '__'+name+'__'
        setattr(cls, name, cls._make_op(name))

    for name in num_binary_ops:
        name = '__r'+name+'__'
        setattr(cls, name, cls._make_op(name))
        name = '__i'+name+'__'
        setattr(cls, name, cls._make_op(name))

    for name in (np_unary_methods | nan_reduce_methods):
        setattr(cls, name, cls._make_op(name))

    for name in nan_reduce_methods:
        name = 'nan'+name
        setattr(cls, name, cls._make_op(name))

inject_ops(Variables)