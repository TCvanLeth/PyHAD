# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:37:54 2015

@author: tcvanleth
"""

from collections import OrderedDict

import dask.array as da
import numpy as np

from phad import alignment
from phad import analysis as an
from phad import common as com
from phad import inout_common as io
from phad import plotting as pl
from phad import ufuncs as uf
from phad import variable as var
from phad.attributes import Attributes
from phad.variable import MetaVar, Variable
from phad.variables import Variables, Indexes, Coords


class Channel:
    """
    Lowest level complete dataframe structure.

    Contains a number of variables, indexes and coordinates plus attributes.
    """
    degree = 0
    treename = 'channel'
    meta = False

    def __init__(self, variables=None, indexes=None, coords=None, name='NA',
                 attrs=None):
        """
        """
        indexes = parse_indexes(indexes)
        newindexes, coords = parse_coords(coords)
        newnewindexes, newcoords, variables = parse_variables(variables)
        self.indexes = indexes.merge(newindexes).merge(newnewindexes)
        self._set_coords(coords.merge(newcoords))
        self._set_vars(variables)
        self.name = name
        self.attrs = attrs
        self.handles = set()

    @classmethod
    def _new(cls, indexes, coords, variables, name, attrs, handles):
        newchan = object.__new__(cls)
        newchan.indexes = indexes
        newchan.coords = coords
        newchan.variables = variables
        newchan._name = name
        newchan._attrs = attrs
        newchan.handles = handles
        return newchan

    def _set_coords(self, coords):
        self.coords = Coords()
        if not coords:
            return
        for cID, icoo in coords.items():
            if icoo.name in self.indexes:
                raise ValueError('coordinate %s is already an index'
                                 % icoo.name)
            dim = icoo.dims[0]
            size = len(icoo)
            if size != len(self.indexes[dim]):
                raise ValueError('coordinate %s has size uneqeal to associated'
                                 'dimension %s' % (icoo.name, size))
            self.coords[icoo.name] = icoo
            if dim not in self.indexes:
                self.indexes[dim] = var.carange(size, name=dim)

    def _set_vars(self, variables):
        self.variables = Variables()
        if not variables:
            return
        for vID, ivar in variables.items():
#            if ivar.name in self.indexes or ivar.name in self.coords:
#                raise ValueError('variable %s is already a coordinate'
#                                 % ivar.name)
            dims = ivar.dims
            shape = ivar.shape
            for i, idim in enumerate(dims):
                n = len(self.indexes[idim])
                if shape[i] != n:
                    raise ValueError('variable %s has size %s unequal to '
                                     'associated dimension %s with size %s'
                                     % (ivar.name, shape[i], idim, n))
            self.variables[vID] = ivar
            for i, idim in enumerate(dims):
                if idim not in self.indexes:
                    self.indexes[idim] = var.carange(shape[i], name=idim)

    def _maybe_call(self, other):
        newvars = {}
        for vID, ivar in other.variables.items():
            if callable(ivar):
                newvars[vID] = ivar(self)
            else:
                newvars[vID] = ivar
        return newvars

    def update(self, other, inplace=True):
#        newvars = self._maybe_call(other)
        newindexes, newcoords, newvars = parse_variables(other)
        indexes = self.indexes.copy()
        indexes.update(newindexes)
        coords = self.coords.copy()
        coords.update(newcoords)
        variables = self.variables.copy()
        variables.update(newvars)

        if inplace:
            newchan = self
        else:
            newchan = object.__new__(type(self))

        newchan.indexes = indexes
        newchan._set_coords(coords)
        newchan._set_vars(variables)
        newchan._name = self._name
        newchan._attrs = self._attrs
        return newchan

    def update_coords(self, other, inplace=True):
#        newcoords = self._maybe_call(other)
        newindexes, newcoords = parse_coords(other)
        if inplace:
            newchan = self
        else:
            newchan = self.copy()
        newchan.indexes.update(newindexes)
        newchan._set_coords(self.coords.update(newcoords))
        return newchan

    def set_coords(self, names, inplace=False):
        if isinstance(names, str):
            names = [names]

        if inplace:
            newchan = self
        else:
            newchan = self.copy()
        for name in names:
            newchan.coords[name] = newchan.variables[name]
            del newchan.variables[name]
        return newchan

    def reset_coords(self, names=None, drop=False, inplace=False):
        if names is None:
            names = self.coords.keys()
        else:
            if isinstance(names, str):
                names = [names]

        if inplace:
            newchan = self
        else:
            newchan = self.copy()
        for name in names:
            if not drop:
                newchan.variables[name] = newchan.coords[name]
            del newchan.coords[name]
        return newchan

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = 'NA'
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def names(self):
        return ([x.name for x in self.variables.values()] +
                [x.name for x in self.coords.values()] +
                [x.name for x in self.indexes.values()])

    def setname(self, name, **kwargs):
        self.variables.setname(name, **kwargs)

    def rename(self, name_dict):
        self.variables.rename(name_dict)
        self.coords.rename(name_dict)
        self.indexes.rename(name_dict)

    @property
    def dims(self):
        return self.variables.dims

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def step(self):
        return self.indexes.step

    def getstep(self, dim):
        return self.step[dim]

    def swap_dims(self, **dims_dict):
        """
        """
        self.indexes.replace(dims_dict)

        dims_dict = {k: v.name if hasattr(v, 'name') else v
                     for k, v in dims_dict.items()}
        self.indexes.swap_dims(dims_dict)
        self.coords.swap_dims(dims_dict)
        self.variables.swap_dims(dims_dict)

    @property
    def attrs(self):
        """Dictionary of attributes on this channel
        """
        if not hasattr(self, '_attrs'):
            self._attrs = Attributes()
        return self._attrs

    @attrs.setter
    def attrs(self, values):
        # set default values for standard variable attributes
        attr = Attributes()
        # only for links
#        attr['ATPC'] = common.get_fill(str)
#        attr['receiver_side'] = common.get_fill(str)
#        attr['polarization'] = common.get_fill(str)
#        attr['frequency'] = common.get_fill(float)
#        attr['modulation'] = common.get_fill(float)
#
#        # only for parsivel
#        attr['beam_height'] = common.get_fill(float)
#        attr['beam_width'] = common.get_fill(float)
#        attr['beam_length'] = common.get_fill(float)

        if values is not None:
            for key, value in values.items():
                attr[key] = value
        self._attrs = attr

    def setattrs(self, **kwargs):
        self.variables.setattrs(**kwargs)

    def delattrs(self, *args):
        self.variables.delattrs(*args)

    def getattrs(self, key, fallback=None):
        if key in self.attrs:
            fallback is self.attrs[key]

        params = self.variables.getattrs(key)
        if len(params) == 0:
            out = Array(self.attrs[key], (), name=key)
            out._name = self._name
            return out
        elif fallback is not None:
            for k in self.variables.keys():
                if k not in params:
                    params[k] = Variable(fallback, (), name=key)
        return self._new(Indexes(), Coords(), params, self.name, self.attrs,
                         self.handles)

    @property
    def _attr_sources(self):
        return [self]

    def __getattr__(self, name):
        if name != '__setstate__':
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the xray object is initialized
            for source in self._attr_sources:
                try:
                    return source[name]
                except KeyError:
                    pass
        raise AttributeError("%r object has no attribute %r" %
                             (type(self).__name__, name))

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = [item for sublist in self._attr_sources
                       for item in sublist]
        return sorted(set(dir(type(self)) + extra_attrs))

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return (key in self.variables or
                key in self.coords or
                key in self.indexes)

    def __len__(self):
        lens = [len(x) for x in self.indexes.values()]
        if not all(lens) or len(lens) == 0:
            return 0
        else:
            return max(lens)

    def __iter__(self):
        return iter(self.variables)

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.variables.values())

    def __getitem__(self, keys):
        """
        """
        if isinstance(keys, (Array, np.ndarray, da.Array)):
            if keys.dtype == bool:
                return self.mask(~keys)
            elif keys.dtype == int and self.ndim == 1:
                return self.isel(**{self.dims[0]: keys})
            else:
                raise Exception(self.ndim, keys.ndim)
        if com.is_dict_like(keys):
            return self.sel(**keys)

        if not isinstance(keys, tuple):
            keys = (keys,)
        if keys[0] is Ellipsis:
            keys = keys[1:]
        if keys[0] == slice(None):
            raise IndexError ('cannot use empty slice in channel object')
        elif isinstance(keys[0], list):
            keys = keys[0]

        newvars = Variables()
        newcoords = Coords()
        dims = set()
        dims2 = set()
        for vID in keys:
            if vID in self.variables:
                if len(keys) == 1:
                    newvars[None] = self.variables[vID].copy()
                else:
                    newvars[vID] = self.variables[vID]
                dims |= set(self.variables[vID].dims)
                dims2 |= set(self.variables[vID].dims)
            elif vID in self.coords:
                if len(keys) == 1:
                    newvars[None] = self.coords[vID]
                else:
                    newcoords[vID] = self.coords[vID]
                dims.add(self.coords[vID].dims[0])
            elif vID in self.indexes:
                if len(keys) == 1:
                    newvars[None] = self.indexes[vID].to_variable()
                dims.add(vID)
            else:
                raise KeyError(vID)

        dims = list(dims)
        dims2 = list(dims2)
        newindexes = self.indexes.select(dims=dims)
        if dims2:
            newcoords = newcoords.merge(self.coords.select(dims=dims2))
        if len(keys) == 1:
            attrs = Attributes.merged([newvars[None], self])
            newvars[None].attrs = attrs
            return Array._new(newindexes, newcoords, newvars, self._name,
                              Attributes(), self.handles)
        else:
            return self._new(newindexes, newcoords, newvars, self._name,
                             self._attrs, self.handles)

    def __setitem__(self, key, value):
        # set single whole variable
        if isinstance(key, str):
            self.update({key: value})
        elif key is slice(None):
            for ivar in self.variables.values():
                ivar[key] = value

        # set specific location for all variables
        elif hasattr(key, 'variables'):
            for vID, ivar in self.variables.items():
                if vID in key.variables:
                    ivar[key.variables[vID]] = value
        elif hasattr(key, 'dims'):
            for ivar in self.variables.values():
                ivar[key] = value
        elif com.is_dict_like(key):
            key = self.indexes.get_indexer(key)
            for ivar in self.variables.values():
                ivar[key] = value
        else:
            raise KeyError('no valid key for Channel: %s' % key)

    def __delitem__(self, keys):
        """Remove a variable from this dataset.

        If this variable is a dimension, all variables containing this
        dimension are also removed.
        """
        if not isinstance(keys, tuple):
            keys = (keys,)
        for key in keys:
            self.variables.pop(key, None)
            self.coords.pop(key, None)
            delvar = self.indexes.pop(key, None)
            if delvar:
                self.variables.drop(*self.variables.select(dims=[key]).keys())

    def copy(self, deep=False):
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy is made, so each variable in the new dataset
        is also a variable in the original dataset.
        """
        indexes = self.indexes.copy(deep=deep)
        coords = self.coords.copy(deep=deep)
        variables = self.variables.copy(deep=deep)
        attrs = self._attrs.copy()
        return self._new(indexes, coords, variables, self._name, attrs,
                         self.handles)

    def __copy__(self):
        return self.copy(deep=False)

    # mutable objects should not be hashable
    __hash__ = None

    def equals(self, other):
        """
        """
        try:
            var_eq = self.variables.equals(other.variables)
            coo_eq = self.coords.equals(other.coords)
            ind_eq = self.indexes.equals(other.indexes)
            return var_eq and coo_eq and ind_eq
        except (TypeError, AttributeError):
            return False

    def identical(self, other):
        """
        """
        try:
            att_eq = self._attrs.equals(other._attrs)
            var_eq = self.variables.identical(other.variables)
            coo_eq = self.coords.identical(other.coords)
            ind_eq = self.indexes.identical(other.indexes)
            return att_eq and var_eq and coo_eq and ind_eq
        except (TypeError, AttributeError):
            return False

    def __repr__(self):
        values = self.get_printvals()
        return self.pformat(*da.compute(values))

    def get_printvals(self):
        indexes = self.indexes.get_printvals()
        coords = self.coords.get_printvals()
        variables = self.variables.get_printvals()
        return {'indexes': indexes, 'coords': coords, 'variables': variables}

    def pformat(self, values):
        """
        Format a summary of the channel for printing to command line.
        """
        indexes = values['indexes']
        coords = values['coords']
        variables = values['variables']

        summary = ['<%s> %s' % (type(self).__name__, self.name)]
        col_width = max(tuple(len(str(k)) for k in self.names)+(7,)) + 6

        summary.append(self.indexes.pformat(indexes, 'dimensions', col_width))
        summary.append(self.coords.pformat(coords, 'coordinates', col_width))
        summary.append(self.variables.pformat(variables, 'variables', col_width))
        summary.append(self.attrs.__repr__(col_width))
        return '\n'.join(summary)

    def select(self, **kwargs):
        """
        make selection based on group attributes
        """
        newvars = self.variables.select(**kwargs)
        if newvars:
            dims = (set(i.dims) for i in newvars.values())
            dims = tuple(set.union(*dims))
            newindexes = self.indexes.select(name=dims)
            newcoords = self.coords.select(dims=dims)
        else:
            newindexes = Indexes()
            newcoords = Coords()
        return self._new(newindexes, newcoords, newvars, self._name,
                         self._attrs, self.handles)

    def aselect(self, **kwargs):
        """
        select first contained variable based on group attributes.
        """
        newchans = []
        newvars = self.variables.select(**kwargs)
        newvars.update(self.coords.select(**kwargs))
        newvars.update(self.indexes.select(**kwargs))
        for vID, ivar in newvars.items():
            newindexes = self.indexes.select(name=ivar.dims)
            newcoords = self.coords.select(dims=ivar.dims)
            newvars = Variables({None: ivar.to_variable()})
            newchan = Array._new(newindexes, newcoords, newvars, self._name,
                                 self._attrs, self.handles)
            newchans.append(newchan)
        if newchans:
            return newchans[0]
        else:
            newindexes = Indexes()
            newcoords = Coords()
            newvars = Variables()
            return self._new(newindexes, newcoords, newvars, self._name,
                             self._attrs, self.handles)

    def isel(self, **indexers):
        """
        Slice the channel according to orthogonal index number.
        """
        newvars = self.variables.isel(**indexers)
        dims = newvars.dims
        lostdims = tuple(set(self.dims) - set(dims))
        attrs = self._attrs
        if len(dims) == 0:
            newindexes  = Indexes()
            newcoords = Coords()
        else:
            newindexes = self.indexes.isel(**indexers)
            if lostdims:
                atindex = newindexes.select(dims=lostdims)
                for key, value in atindex.items():
                    attrs[key] = value.values[0]
            newindexes = newindexes.select(dims=dims)
            newcoords = self.coords.isel(**indexers).select(dims=dims)

        return self._new(newindexes, newcoords, newvars, self._name, attrs,
                         self.handles)

    def idrop(self, *variables, **indexers):
        newvars = self.variables.idrop(*variables, **indexers)
        newvars = newvars.idrop(*newvars.select(dims=variables).keys())
        newcoords = self.coords.idrop(*variables, **indexers)
        newcoords = newcoords.idrop(*newcoords.select(dim=variables).keys())
        newindexes = self.indexes.idrop(*variables, **indexers)
        return self._new(newindexes, newcoords, newvars, self.name,
                         self._attrs, self.handles)

    def sel(self, how=None, atol=None, **indexers):
        """
        Slice the channel according to orthogonal index variables.
        """
        newindexers = {}
        swapdims = {}
        for idim, iindex in indexers.items():
            if hasattr(iindex, 'variables'):
                if len(iindex.indexes) > 1:
                    raise Exception
                elif len(iindex.indexes) == 1:
                    swapdims[idim] = list(iindex.indexes.values())[0]
                newindexers[idim] = iindex.variables[None]
            else:
                newindexers[idim] = iindex

        locs = self.indexes.get_indexer(newindexers, how=how, atol=atol)
        new = self.isel(**locs)

        if len(swapdims) > 0:
            new.swap_dims(**swapdims)
        return new

    def drop(self, *variables, **indexers):
        """
        """
        locs = self.indexes.get_indexer(indexers)
        return self.idrop(*variables, **locs)

    def mask(self, mask):
        """
        Return a new channel with a mask applied to all variables.
        """
        inputs = alignment.align([self, mask])
        indexes = Indexes()
        coords = Coords()
        handles = set()
        for idat in inputs:
            if hasattr(idat, 'indexes'):
                indexes.update(idat.indexes)
            elif hasattr(idat, 'dims'):
                indexes[idat.name] = idat
            if hasattr(idat, 'coords'):
                coords.update(idat.coords)
            if hasattr(idat, 'handles'):
                handles |= idat.handles
        attrs = inputs[0]._attrs

        newvars = inputs[0].variables.mask(inputs[1].variables[None])
        return self._new(indexes, coords, newvars, self._name, attrs, handles)

    def transpose(self, *dims):
        """
        Rearange the named dimensions and asociated array axes.
        """
        if dims:
            if set(dims) ^ set(self.indexes.keys()):
                raise ValueError('arguments to transpose (%s) must be '
                                 'permuted dataset dimensions (%s)'
                                 % (dims, tuple(self.indexes.keys())))
        newchan = self.copy()
        newchan.variables = self.variables.transpose(*dims)
        for name, ivar in self.variables.items():
            var_dims = tuple(dim for dim in dims if dim in ivar.dims)
            newchan.variables[name] = ivar.transpose(*var_dims)
        return newchan

    @property
    def T(self):
        return self.transpose()

    def squeeze(self, dim=None):
        """
        """
        return com.squeeze(self, self.dims, dim)

    def resample(self, step, dim='time', how='mean'):
        """
        downsample to a multiple of the current step.

        Standard behaviour is to make an interval which contains any NAN value
        into a NAN value.
        """
        newchan = self.ortho(step, dim)
        newchan = getattr(uf, how)(newchan, dim=dim+'_offset')
        return newchan

    def upsample(self, how='ffill', **steps):
        steps = {x: steps[x] for x in steps.keys() if x in self.indexes}
        if steps == self.step:
            return self
        newchan = self
        for dim, step in steps.items():
            tag = 'offset'
            while '_'.join([dim, tag]) in self.indexes:
                tag = tag + '1'
            newchan = newchan.extend(step, dim, how=how, tag=tag)
            newchan = newchan.inline(dim, tag=tag)
        return newchan

    def groupby(self, *dims):
        newchan = self
        for dim in dims:
            if isinstance(newchan, list):
                chans = []
                for ID, ichan in newchan:
                    chans += ichan.split(dim, name=ID)
                newchan = chans
            else:
                newchan = newchan.split(dim)
        return newchan

    def extend(self, newstep, dim, how=None, tag='offset'):
        if dim not in self.indexes:
            return self
        if self.indexes[dim].dtype.kind == 'M':
            newstep = com.str_to_td64(newstep)
        oldstep = self.indexes[dim].step
        if newstep == oldstep:
            return self

        name = dim + '_' + tag
        chunksize = self.indexes[dim].chunksize
        index = var.carange(0, oldstep, newstep, name=name,
                            chunksize=chunksize)

        attrs = self._attrs.copy()
        indexes = self.indexes.copy()
        indexes[name] = index
        coords = self.coords.extend(index, how=how, dim=dim)
        variables = self.variables.extend(index, how=how, dim=dim)
        return self._new(indexes, coords, variables, self._name, attrs,
                         self.handles)

    def split(self, dim, name=None):
        if '.' in dim:
            dim, step = dim.split('.')
            step = com.time_component(step)
            newchan = self.ortho(step, dim, tag='group')
            newchan._attrs['temporal_resolution'] = self.indexes[dim].step
        else:
            newchan = self
        if dim not in newchan:
            if name is None:
                return newchan
            return [(name, newchan)]

        n = len(newchan[dim])
        varlist = newchan.variables.split(dim, n)
        coordlist = newchan.coords.split(dim, n)
        indexlist = newchan.indexes.split(dim, n)

        chanlist = []
        for i in range(n):
            attrs = newchan._attrs
            ID = com.to_nptype(indexlist[i].pop(dim).values.item())
            attrs[dim] = ID
            ID = frozenset(((dim, ID),))
            if dim + '_group' in indexlist[i]:
                grp = indexlist[i].pop(dim + '_group')
                grp += attrs.pop(dim)
                if dim == 'time':
                    indexlist[i][dim] = var.TimeScale(grp.values)
                else:
                    indexlist[i][dim] = grp
                indexlist[i][dim].name = dim
                indexlist[i][dim].dims = (dim,)
            for cID, icoo in coordlist[i].items():
                if len(icoo.dims) == 0:
                    attrs[cID] = coordlist[i].pop(cID).values[0]

            indexes = indexlist[i]
            coords = coordlist[i]
            variables = varlist[i]
            ichan = self._new(indexes, coords, variables, self._name, attrs,
                              self.handles)
            if name is None:
                chanlist.append((ID, ichan))
            else:
                chanlist.append((name | ID, ichan))
        return chanlist

    def ortho(self, step, dim, tag='offset', rename=False):
        """
        losslessly conform to lower resolution by pushing high resolution to
        orthogonal 'offset' dimension.

        NOTE: does not work for months, years, etc. which represent an
        irregular timespan
        """
        if dim not in self.indexes:
            return self

        oldindex = self.indexes[dim]
        if oldindex.dtype.kind == 'M':
            newstep = com.str_to_td64(step).astype('<m8[us]')
        else:
            newstep = step
        oldstep = oldindex.step
        if newstep == oldstep:
            return self
        if newstep < oldstep:
            raise ValueError('new sampling step must be larger than current step')

        start = oldindex.start.astype('<M8[us]')
        stop = oldindex.stop.astype('M8[us]')

        name = dim + '_' + tag
        if oldindex.dtype.kind == 'M':
            newstart = start.astype(int)
            instep = newstep.astype(int)
            newstart = (newstart - newstart % instep).astype('<M8[us]')

            newstop = stop.astype(int)
            newstop = (newstop - newstop % instep).astype('<M8[us]')
        else:
            newstart = start - start % newstep
            newstop = stop - stop % newstep

        aligned = int((start - newstart) / oldstep)
        index1 = var.carange(newstart + newstep, newstop + newstep, newstep, name=dim)
        index2 = var.carange(0, newstep, oldstep, name=name)

        index2.attrs['quantity'] = index1.attrs['quantity']
        attrs = self._attrs.copy()
        if dim == 'time':
            attrs['temporal_resolution'] = step
        indexes = self.indexes.copy()
        indexes[dim] = index1
        indexes[name] = index2
        coords = self.coords.ortho(index1, index2, rename, aligned)
        variables = self.variables.ortho(index1, index2, rename, aligned)
        return self._new(indexes, coords, variables, self._name, attrs,
                         self.handles)

    def inline (self, dim, tag='offset'):
        """
        reverse of ortho
        """
        name = dim + '_' + tag
        if name not in self.indexes:
            return self

        start = self.indexes[dim].start
        stop = self.indexes[dim].stop
        oldstep = self.indexes[dim].step
        step = self.indexes[name].step

        chunksize = self.indexes[dim].chunksize
        newindex = var.carange(start - oldstep, stop - oldstep, step, chunksize=chunksize)

        attrs = self._attrs.copy()
        indexes = self.indexes.copy()
        indexes[dim] = newindex
        del indexes[name]
        coords = self.coords.inline(newindex, name)
        variables = self.variables.inline(newindex, name)
        return self._new(indexes, coords, variables, self._name, attrs,
                         self.handles)

    def rechunk(self, chunks):
        """
        Rearange the splits in the chunked array.
        """
        newvars = self.variables.rechunk(chunks)
        newcoords = self.coords.rechunk(chunks)
        newindexes = self.indexes.rechunk(chunks)
        return self._new(newindexes, newcoords, newvars, self._name,
                         self._attrs, self.handles)

    def astype(self, dtype):
        """
        Convert all variables to the given numerical type.
        """
        newvars = self.variables.astype(dtype)
        return self._new(self.indexes, self.coords, newvars, self._name,
                         self._attrs, self.handles)

    @property
    def level(self):
        return 1

    def conform(self):
        """
        Convert all dimensional variables to base SI units.
        """
        newvars = self.variables.conform()
        newcoords = self.coords.conform()
        newindexes = self.indexes.conform()
        return self._new(newindexes, newcoords, newvars, self._name,
                         self._attrs, self.handles)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to all the variables in the channel.
        """
        newvars = self.variables.apply(func, *args, **kwargs)
        return self._new(self.indexes, self.coords, newvars, self._name,
                         self._attrs, self.handles)

    @staticmethod
    def _make_op(method):
        def func(self, *args, **kwargs):
            return self._invoke_method(method, (self, *args), **kwargs)
        return func

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        return self._invoke_method(method, inputs, obj=ufunc, **kwargs)

    def _invoke_method(self, method, inputs, obj=None, **kwargs):
        if not all(isinstance(x, self.comp_tup) for x in inputs):
            return NotImplemented
        if any(isinstance(x, Channel) and not x.isvariable() for x in inputs):
            if self.isvariable():
                return NotImplemented

        inputs = alignment.align(inputs)

        indexes = Indexes()
        coords = Coords()
        handles = set()
        for idat in inputs:
            if hasattr(idat, 'indexes'):
                indexes.update(idat.indexes)
            elif hasattr(idat, 'dims'):
                indexes[idat.name] = idat
            if hasattr(idat, 'coords'):
                coords.update(idat.coords)
            if hasattr(idat, 'handles'):
                handles |= idat.handles

        kwargs2 = {**kwargs}
        new_index = kwargs2.pop('new_index', None)
        if new_index is not None:
            dims = ()
            chunks = ()
            for iindex in new_index:
                dims += (iindex.name,)
                chunks += iindex.chunks
                indexes[iindex.name] = iindex
            kwargs2['new_chunks'] = chunks
            kwargs2['new_dim'] = dims

        drop_dim = kwargs2.get('drop_dim')
        if drop_dim is not None:
            for iindex in drop_dim:
                del indexes[iindex]

        window = kwargs2.get('window')
        if window is not None:
            window2 = {}
            for k, v in window.items():
                if self.indexes[k].dtype.kind == 'M':
                    v = com.str_to_td64(v)
                window2[k] = int(v / indexes[k].step)
            kwargs2['window'] = window2

        args = tuple(x.variables if hasattr(x, 'variables') else x
                     for x in inputs)
        if obj is None:
            obj, args = args[0], args[1:]
        outvars = getattr(obj, method)(*args, **kwargs2)

        attrs = Attributes.merge_reduce([idat for idat in inputs
                                        if hasattr(idat, 'variables')])
        if isinstance(outvars, tuple):
            return tuple(self._wrap_output(x, indexes, coords, attrs, handles)
                        for x in outvars)
        return self._wrap_output(outvars, indexes, coords, attrs, handles)

    def _wrap_output(self, outvars, indexes, coords, outattrs, handles):
        if isinstance(next(iter(outvars.values())), MetaVar):
            return MetaChan(params=outvars, attrs=outattrs)
        if outvars.dims:
            outindexes = indexes.select(dims=outvars.dims)
        else:
            outindexes = Indexes()
        for outvar in outvars.values():
            for idim in outvar.dims:
                if idim not in indexes:
                    axis = outvar.get_axis_num(idim)[0]
                    outindexes[idim] = var.carange(0, outvar.shape[axis], 1,
                                                   name=idim)

        outcoords = coords.select(dims=outvars.dims)
        return self._new(outindexes, outcoords, outvars, self._name, outattrs,
                         handles)

    def _compute(self):
        vals = {}
        for iname, ivar in self.variables.items():
            vals[iname] = ivar._values
        return vals

    def compute(self):
        """
        Perform any delayed calculations.
        """
        vals = self._compute()
        outvals = da.compute(vals)[0]
        return self._comp_out(outvals)

    def _comp_out(self, outvals):
        outvars = Variables()
        for iname, ival in outvals.items():
            ivar = self.variables[iname]
            outvars[iname] = Variable(ival, ivar.dims, ivar.name, ivar.attrs,
                                      chunksize=ivar.chunksize)
        return self._new(self.indexes, self.coords, outvars, self._name, self.attrs,
                         self.handles)

    def isvariable(self):
        return len(self.variables) == 1 and None in self.variables

    def reindex(self, how=None, **indexers):
        """
        Rearrange and interpolate the data in this channel to a new index variable.
        """
        newindexers = {}
        swapdims = {}
        for idim, iindex in indexers.items():
            if hasattr(iindex, 'variables'):
                if len(iindex.indexes) > 1:
                    raise Exception
                elif len(iindex.indexes) == 1:
                    swapdims[idim] = list(iindex.indexes.values())[0]
                newindexers[idim] = iindex.variables[None]
            else:
                newindexers[idim] = var.as_index(iindex, name=idim)

        new = self._reindex(newindexers, how=how)

        if len(swapdims) > 0:
            new.swap_dims(**swapdims)
        return new

    def reindex_like(self, other, how=None):
        indexers = other.indexes
        return self._reindex(indexers, how=how)

    def _reindex(self, indexers, how=None):
        locs = self.indexes.get_indexer(indexers, how=how)
        return self._apply_index(locs, indexers)

    def _apply_index(self, locs, indexers):
        newvars = self.variables.apply_index(locs)
        dims = newvars.dims
        newindexes = self.indexes.copy().select(dims=dims)
        newindexes.update({k:v for k, v in indexers.items() if k in dims})
        newcoords = self.coords.apply_index(locs).select(dims=dims)
        newattrs = self._attrs
        handles = self.handles
        return self._new(newindexes, newcoords, newvars, self._name, newattrs,
                         handles)

    def _interpolate(self, indexers, how='linear'):
        pass
        # for now assume 1d.

    def diff(self, dim, n=1, label='upper'):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : str, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        Examples
        --------
        >>> ds = xray.Dataset({'foo': ('x', [5, 5, 6, 6])})
        >>> ds.diff('x')
        <xray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 1 2 3
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff('x', 2)
        <xray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
        * x        (x) int64 2 3
        Data variables:
        foo      (x) int64 1 -1

        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError('order `n` must be non-negative but got {0}'
                             ''.format(n))

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == 'upper':
            kwargs_new = kwargs_end
        elif label == 'lower':
            kwargs_new = kwargs_start
        else:
            raise ValueError('The \'label\' argument has to be either '
                             '\'upper\' or \'lower\'')

        variables = OrderedDict()

        for name, ivar in self.variables.items():
            if dim in ivar.dims:
                if name in self.variables:
                    variables[name] = (ivar.isel(**kwargs_end) -
                                       ivar.isel(**kwargs_start))
                else:
                    variables[name] = ivar.isel(**kwargs_new)
            else:
                variables[name] = ivar

        difference = self._replace_vars_and_dims(variables)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def _create_store(self, outh):
        """
        write a single channel to the given hdf5 group handle
        """
        attrs = self._convert_for_store()
        attrs.to_store(outh.attrs)

        sources, targets = self.indexes._create_store(outh)
        s, t = self.coords._create_store(outh, coord=True)
        sources += s
        targets += t
        s, t = self.variables._create_store(outh)
        sources += s
        targets += t
        return sources, targets

    def _convert_for_store(self):
        attrs = self._attrs
        attrs['channel_id'] = self._name
        return attrs

    def store(self, outpath):
        """
        Store channel as hdf5 file in the given path.
        """
        io.to_hdf(self, outpath)

    @classmethod
    def merged(cls, datlist, xdim=None, **kwargs):
        """
        concatenate/merge channels

        combine the variables in the list of Channel objects 'datlist' into one
        new Channel object. If the same variable is present in more than one
        Channel, try to concatenate them along existing dimensions according to
        their coordinate values. If multiple similarly named variables have
        overlapping coordinates, then for each element of the variable,
        preference is given to the first non-NA value in list order.

        when xdim is specified, Variables are concatenated along a new
        dimension 'xdim' instead of along existing dimensions. any gaps are
        filled with NA values appropriate to data type.
        """
        if not datlist:
            return cls()
        datlist = com.prep_merger(datlist)

        # upsample regular arrays
        steps = alignment.get_highest_res2(datlist)
        datlist = alignment.upsample2(datlist, steps)
        if len(datlist) == 1:
            return next(iter(datlist.values()))

        # concatenate arrays
        if xdim is None:
            attrs = Attributes.merged(datlist)
        else:
            attrs, xdim = merge_attrs(datlist, xdim)
        indexes, slicers, length = Indexes.merged(datlist, xdim=xdim, **kwargs)
        coords = Coords.merged(datlist, slicers, length, xdim=xdim, **kwargs)

        if xdim is not None:
            xdim = next(iter(xdim.indexes.keys()))

        if cls == Array and xdim is not None:
            newvars = Variables()
            newvars[None] = Variable.merged([x.variables[None] for x in datlist.values()], slicers,
                                            length, xdim=xdim, **kwargs)
        else:
            newvars = Variables.merged(datlist, slicers, length, xdim=xdim, **kwargs)
            if cls == Array and len(newvars) > 1:
                cls = Channel
            if cls == Array:
                newvars[None] = newvars.popitem()[1]

        handles = set.union(*(v.handles for k, v in datlist.items()))
        name = next(iter(datlist.values()))._name
        newchan = cls._new(indexes, coords, newvars, name, attrs, handles)
        return newchan

    def merge(self, other, **kwargs):
        """
        Concatenate/merge given channel with this channel.
        """
        return self.merged([self, other], **kwargs)

    @classmethod
    def from_store(cls, inh, hi_res=False):
        """
        read a single channel from hdf5 group handle
        """
        attrs = Attributes.from_store(inh)
        name = attrs.pop('channel_id')

        # read coordinates and variables
        indexes = Indexes()
        coords = Coords()
        variables = Variables()
        for lID, leaf in list(inh.items()):
            if 'time_offset' in [idim.label for idim in leaf.dims] and not hi_res:
                continue
            arr = var.from_store(leaf)
            if io.isindex(leaf):
                indexes[lID] = arr
            elif io.iscoord(leaf):
                coords[lID] = arr
            else:
                variables[lID] = arr

#        if len(variables) == 1:
#            return Array._new(indexes, coords, variables, attrs, set())
        return cls._new(indexes, coords, variables, name, attrs, set())

    @classmethod
    def from_yaml(cls, inh, **kwargs):
        """
        Read metadata from a yaml file into a new Channel.
        """
        data = MetaChan(attrs=Attributes(inh['attrs']))
        for ID, iparam in inh['params'].items():
            iparam['array_id'] = ID
            data.variables[ID] = MetaVar.from_yaml(iparam, **kwargs)
        return data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.close()

    def plot(self, xname, yname, style='o', **kwargs):
        """
        Draw a graph plotting two variables in the channel against each other.
        """
        x = self[xname]
        y = self[yname]
        return pl.plot(x, y, style, aligned=True, **kwargs)

    def histplot(self, xname, yname, **kwargs):
        """
        Draw a 2D histogram of two variables in the channel.
        """
        x = self[xname]
        y = self[yname]
        return pl.hist2dplot(x, y, aligned=True, **kwargs)

    def kdeplot(self, xname, yname, **kwargs):
        """
        Draw a 2D kernel density plot of two variables in the channel.
        """
        x = self[xname]
        y = self[yname]
        return pl.kdeplot(x, y, aligned=True, **kwargs)

    def corr(self, *names, **kwargs):
        """
        Calculate pearson corrlation coefficients for the given variables.
        """
        variables = tuple(self[name] for name in names)
        return an.corr(*variables, **kwargs)


def merge_attrs(datlist, xdim):
    datlist = com.prep_merger(datlist)

    n = max(datlist.keys(), default=0) + 1
    dd = {}
    for i, idat in datlist.items():
        for key, value in idat._attrs.items():
            if key not in dd:
                dd[key] = com.null_array(n, type(value))
            dd[key][i] = value

    if isinstance(xdim, str):
        xname = xdim
        if xname in dd:
            xcoo = var.Index(dd[xname], xname)
        else:
            xcoo = var.Index(np.arange(n), xdim)
        xdim = Channel(coords={xdim: xcoo})
    elif hasattr(xdim, 'indexes'):
        xname = next(iter(xdim.indexes.keys()))
        if xname in dd:
            xcoo = var.Index(dd[xname], xname)
            xdim.update_coords({xname:xcoo})
    elif hasattr(xdim, 'name'):
        xname = xdim.name
        xcoo = xdim
        xdim = Channel(coords={xdim.name: xdim})
    else:
        raise ValueError(xname)

    attrs = Attributes()
    for key, value in dd.items():
        if np.all(value == value[0]):
            attrs[key] = value[0]
        elif np.all(com.isnull(value)) or key == xname:
            continue
        else:
            xdim[key] = var.Variable(value, xname, name=key)
    return attrs, xdim


def parse_indexes(values):
    if hasattr(values, 'indexes'):
        return values.indexes
    newindexes = Indexes()
    if com.is_dict_like(values):
        for key, value in values.items():
            if hasattr(value, 'variables'):
                value = value.variables[None]
            newindexes[key] = var.as_index(value, name=key)
    return newindexes


def parse_coords(values):
    newindexes = Indexes()
    newcoords = Coords()
    if not values:
        return newindexes, newcoords

    if com.is_dict_like(values):
        for key, value in values.items():
            if hasattr(value, 'indexes'):
                newindexes = newindexes.merge(value.indexes)
                if not value.name in value.dims:
                    newcoords[key] = value.variables[None]
            else:
                newval = var.as_variable(value, name=key)
                if newval.ndim != 1:
                    raise TypeError('coordinate %s has more then one dimension'
                                    % key)
                if newval.name == newval.dims[0]:
                    newindexes[key] = newval.to_index()
                else:
                    newcoords[key] = value
    else:
        for value in values:
            key = value.name
            if hasattr(value, 'indexes'):
                newindexes = newindexes.merge(value.indexes)
                if not value.name in value.dims:
                    newcoords[key] = value.variables[None]
            else:
                newval = var.as_variable(value, name=key)
                if newval.ndim != 1:
                    raise TypeError('coordinate %s has more then one dimension'
                                    % key)
                if newval.name == newval.dims[0]:
                    newindexes[key] = newval.to_index()
                else:
                    newcoords[key] = value
    return newindexes, newcoords

def parse_variables(values):
    newindexes = Indexes()
    newcoords = Coords()
    newvars = Variables()
    if not values:
        return newindexes, newcoords, newvars

    if com.is_dict_like(values):
        for key, value in values.items():
            if hasattr(value, 'indexes'):
                newindexes = newindexes.merge(value.indexes)
                newcoords = newcoords.merge(value.coords)
                if not value.name in value.dims:
                    newvars[key] = value.variables[None]
            else:
                newvars[key] = var.as_variable(value)
            #if not hasattr(value, 'name'):
            newvars[key].name = key
    else:
        if not isinstance(values, (list, tuple)):
            values = (values,)
        for value in values:
            key = value.name
            if hasattr(value, 'indexes'):
                newindexes = newindexes.merge(value.indexes)
                newcoords = newcoords.merge(value.coords)
                if not value.name in value.dims:
                    newvars[key] = value.variables[None]
            else:
                newvars[key] = var.as_variable(value)
    return newindexes, newcoords, newvars


def union(inputs):
    """
    take the union of all dimensions
    """
    if len(inputs) == 1:
        return inputs[0].indexes, inputs[0].coords
    inputs = list(inputs)
    indexes, slicers, shape = Indexes.merged(inputs)
    coords = Coords.merged(inputs, slicers, shape)
    return indexes, coords


def align(inputs, indexes):
    """
    align variables to indexes
    """
    newlist = []
    attlist = []
    for idat in inputs:
        if hasattr(idat, '_reindex'):
            newlist += [idat._reindex(indexes)]
        else:
            newlist += [idat]
        if hasattr(idat, 'variables'):
            attlist += [idat]
    newattrs = Attributes.merged(attlist)
    return newlist, newattrs


Channel.comp_tup = var.Variable.comp_tup+(Channel,)

def inject_ops(cls):
    """
    Helper function to add unversal function methods to array structures.
    """
    other_methods = {'item', 'argsort', 'searchsorted'}
    simple_ops = {'lt', 'le', 'ge', 'gt', 'eq', 'ne', 'neg', 'pos', 'abs', 'invert'}
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

inject_ops(Channel)


###############################################################################
class Array(Channel):
    """
    Special dataframe containing one variable and its indexes.
    """
    def __init__(self, values=None, coords=None, name=None, attrs=None):
        dims = ()
        newcoords = {}
        if values is None:
            newvar = var.Variable(None, ())
            super().__init__(variables={None: newvar})
            return
        if not hasattr(values, 'shape'):
            values = com.to_nptype(values)

        if coords:
            for i, icoo in enumerate(coords):
                if hasattr(icoo, 'name'):
                    dims += (icoo.name,)
                    newcoords[dims[i]] = icoo
                else:
                    dims += (icoo,)
                    newcoords[dims[i]] = var.carange(values.shape[i], name=icoo)
        else:
            for i, shp in enumerate(values.shape):
                dims += ('dim' + str(i),)
                newcoords[dims[i]] = var.carange(shp, name=name)

        newvar = var.as_variable(values, dims=dims, name=name, attrs=attrs)
        self._attrs = Attributes()
        self._name = 'NA'
        super().__init__(variables={None: newvar}, coords=newcoords,
                         name=name, attrs=attrs)

    def __getattribute__(self, name):
        if name == 'name':
            return self.variables[None].name
        elif name == 'values':
            return self.variables[None].values
        elif name == 'dims':
            return self.variables[None].dims
        elif name == 'attrs':
            return self.variables[None].attrs
        elif name == 'shape':
            return self.variables[None].shape
        elif name == 'dtype':
            return self.variables[None].dtype
        elif name == '__array__':
            return self.variables[None].__array__
        elif name == 'chunks':
            return self.variables[None].chunks
        elif name == 'task':
            return self.variables[None]._values
        elif name == 'get_axis_num':
            return self.variables[None].get_axis_num
        elif name == 'chunksize':
            return self.variables[None].chunksize
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == 'name':
            self.variables[None].name = value
        elif name == 'values':
            self.variables[None].values = value
        elif name == 'dims':
            self.variables[None].dims = value
        elif name == 'attrs':
            self.variables[None].attrs = value
        elif name == 'shape':
            self.variables[None].shape = value
        elif name == 'dtype':
            self.variables[None].dtype = value
        elif name == 'task':
            self.variables[None]._values = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == 'name':
            del self.variables[None].name
        elif name == 'values':
            del self.variables[None].values
        elif name == 'dims':
            del self.variables[None].dims
        elif name == 'attrs':
            del self.variables[None].attrs
        elif name == 'shape':
            del self.variables[None].shape
        elif name == 'dtype':
            del self.variables[None].dtype
        else:
            object.__delattr__(self, name)

    def __setitem__(self, key, value):
        # set single whole variable
        if isinstance(key, str):
            self.update({key: value})
        elif key is Ellipsis:
            for ivar in self.variables.values():
                ivar[key] = value
        else:
            # set specific location for all variables
            if isinstance(value, Array):
                value = value.values

            if hasattr(key, 'variables'):
                for vID, ivar in self.variables.items():
                    if vID in key.variables:
                        ivar[key.variables[vID]] = value
            elif hasattr(key, 'dims'):
                for ivar in self.variables.values():
                    ivar[key] = value
            elif com.is_dict_like(key):
                key = self.indexes.get_indexer(key)
                for ivar in self.variables.values():
                    ivar[key] = value
            else:
                raise KeyError('no valid key for Channel: %s' % key)

    def append(self, other):
        newvar = self.variables[None].append(other.variables[None])
        newvars = Variables({None: newvar})
        newindexes = self.indexes.merge(other.indexes)
        newcoords = self.coords.merge(other.coords)
        handles = self.handles
        return self._new(newindexes, newcoords, newvars, self._name,
                         self._attrs, handles)

    def plot(self, dim=None, style='-', **kwargs):
        if dim is None:
            x = next(iter(self.indexes.values()))
        else:
            x = self[dim]
        y = self
        return pl.plot(x, y, style, aligned=True, **kwargs)

    def barplot(self, **kwargs):
        x = next(iter(self.indexes.values()))
        for key in self.coords.keys():
            if 'binwidth' in key:
                kwargs['width'] = self.coords[key]
                break
        y = self
        return pl.barplot(x, y, aligned=True, **kwargs)

    def pixplot(self, **kwargs):
        z = self
        indexes = []
        for idim in self.dims[::-1]:
            iindex = self.indexes[idim]
            if iindex.step is None:
                step = np.diff(iindex.values).min()
                iindex = var.carange(iindex.start, iindex.stop, step, name=idim,
                                     attrs=iindex.attrs)
                z = z.reindex(**{idim: iindex}, how='nearest')
            indexes.append(iindex)
        return pl.meshplot(*indexes, z, aligned=True, **kwargs)

    def contplot(self, **kwargs):
        indexes = []
        for idim in self.dims[::-1]:
            indexes.append(self.indexes[idim])
        z = self
        return pl.contourplot(*indexes, z, aligned=True, **kwargs)

    def fillplot(self, **kwargs):
        x = next(iter(self.indexes.values()))
        y = self
        return pl.fillplot(x, y, **kwargs)

    def histplot(self, **kwargs):
        return pl.histplot(self, aligned=True, **kwargs)


###############################################################################
class MetaChan(Channel):
    """
    Metadata dataframe structure. Contains only attributes.
    """
    _sub = MetaVar
    treename = 'channel'
    meta = True

    def __init__(self, params=None, attrs=None):
        if params:
            self.variables = params
        else:
            self.variables = OrderedDict()

        self.attrs = Attributes()
        self.attrs['level'] = com.get_fill(str)
        self.attrs['set_id'] = com.get_fill(str)
        self.attrs['pro_id'] = com.get_fill(str)
        self.attrs['ptype'] = com.get_fill(str)
        self.attrs['htype'] = com.get_fill(str)
        self.attrs['climate'] = com.get_fill(str)

        if attrs:
            for key, values in attrs.items():
                self.attrs[key] = values

    def setattrs(self, **kwargs):
        for iparam in self.variables.values():
            iparam.setattrs(**kwargs)

    def delattrs(self, *args):
        for iparam in self.variables.values():
            iparam.delattrs(*args)

    def setname(self, name, **kwargs):
        for key in tuple(self.variables.keys()):
            iparam = self.variables.pop(key)
            iparam.setname(name, **kwargs)
            self[iparam.name] = iparam

    def __setitem__(self, key, value):
        self.variables[key] = value

    def __getitem__(self, key):
        new = self.variables[key]
        new.attrs = Attributes.merged([new, self])
        return new

    def __repr__(self):
        string = '<MetaChan> \n'
        for ID, iparam in self.variables.items():
            string += repr(iparam) + '\n'
        string += repr(self.attrs)
        return string

    def get_printvals(self):
        v_short = {}
        for k, v in self.variables.items():
            v_short[k] = v.pars
        return v_short

    def pformat(self, values):
        string = '<MetaChan> \n'
        for ID, iparam in self.variables.items():
            string += iparam.pformat(values[ID]) + '\n'
        string += repr(self.attrs)
        return string

    def select(self, **kwargs):
        data = self
        for key, values in kwargs.items():
            select = type(self)()
            if not values:
                continue
            for vID, ivar in data.variables.items():
                if ivar.meta(key, values):
                    select.variables[vID] = ivar
            data = select
        return data

    def aselect(self, **kwargs):
        newvars = list(self.select(**kwargs).variables.values())
        if newvars:
            return newvars[0]
        else:
            return type(self)()

    def sel(self, **kwargs):
        return self

    def to_store(self, outh):
        outh['attrs'] = {}
        self.attrs.to_store(outh['attrs'])
        outh['params'] = {}
        for ID, iparam in self.variables.items():
            outh['params'][ID] = {}
            iparam.to_store(outh['params'][ID])

    @classmethod
    def from_yaml(cls, inh, **kwargs):
        data = cls(attrs=Attributes(inh['attrs']))
        for ID, iparam in inh['params'].items():
            iparam['array_id'] = ID
            data.variables[ID] = cls._sub.from_yaml(iparam, **kwargs)
        return data

    @classmethod
    def merged(cls, datlist, **kwargs):
        if not datlist:
            return cls()
        datlist = com.prep_merger(datlist)
        newdat = cls(attrs=Attributes.merged(datlist))

        for i, idat in datlist.items():
            for ID, iparam in idat.variables.items():
                newdat.variables[ID] = iparam
        return newdat


###############################################################################
def index(values, name, attrs=None, **kwargs):
    """
    Create new index array.
    """
    _index = var.Index(values, name, attrs, **kwargs)
    return Array(values, [_index], name, attrs)


def arange(start, stop, step, name, **kwargs):
    """
    Create a new index array containing a continuous range of values.
    """
    values = var.carange(start, stop, step, name, **kwargs)
    return Array(values.to_variable(), [values], name, **kwargs)


def zeros(indexes, dtype=float, **kwargs):
    """
    create an array filled with zeros.
    """
    shape = OrderedDict((idim.name, len(idim)) for idim in indexes)
    values = var.zeros(shape, dtype)
    return Array(values, indexes, **kwargs)


def zeros_like(data, dtype=None, **kwargs):
    """
    Create and array filled with zeros.
    """
    indexes = data.indexes.values()
    if dtype is None:
        dtype = data.dtype
    return zeros(indexes, dtype, **kwargs)


def ones(indexes, dtype=float, **kwargs):
    """
    Create and array filled with ones.
    """
    shape = OrderedDict((idim.name, len(idim)) for idim in indexes)
    values = var.ones(shape, dtype)
    return Array(values, indexes, **kwargs)


def ones_like(data, dtype=None, **kwargs):
    """
    Create an array filled with ones.
    """
    indexes = data.indexes.values()
    if dtype is None:
        dtype = data.dtype
    return ones(indexes, data.dtype, **kwargs)


def full(indexes, fill_value, dtype=None, **kwargs):
    """
    Create an array filled with a given fill value.
    """
    shape = OrderedDict((idim.name, len(idim)) for idim in indexes)
    values = var.full(shape, fill_value, dtype=dtype)
    return Array(values, indexes, **kwargs)


def full_like(data, dtype=None, **kwargs):
    """
    Create an array with a given fill value.
    """
    indexes = data.indexes.values()
    if dtype is None:
        dtype = data.dtype
    return full(indexes, data.dtype, **kwargs)
