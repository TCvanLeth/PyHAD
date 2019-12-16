# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:43:09 2016

@author: tcvanleth
"""

from collections import OrderedDict
from types import FunctionType

import dask.array as da
import numpy as np

from phad import common as com
from phad import creation
from phad import indexing
from phad import inout_common as io
from phad import printing as pri
from phad import units
from phad.assignment import assign
from phad.attributes import Attributes
from phad.selection import select
from phad.resampling import reshape


class Variable(object):
    """
    """
    treename = 'variable'
    meta = False

    def __init__(self, values, dims, name=None, attrs=None, start=None,
                 stop=None, step=None, chunksize=com.CHUNKSIZE):
        """
        """
        # TODO: datetime conversion!
        self.chunksize = chunksize
        self.attrs = attrs
        self.name = name
        self.values = values
        self.dims = dims

    @classmethod
    def _new(cls, values, dims, name, attrs, start=None, stop=None, step=None,
             chunksize=com.CHUNKSIZE):
        """
        shortcut to skip data checking and conformation on creation
        """
        newvar = object.__new__(cls)
        newvar._values = values
        newvar._dims = dims
        newvar._name = name
        newvar._attrs = attrs
        newvar.chunksize = chunksize
        return newvar

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        self._name = value

    def setname(self, name, include=None, append=False):
        if include is not None:
            name = '_'.join([name] + [str(self.attrs[i]) for i in include])
        if append:
            name = self.name + name
        self.name = name

    @property
    def values(self):
        if not isinstance(self._values, np.ndarray):
            values = np.asarray(self._values)
        else:
            values = self._values
        return values

    @values.setter
    def values(self, values):
        if not isinstance(values, (np.ndarray, da.Array)):
            values = np.asarray(values)
        if isinstance(values, np.ndarray):
            values = da.from_array(values, chunks=self.chunksize)
        if hasattr(self, '_values') and values.shape != self.shape:
            raise ValueError(
                "replacement data must match the Variable's shape")

        if values.dtype.char in ('S', 'U'):
            values = values.astype('U64')
        elif values.dtype.kind == 'M':
            values = values.astype('datetime64[us]')
        elif values.dtype.kind == 'm':
            values = values.astype('timedelta64[us]')
        self._values = values

    @property
    def dims(self):
        """
        Tuple of dimension names with which this variable is associated.
        """
        return self._dims

    @dims.setter
    def dims(self, values):
        if isinstance(values, str):
            values = (values,)
        dims = tuple(values)
        if len(dims) != self.ndim:
            raise ValueError('dimensions %s must have the same length as the '
                             'number of data dimensions, ndim=%s'
                             % (dims, self.ndim))
        self._dims = dims

    def swap_dims(self, dims_dict):
        dims = list(self.dims)
        for oldname, newname in dims_dict.items():
            dims = tuple(newname if x == oldname else x for x in dims)
        self.dims = dims

    @property
    def attrs(self):
        """Dictionary of attributes on this variable.
        """
        if not hasattr(self, '_attrs'):
            self._attrs = Attributes()
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        # set default values for standard variable attributes
        attr = Attributes()
        attr['quantity'] = com.get_fill(str)
        attr['unit'] = com.get_fill(str)
        attr['sampling'] = com.get_fill(str)

        # only for links
        attr['side'] = com.get_fill(str)

        if value is not None:
            for key, value in value.items():
                attr[key] = value
        self._attrs = attr

    def setattrs(self, **kwargs):
        for key, value in kwargs.items():
            self.attrs[key] = value

    def delattrs(self, *args):
        for key in args:
            if key in self.attrs:
                del self.attrs[key]

    def getattrs(self, key):
        param = self.attrs[key]
        return type(self)(param, (), name=self.name)

    def assign_attr(self, **kwargs):
        for key, value in kwargs.items():
            self.attrs[key] = value

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def size(self):
        return np.prod(self.shape)

    def __len__(self):
        try:
            return self._values.shape[0]
        except IndexError:
            return 1

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def chunks(self):
        return self._values.chunks

    @property
    def numblocks(self):
        return self._values.numblocks

    @property
    def npartitions(self):
        return self._values.npartitions

    def rechunk(self, chunks):
        if isinstance(chunks, dict):
            nchunks = {}
            for k, v in chunks.items():
                nkey = get_axis(self.dims, k)
                if len(nkey) == 1:
                    nchunks[nkey[0]] = v
            chunks = nchunks
            chunksize = self.chunksize
        else:
            chunksize = chunks
        values = self._values.rechunk(chunks)
        return self._new(values, self.dims, self.name, self.attrs,
                         chunksize=chunksize)

    def astype(self, dtype, **kwargs):
        values = self._values.astype(dtype, **kwargs)
        return self._new(values, self.dims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def conform(self):
        return units.base_SI(self)

    def meta(self, key, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        if key == 'dtype':
            return any([self.dtype == x for x in values])
        elif key == 'name':
            return any([self.name == x for x in values])
        elif key == 'dims':
            return any([x in self.dims for x in values])
        else:
            return (key in self.attrs and
                    any([self.attrs[key] == x for x in values]))

    def __bool__(self):
        return bool(len(self._values))

    def __float__(self):
        return float(self.values)

    def __int__(self):
        return int(self.values)

    def __complex__(self):
        return complex(self.values)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self):
        values = pri.get_printvals(self._values, self.size, self.shape)

        if self.name is not None:
            name_str = '%r ' % self.name
        else:
            name_str = ''
        dim_summary = ', '.join('%s: %s' % (k, v)
                                for k, v in zip(self.dims, self.shape))
        summary = ['<%s %s(%s)>' %
                   (type(self).__name__, name_str, dim_summary)]
        summary.append(pri.format_var(values.compute()))
        summary.append(repr(self.attrs))
        return '\n'.join(summary)

    def get_printvals(self):
        return pri.get_printvals(self._values, self.size, self.shape)

    def pformat(self, values, col_width=None):
        if self.name in self.dims:
            name = '    %s ' % self.name
            dims_str = '*%s* ' % len(self)
        else:
            name = '    %s ' % self.name
            if self.dims:
                dims_str = '(%s) ' % ', '.join(map(str, self.dims))
            else:
                dims_str = ''

        if not col_width:
            col_width = max(len(self.name), 7) + 6
        first_col = pri.pprint(name, maxlen=col_width)
        front_str = '\n' + first_col + dims_str + ('%s ' % self.dtype)
        max_width = 80 - len(front_str)
        return front_str + pri.format_var(values, maxlen=max_width)

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        for i in range(len(self)):
            yield self[i]

    def get_axis_num(self, dim):
        """
        Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        return get_axis(self.dims, dim)

    def to_variable(self):
        """Return this variable as a base Variable"""
        return Variable(self._values, self.dims, self.name, self._attrs,
                        chunksize=self.chunksize)

    def to_index(self):
        """Return this variable as a Coordinate"""
        return Index(self._values, self.name, attrs=self._attrs)

    def __getitem__(self, keys):
        if isinstance(keys, Variable) and keys.ndim > 1 and keys.dtype == bool:
            return self.mask(~keys)

        keys = self._parse_keys(keys)
        if all(isinstance(key, slice) and key == slice(None) for key in keys):
            return self

        values = select(self._values, keys)
        dims = tuple(idim for ishape, idim in zip(values.shape, self.dims)
                     if ishape > 1)
        values = da.squeeze(values)
        return self._new(values, dims, self._name, self._attrs,
                         chunksize=self.chunksize)

    def __setitem__(self, keys, value):
        keys = self._parse_keys(keys)
        if not hasattr(value, 'chunks'):
            value = da.from_array(np.asarray(value), chunks=self.chunksize)
        if value.ndim == 0:
            value = value.reshape(1)
        self._values = assign(self._values, keys, value)

    def __delitem__(self, keys):
        keys = self._parse_keys(keys)
        self.idrop(keys)

    def _parse_keys(self, keys):
        """Given a key for orthogonal array indexing, returns an equivalent key
        suitable for indexing a numpy.ndarray with fancy indexing.
        """
        if isinstance(keys, Variable) and keys.ndim == self.ndim:
            keys = keys.transpose(*self.dims)
            return keys.values

        # convert to tuples
        if com.is_dict_like(keys):
            keys = tuple(keys.get(dim, slice(None)) for dim in self.dims)
        if not isinstance(keys, tuple):
            keys = (keys,)

        # expand to full number of dimensions
        outkeys = []
        found_ellipsis = False
        for ikey in keys:
            if ikey is Ellipsis:
                if not found_ellipsis:
                    outkeys.extend((self.ndim + 1 - len(keys)) * [slice(None)])
                    found_ellipsis = True
                else:
                    outkeys.append(slice(None))
            else:
                outkeys.append(ikey)

        if len(outkeys) > self.ndim:
            raise IndexError('too many indices')

        outkeys.extend((self.ndim - len(outkeys)) * [slice(None)])
        keys = outkeys

        # convert from orthogonal to grid-based indexing
        if any(not isinstance(ikey, slice) for ikey in keys):
            outkeys = []
            for ikey in keys:
                if isinstance(ikey, slice):
                    outkeys.append(ikey)
                else:
                    if not hasattr(ikey, 'chunks'):
                        ikey = da.from_array(np.asarray(ikey),
                                             chunks=self.chunksize)
                    if ikey.ndim > 1 or ikey.dtype.kind not in ('i', 'b'):
                        raise ValueError(ikey)
                    if ikey.ndim == 0:
                        ikey = ikey.reshape(1)
                    outkeys.append(ikey)
        keys = outkeys
        return tuple(keys)

    def isel(self, **indexers):
        if isinstance(indexers, Variable) and indexers.ndim > 1 and indexers.dtype == bool:
            return self.mask(~indexers)

        return self.apply_index(indexers)

    def apply_index(self, indexers):
        key = [slice(None)] * self.ndim
        for i, dim in enumerate(self.dims):
            if dim in indexers:
                key[i] = indexers[dim]

        assign_fr = [slice(None)] * self.ndim
        assign_to = [slice(None)] * self.ndim
        chunks = list(self.chunks)
        shape_to = list(self.shape)
        for i, iloc in enumerate(key):
            if isinstance(iloc, tuple):
                assign_fr[i] = iloc[0]
                assign_to[i] = iloc[1]
                shape_to[i] = iloc[2]
                chunks[i] = iloc[1].chunks[0]
            else:
                assign_fr[i] = iloc
        assign_fr = tuple(assign_fr)
        assign_to = tuple(assign_to)
        shape_to = tuple(shape_to)

        if not all(isinstance(x, slice) and x == slice(None) for x in assign_to):
            newvals = da.full(shape_to, com.get_fill(self.dtype),
                              dtype=self.dtype, chunks=chunks)

            ival = select(self._values, assign_fr)
            newvals = assign(newvals, assign_to, ival)
            dims = tuple(idim for ishape, idim in zip(newvals.shape, self.dims)
                         if ishape > 1)
            newvals = da.squeeze(newvals)
            return self._new(newvals, dims, self.name, self.attrs,
                             chunksize=self.chunksize)
        elif not all(isinstance(x, slice) and x == slice(None) for x in assign_fr):
            newvals = select(self._values, assign_fr)
            dims = tuple(idim for ishape, idim in zip(newvals.shape, self.dims)
                         if ishape > 1)
            newvals = da.squeeze(newvals)
            return self._new(newvals, dims, self.name, self.attrs,
                             chunksize=self.chunksize)
        else:
            return self.copy()

    def idrop(self, **indexers):
        key = [slice(None)] * self.ndim
        for i, dim in enumerate(self.dims):
            if dim in indexers:
                mask = np.arange(self.shape[i])
                if isinstance(indexers[dim], slice):
                    key[i] = np.delete(mask, indexers[dim])
                else:
                    key[i] = np.delete(mask, [indexers[dim]])
        return self[tuple(key)]

    def append(self, other):
        oval = other._values
        if oval.shape == ():
            oval = oval.reshape(1)
        newval = da.concatenate([self._values, oval])
        return self._new(newval, self.dims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def mask(self, mask):
        newdims = _unified_dims((self, mask))
        invals = self._broadcast_compat((self, mask), newdims)
        shape = np.maximum(invals[0].shape, invals[1].shape)
        invals = tuple(da.broadcast_to(ival, shape) for ival in invals)
        newval = assign(invals[0], invals[1], com.get_fill(self.dtype))
        return self._new(newval, newdims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def extend(self, index, how=None, dim=None):
        n = len(index)
        dtype = self.dtype

        newdims = self.dims + (index.name,)
        newshape = self.shape + (n,)
        if how == 'ffill' or how == 'bfill':
            values = da.broadcast_to(self._values[..., None], newshape)
        elif how == 'linear':
            newdims = list(self.dims).remove(dim)
            newdims = tuple(newdims) + (dim,)
            newvar = self.transpose(*newdims)
            values = np.empty(newshape, dtype=dtype)
            for i in range(n):
                values[..., i] = ((i/n) * (values[..., 1:, 0] -
                                           values[..., :-1, 0]))
        elif how is None:
            values = da.full(newshape, com.get_fill(dtype), dtype=dtype,
                             chunks=self.chunks)
            values[..., 0] = self._values

        newvar = self._new(values, newdims, self.name, self.attrs,
                           chunksize=self.chunksize)
        return newvar

    def ortho(self, nstep, mstep, dim, dim2, rename=False, aligned=0):
        newdims = list(self.dims)
        newdims.remove(dim)
        newdims = tuple(newdims) + (dim,)
        newvar = self.transpose(*newdims)
        newdims += (dim2,)

        values = reshape(newvar._values, nstep, mstep, self.chunksize, aligned)
        newvar = self._new(values, newdims, self.name, self.attrs,
                           chunksize=self.chunksize)
        if rename:
            newvar.attrs['sampling'] = 'hi_res'
        return newvar

    def inline(self, nstep, dim, dim2, rename=False):
        newdims = list(self.dims)
        newdims.remove(dim)
        newdims.remove(dim2)
        newdims = tuple(newdims) + (dim, dim2)
        newvar = self.transpose(*newdims)

        values = newvar._values
        shape = newvar.shape
        newshape = shape[:-2] + (nstep,)
        values = values.reshape(newshape).rechunk(self.chunksize)
        newdims = newvar.dims[:-1]

        newvar = self._new(values, newdims, self.name, self.attrs,
                           chunksize=self.chunksize)
        return newvar

    def split(self, dim, n):
        if dim not in self.dims:
            return [self for i in range(n)]

        axis = self.get_axis_num(dim)[0]
        assert n == self.shape[axis]

        values = self._values
        dims = list(self.dims)
        dims.remove(dim)
        dims = tuple(dims)

        varlist = []
        if self.ndim == 1:
            for i in range(n):
                newvar = self._new(values[i], dims, self._name, self._attrs,
                                   chunksize=self.chunksize)
                varlist.append(newvar)
        else:
            keys = [slice(None)] * self.ndim
            for i in range(n):
                keys[axis] = i
                newval = values[tuple(keys)]
                newvar = self._new(newval, dims, self._name, self._attrs,
                                   chunksize=self.chunksize)
                varlist.append(newvar)

        if dim+'_group' in self.dims:
            for i in range(n):
                varlist[i].swap_dims({dim+'_group': dim})
        return varlist

    def assign(self, values):
        newvar = object.__new__(type(self))
        newvar._values = values
        newvar.dims = self.dims
        newvar.name = self.name
        newvar.attrs = self.attrs
        return newvar

    def copy(self, deep=False):
        """Returns a copy of this object.
        """
        values = self._values
        return self._new(values, self.dims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    @property
    def T(self):
        return self.transpose()

    def transpose(self, *dims):
        """
        rearange order of dimensions in internal array
        """
        if len(dims) == 0:
            dims = self.dims[::-1]
        if dims == self.dims:
            return self
        axes = self.get_axis_num(dims)
        values = self._values.transpose(axes)
        dims = tuple(self.dims[axis] for axis in axes)
        return self._new(values, dims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def squeeze(self, dim=None):
        """
        """
        dims = dict(zip(self.dims, self.shape))
        if dim is None:
            dim = [d for d, s in dims.items() if s == 1]
        else:
            if isinstance(dim, str):
                dim = [dim]
            if any(dims[k] > 1 for k in dim):
                raise ValueError('cannot select a dimension to squeeze out '
                                 'which has length greater than one')
        return self.isel(**dict((d, 0) for d in dim))

    def _broadcast(self, args):
        newdims = _unified_dims(args)
        return self._broadcast_compat(args, newdims)

    @staticmethod
    def _make_op(method):
        def func(self, *args, **kwargs):
            return self._invoke_method(method, (self, *args), method, **kwargs)
        return func

    def __numpy_ufunc__(self, ufunc, method, i, args, **kwargs):
        return self._invoke_method(method, args, ufunc.__name__, obj=ufunc,
                                   **kwargs)

    def _invoke_method(self, method, args, name, obj=None,
                       rename=False, **kwargs):
        if not all(isinstance(x, self.comp_tup) for x in args):
            return NotImplemented

        newattrs = Attributes.merged(list(args))
        if 'unit' not in newattrs:
            newattrs['unit'] = 'NA'
        if 'quantity' not in newattrs:
            newattrs['quantity'] = 'NA'

        newdims = _unified_dims(args)
        args = self._broadcast_compat(args, newdims)
        n = len(args)
        chunkargs = (x for x in args if hasattr(x, 'chunks') and x.ndim > 0)
        chunks = (arg.chunks for arg in chunkargs)
        newchunks = tuple(max(x) for x in zip(*tuple(chunks)))

        # take care of computations that change the dimensions of result array
        drop_dim = kwargs.pop('drop_dim', None)
        new_dim = kwargs.pop('new_dim', None)
        if drop_dim is not None:
            drop_axis = get_axis(newdims, drop_dim)
            newdims = tuple(x for x in newdims if x not in drop_dim)
            newchunks = tuple(x for i, x in enumerate(newchunks)
                              if i not in drop_axis)
            kwargs['chunks'] = newchunks
            kwargs['drop_axis'] = drop_axis

        if new_dim is not None:
            if drop_dim is not None and len(drop_dim) == len(new_dim):
                del kwargs['drop_axis']
            else:
                kwargs['new_axis'] = range(len(newdims), len(newdims)+len(new_dim))
            newchunks += kwargs.pop('new_chunks')
            newdims += new_dim
            kwargs['chunks'] = newchunks

        # for aggregation functions
        dim = kwargs.pop('dim', None)
        if dim is not None:
            axes = get_axis(newdims, dim)
            if not axes and n == 1:
                return self
            kwargs['axis'] = axes[0] if len(axes) == 1 else axes

        window = kwargs.get('window')
        if window is not None:
            newwindow = {}
            for k, v in window.items():
                newwindow[get_axis(newdims, k)[0]] = v
            kwargs['window'] = newwindow

        if obj is None:
            obj, args = args[0], args[1:]
        newval = getattr(obj, method)(*args, **kwargs)
        newname = self.name
        if rename:
            if n == 1:
                newname += '_' + name
            for arg in args:
                if hasattr(arg, 'name') and arg.name is not None:
                    othername = arg.name
                else:
                    othername = 'value'
                newname += '_' + name + '_' + othername

        if isinstance(newval, tuple):
            return tuple(self._wrap_output(x, newdims, dim, name, newname,
                                           newattrs)
                         for x in newval)
        return self._wrap_output(newval, newdims, dim, name, newname, newattrs)

    def _broadcast_compat(self, inputs, dims):
        newvals = ()
        for var in inputs:
            if not hasattr(var, 'dims'):
                if hasattr(var, 'pars'):
                    newval = var.pars
                else:
                    newval = var
            elif var.dims == dims:
                newval = var._values
            elif var.ndim == 0:
                newval = var._values
            else:
                newdims = tuple(d for d in dims if d not in var.dims)
                newdims += var.dims
                axes = [newdims.index(idim) for idim in dims]
                axes += [i for i, x in enumerate(newdims) if x not in dims]

                ival = var._values[(None,) * (len(newdims) - var.ndim)]
                newval = da.transpose(ival, axes=axes)
            newvals += (newval,)
        return newvals

    def _wrap_output(self, outval, outdims, dim, name, newname, outattrs):
        if not hasattr(outval, 'chunks'):
            return MetaVar(pars=outval, attrs=outattrs, name=newname)

        # adapt dimensions
        if len(outval.shape) == 0:
            outdims = ()
        elif dim is not None and outval.ndim < len(outdims):
            if isinstance(dim, str):
                dim = (dim,)
            outdims = tuple(idim for idim in outdims if idim not in dim)
        elif dim is None and outval.size == 1:
            outdims = ()
        elif outval.ndim > len(outdims):
            for i in range(outval.ndim-len(outdims)):
                j = i
                dimstr = 'dim'+str(j)
                while dimstr in outdims:
                    dimstr = 'dim'+str(j)
                    j += 1
                outdims += (dimstr,)

        # adapt other metadata
        if len(outdims) < len(self.dims):
            outattrs['sampling'] = name

        outname = newname # TODO: change name
        return self._new(outval, outdims, outname, outattrs,
                         chunksize=self.chunksize)

    def compute(self):
        """
        evaluate the contained task-graph and store the results in memory.
        """
        val = self._values
        outval = da.compute(val)
        return Variable(outval, self.dims, self.name, self.attrs,
                        chunksize=self.chunksize)

    def equals(self, other):
        """True if two Variables have the same dimensions and values;
        otherwise False.

        Variables can still be equal if they have NaN values in the same
        locations.

        This method is necessary because `v1 == v2` for Variables
        does element-wise comparisions (like numpy.ndarrays).
        """
        other = getattr(other, 'variable', other)
        try:
            return (self.dims == other.dims and
                    com.array_equal(self._values, other._values))
        except (TypeError, AttributeError):
            return False

    def expand_dims(self, dims, shape=None):
        """Return a new variable with expanded dimensions.

        When possible, this operation does not copy this variable's data.

        Parameters
        ----------
        dims : str or sequence of str or dict
            Dimensions to include on the new variable. If a dict, values are
            used to provide the sizes of new dimensions; otherwise, new
            dimensions are inserted with length 1.

        Returns
        -------
        Variable
        """
        if isinstance(dims, str):
            dims = [dims]

        if shape is None and com.is_dict_like(dims):
            shape = dims.values()

        missing_dims = set(self.dims) - set(dims)
        if missing_dims:
            raise ValueError('new dimensions must be a superset of existing '
                             'dimensions')

        self_dims = set(self.dims)
        expanded_dims = tuple(d for d in dims if d not in self_dims)
        expanded_dims = expanded_dims + self.dims
        if shape is not None:
            dims_map = dict(zip(dims, shape))
            tmp_shape = [dims_map[d] for d in expanded_dims]
            expanded_dat = da.broadcast_to(self._values, tmp_shape)
        else:
            expansion = (None,) * (len(expanded_dims) - self.ndim)
            expanded_dat = self._values[expansion]
        expanded_var = Variable(expanded_dat, expanded_dims, attrs=self._attrs)
        return expanded_var.transpose(*dims)

    def identical(self, other):
        """Like equals, but also checks attributes.
        """
        try:
            return self.attrs.equivalent(other.attrs) and self.equals(other)
        except (TypeError, AttributeError):
            return False

    def isnull(self):
        newval = com.isnull(self._values)
        return self._new(newval, self.dims, self.name, self.attrs,
                         chunksize=self.chunksize)

    def _create_store(self, outh, coord=False):
        source, attrs = self._convert_for_store()
        if self.ndim == 0:
            target = outh.create_dataset(self.name, shape=source.shape,
                                dtype=source.dtype)
        else:
            chunks = [x[0] for x in source.chunks]
            bytesize = source.dtype.itemsize * np.prod(chunks)
            if bytesize >= 4e9:
                imax, vmax = max(enumerate(chunks), key=lambda x: x[1])
                chunks[imax] = int(np.floor((4e9 / bytesize) * vmax))

            chunks = tuple(chunks)
            target = outh.create_dataset(self.name, shape=source.shape,
                                         dtype=source.dtype, chunks=chunks,
                                         compression='gzip', fletcher32=True)
        attrs.to_store(target.attrs)
        if len(self.dims) and self.name == self.dims[0] or coord == True:
            target.dims.create_scale(target, self.name)
            target.dims[0].label = self.dims[0]
        else:
            for i, idim in enumerate(self.dims):
                target.dims[i].label = idim
                target.dims[i].attach_scale(outh[idim])
        return source, target

    def store(self, outpath):
        io.to_hdf(self, outpath)

    def _convert_for_store(self):
        attrs = self.attrs
        if hasattr(self, '_step'):
            attrs['resolution'] = self.step
        if hasattr(self, '_start'):
            attrs['start'] = self.start
        if hasattr(self, '_stop'):
            attrs['stop'] = self.stop

        values = self._values
        if values.dtype.kind == 'U':
            length = str(values.dtype)[2:]
            values = values.astype('S'+length)
        elif values.dtype.kind == 'M':
            values = com.dt64_to_posix(values)
            attrs['unit'] = 's'
            attrs['epoch'] = '1970-01-01 00:00'
        elif values.dtype.kind == 'm':
            values = values/np.timedelta64(1, 's')
            attrs['unit'] = 's'
        attrs['array_id'] = self.name
        return values, attrs

    @classmethod
    def merged(cls, datlist, slicers, length, xdim=None,
               chunksize=com.CHUNKSIZE):
        datlist = com.prep_merger(datlist)
        i0, dat0 = next(iter(datlist.items()))
        if all(x._values.name == dat0._values.name for x in datlist.values()):
            if all(len(slicers[x]) <= 1 for x in dat0.dims):
                return dat0

        attrs = Attributes.merged(datlist)
        dims = dat0.dims
        if xdim is not None:
            dims = dims + (xdim,)
            dats = []
            for i, idat in datlist.items():
                # determine alignment
                slicer = []
                idat = idat.transpose(*dims)
                for dID in idat.dims:
                    slicer.append(slicers[dID][i])
                slicer = tuple(slicer)
                dats.append(idat[slicer])
            values = da.stack(dats, axis=-1)
            return cls(values, dims, name=dat0.name, attrs=attrs,
                       chunksize=chunksize)

        if len(datlist) == 1:
            shape = [length[dID] for dID in dims]
            null = com.get_fill(dat0.dtype)
            values = da.full(shape, null, dtype=dat0.dtype, chunks=chunksize)

            # determine alignment
            slicer = []
            for dID in dat0.dims:
                slicer.append(slicers[dID][i0])
            slicer = tuple(slicer)

            # fill in values
            values = assign(values, slicer, dat0._values)
            return cls(values, dims, name=dat0.name, attrs=attrs,
                       chunksize=chunksize)

        if len([x for x in [list(x.values())[0] for x in slicers.values()] if x is not slice(None)]) == 1:
            slices = [list(x.values()) for x in slicers.values()][0]
            if all(slices[i].start == slices[i-1].stop for i in range(1, len(slices))):
                values = da.concatenate([idat._values for idat in datlist.values()])
                return cls(values, dims, name=dat0.name, attrs=attrs,
                           chunksize=chunksize)

        shape = [length[dID] for dID in dims]
        null = com.get_fill(dat0.dtype)
        values = da.full(shape, null, dtype=dat0.dtype, chunks=chunksize)
        for i, idat in datlist.items():
            # determine alignment
            slicer = []
            idat = idat.transpose(*dims)
            for dID in idat.dims:
                slicer.append(slicers[dID][i])
            slicer = tuple(slicer)

            # fill in values
            o_val = values[slicer]
            n_val = da.where(com.isnull(o_val), idat._values, o_val)
            n_val = n_val.rechunk(chunksize)
            values = assign(values, slicer, n_val)
        return cls(values, dims, name=dat0.name, attrs=attrs,
                   chunksize=chunksize)


Variable.comp_tup = (Variable, da.Array, np.ndarray, list, tuple, int, float,
                     complex, bool, str, np.datetime64, np.timedelta64,
                     np.float32, np.int32,
                     FunctionType)


def inject_ops(cls):
    """
    Helper function to add universal function methods to variables.
    """
    other_methods = {'item', 'where', 'argsort', 'searchsorted'}
    simple_ops = {'lt', 'le', 'ge', 'gt', 'eq', 'ne', 'neg', 'pos', 'abs',
                  'invert'}
    num_binary_ops = {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                      'pow', 'and', 'xor', 'or', 'matmul'}
    unary_methods = {'clip', 'conj', 'real', 'imag', 'all', 'any'}
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

    for name in unary_methods | nan_reduce_methods:
        setattr(cls, name, cls._make_op(name))

    for name in nan_reduce_methods:
        name = 'nan'+name
        setattr(cls, name, cls._make_op(name))
inject_ops(Variable)


def as_variable(obj, dims=None, name=None, attrs=None):
    if not (hasattr(obj, 'dims') and
            hasattr(obj, 'name') and
            hasattr(obj, 'attrs')):
        if isinstance(obj, tuple):
            obj = Variable(*obj)
        elif dims is not None:
            obj = Variable(obj, dims)
        else:
            raise TypeError

    if dims is not None:
        obj.dims = dims
    if name is not None:
        obj.name = name
    if attrs is not None:
        obj.attrs = attrs
    return obj


def get_axis(dims, dim):
    if isinstance(dim, str):
        dim = (dim,)
    axes = []
    for idim in dim:
        try:
            axes.append(dims.index(idim))
        except ValueError:
            pass
    return tuple(axes)


def _unified_dims(variables):
    if len(variables) == 1:
        return variables[0].dims
    all_dims = OrderedDict()
    for var in variables:
        try:
            var_dims = var.dims
        except AttributeError:
            continue
        if len(set(var_dims)) < len(var_dims):
            raise ValueError('broadcasting cannot handle duplicate '
                             'dimensions: %r' % list(var_dims))
        for idim, ishape in zip(var_dims, var.shape):
            if idim not in all_dims:
                all_dims[idim] = ishape
            elif all_dims[idim] != ishape:
                raise ValueError('operands cannot be broadcast together '
                                 'with mismatched lengths for dimension %r: %s'
                                 % (idim, (all_dims[idim], ishape)))
    return tuple(all_dims)


###############################################################################
class Index(Variable):
    """
    Searchable monotonically increasing Variable.
    """
    def __init__(self, values, name, attrs=None, start=None, stop=None,
                 step=None, **kwargs):
        super().__init__(values, name, name=name, attrs=attrs, **kwargs)
        self.step = step
        if start is None:
            self.start = values[0]
        else:
            self.start=start
        if stop is None:
            if step is not None and values.dtype.kind not in ('U', 'S'):
                self.stop = values[-1] + step
            else:
                self.stop = values[-1]
        else:
            self.stop = stop
        if self.ndim != 1:
            raise ValueError('%s objects must be 1-dimensional' %
                             type(self).__name__)

    @classmethod
    def _new(cls, values, dims, name, attrs, start=None, stop=None, step=None,
             chunksize=com.CHUNKSIZE):
        """
        shortcut to skip data checking and conformation on creation
        """
        newvar = object.__new__(cls)
        newvar._values = values
        newvar._dims = dims
        newvar._name = name
        newvar._attrs = attrs
        if start is not None:
            newvar.start = start
        if stop is not None:
            newvar.stop = stop
        newvar._step = step
        newvar.chunksize = chunksize
        return newvar

    @property
    def start(self):
        if not hasattr(self, '_start'):
            #print('warning! deriving start value: %s' % self.name)
            self._start = self._values[0].compute()
        return self._start

    @start.setter
    def start(self, value):
        if hasattr(value, 'compute'):
            value = value.compute()
        self._start = value

    @property
    def stop(self):
        if not hasattr(self, '_stop'):
            #print('warning! deriving stop value: %s' % self.name)
            if self.step == None:
                self._stop = self._values[-1].compute()
            else:
                self._stop = self._values[-1].compute() + self.step
        return self._stop

    @stop.setter
    def stop(self, value):
        if hasattr(value, 'compute'):
            value = value.compute()
        self._stop = value

    @property
    def step(self):
        if not hasattr(self, '_step'):
            print('warning! no resolution defined')
            values = self._values
            if self.dtype.kind in ('S', 'U'):
                self._step = None
            elif len(self) <= self.chunks[0][0]:
                diff = values[1:] - values[:-1]
                if values.dtype.kind == 'f':
                    cond = np.all(np.isclose(diff, diff[0]))
                else:
                    diff = values[1:] - values[:-1]
                    cond = np.all(diff == diff[0])

                if cond:
                    self._step = diff[0].compute()
                else:
                    self._step = None
            else:
                self._step = (values[1] - values[0]).compute()
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def astype(self, dtype, **kwargs):
        values = self._values.astype(dtype, **kwargs)
        return self._new(values, self.dims, self.name, self.attrs, self.start,
                         self.stop, self.step, self.chunksize)

    def rechunk(self, chunks):
        if isinstance(chunks, dict):
            nchunks = {}
            for k, v in chunks.items():
                nkey = get_axis(self.dims, k)
                if len(nkey) == 1:
                    nchunks[nkey[0]] = v
            chunks = nchunks
            chunksize = self.chunksize
        else:
            chunksize = chunks
        values = self._values.rechunk(chunks)
        return self._new(values, self.dims, self.name, self.attrs, self.start,
                         self.stop, self.step, chunksize=chunksize)

    def append(self, other):
        oval = other._values
        if oval.shape == ():
            oval = oval.reshape(1)
        newval = da.concatenate([self._values, oval])
        return self._new(newval, self.dims, self.name, self.attrs, self.start,
                         other.stop, self.step, self.chunksize)

    def copy(self, deep=False):
        """Returns a copy of this object.
        """
        values = self._values
        return self._new(values, self.dims, self.name, self.attrs, self.start,
                         self.stop, self.step, self.chunksize)

    def __getitem__(self, keys):
        if isinstance(keys, Variable) and keys.ndim > 1 and keys.dtype == bool:
            return self.mask(~keys)

        keys = self._parse_keys(keys)
        if all(isinstance(key, slice) and key == slice(None) for key in keys):
            return self
        if self._step is not None:
            newindex = self.selectregular(keys[0])
        else:
            values = select(self._values, keys)
            newindex = self._new(values, self._dims, self._name, self._attrs,
                                 self.start, self.stop, self.step,
                                 self.chunksize)
        if newindex.shape[0] == 1:
            newindex.step = None
        return newindex

    def isel(self, **indexers):
        #keys = self._parse_keys(keys)
        #if all(isinstance(key, slice) and key == slice(None) for key in indexers):
        #    return self
        if (isinstance(indexers, Variable) and
            indexers.ndim > 1 and
            indexers.dtype == bool):
            return self.mask(~indexers)

        newindex = self.apply_index(indexers)
        newindex.step = self.step
#        if self._step is not None:
#            newindex = self.selectregular(keys[0])
#        else:
#            values = select(self._values, keys)
#            newindex = self._new(values, self._dims, self._name, self._attrs,
#                                 self.start, self.stop, self.step,
#                                 self.chunksize)

        #if newindex.shape[0] == 1:
        #    newindex.step = None
        return newindex

    def selectregular(self, loc):
        if isinstance(loc, slice):
            if loc.start is None:
                start = self.start
            else:
                if loc.start >= 0:
                    start = self.start + self.step * loc.start
                else:
                    start = max(self.stop + self.step * loc.start, self.start)
            if loc.stop is None:
                stop = self.stop
            else:
                if loc.stop >= 0:
                    stop = min(self.start + self.step * loc.stop, self.stop)
                else:
                    stop = self.stop + self.step * loc.stop
            if loc.step is None:
                step = self.step
            else:
                step = self.step * loc.step
            return carange(start, stop, step, dtype=self.dtype, name=self._name,
                           attrs=self._attrs)
        else:
            values = da.where(loc >= 0, self.start + self.step * loc,
                              self.stop + self.step * loc)
            return self._new(values, self._dims, self._name, self._attrs,
                             self.chunksize)

    def __setitem__(self, key, value):
        raise TypeError('%s values cannot be modified' % type(self).__name__)

    def to_index(self):
        return self

    def _parse_label(self, label):
        if hasattr(label, '_values'):
            label = label._values
        elif isinstance(label, slice):
            start = com.to_nptype(label.start)
            stop = com.to_nptype(label.stop)
            step = com.to_nptype(label.step)
            if self.dtype.kind == 'M':
                if start is None:
                    start = None
                else:
                    start = start.astype('<M8[us]')
                if stop is None:
                    stop = None
                else:
                    stop = stop.astype('<M8[us]')
                if step is None:
                    step = None
                else:
                    step = com.str_to_td64(step)
            label = slice(start, stop, step)
        else:
            label = com.to_nptype(label)
            if self.dtype.kind == 'M' and label.size <= 1:
                label = label.astype(np.datetime64)
                label = com.dt64_to_slice(label)
            else:
                if label.ndim == 0:
                    label = label.reshape(1)
                label = da.from_array(label, chunks=self.chunksize)
                if self.dtype.kind == 'M':
                    label = label.astype(np.datetime64)

        return label

    def searchsorted(self, label, **kwargs):
        if self.step is None:
            label = da.from_array(label, self.chunksize)
            if self._values.dtype.kind in ('U', 'S'):
                values = self._values.astype(object)
            else:
                values = self._values

            return indexing.searchdask(values, label, **kwargs)

        return indexing.searchdaskuniform(self.start, self.step, self.shape[0],
                                          label, **kwargs)

    def get_indexer(self, label, **kwargs):
        label = self._parse_label(label)

        if isinstance(label, slice):
            if label.start is None or label.start <= self.start:
                start = None
            else:
                start = int(self.searchsorted(label.start, how='bfill'))
            if label.stop is None or label.stop >= self.stop:
                stop = None
            else:
                stop = int(self.searchsorted(label.stop, how='bfill'))
            if label.step is None:
                step = None
            else:
                step = label.step / self.step
            fr_indxr = slice(start, stop, step)
        elif label.dtype.kind == 'b':
            fr_indxr = self._values[label].nonzero()[0]
        else:
            if label.ndim == 0:
                label = label.reshape(1)

            if label.dtype.kind in ('U', 'S'):
                label = label.astype(object)

            fr_indxr = self.searchsorted(label, **kwargs)
            cond = fr_indxr != self.shape[0]
            fr_indxr = select(fr_indxr, cond)

            to_shape = len(label)
            to_indxr = creation.arange(to_shape, chunks=self.chunksize)
            to_indxr = select(to_indxr, cond)
            return fr_indxr, to_indxr, to_shape

        return fr_indxr

    @classmethod
    def merged(cls, datlist, regular=True, sort=True, **kwargs):
        datlist = com.prep_merger(datlist)
        dat0 = next(iter(datlist.values()))
        if all(x._values.name == dat0._values.name for x in datlist.values()):
            slicer = OrderedDict({i: slice(None) for i in datlist.keys()})
            return dat0, slicer

        attrs = Attributes.merged(datlist)
#        if dat0.dtype.kind in ('S', 'U'):
#            arrays = [idat._values for idat in datlist.values()]
#            array, index = np.unique(np.concatenate(arrays), return_index=True)
#            array = array[index.argsort()]
#            slicer = OrderedDict()
#            for i, idat in datlist.items():
#                ival = idat._values
#                x0 = np.where(array == ival[0])[0][0]
#                x1 = np.where(array == ival[-1])[0][0] + 1
#                slicer[i] = slice(x0, x1)
#            return Index(array, dat0.name, attrs=attrs), slicer

        if dat0.step is not None:
            return cls.merge_regular(datlist, attrs, **kwargs)
        else:
            if dat0.dtype.kind in ('U', 'S'):
                dtype = dat0.dtype
                datlist = OrderedDict({k:v.astype(object) for k, v in datlist.items()})

            def sortfunc(item):
                return item[1].start
            if sort:
                datlist = sorted(datlist.items(), key=sortfunc)
            else:
                datlist = list(datlist.items())
            newdat, slicer = cls.merge_irregular(datlist, attrs, **kwargs)

            if dat0.dtype.kind in ('U', 'S'):
                newdat = newdat.astype(dtype)
            return newdat, slicer

    @classmethod
    def merge_irregular(cls, datlist, attrs, chunksize=com.CHUNKSIZE):
        i, dat0 = datlist[0]
        searchfunc = indexing.searchirregular

        array = dat0._values
        slicer = OrderedDict({i: slice(0, len(array))})
        end = dat0.stop
        begin = dat0.start
        for i, idat in datlist[1:]:
            ival = idat._values
            istart = idat.start
            istop = idat.stop
            if istop == end and istart == begin:
                x0 = 0
                x1 = len(array)
            elif istop < end:
                x0 = searchfunc(array, istart)
                x1 = (searchfunc(array, istop) + 1)
            else:
                x0 = len(array)
                if istart < end:
                    indexer = searchfunc(ival, end)
                    array = da.concatenate([array, ival[indexer + 1:]])
                    x0 = x0 - indexer - 1
                else:
                    array = da.concatenate([array, ival])
                end = istop
                x1 = len(array)
            slicer[i] = slice(x0, x1)
        array = array.rechunk(chunksize)
        newdat = cls._new(array, dat0.dims, dat0.name, attrs, start=begin,
                          stop=end, chunksize=chunksize)
        return newdat, slicer

    @classmethod
    def merge_regular(cls, datlist, attrs, **kwargs):
        datlist2 = sorted(datlist.items(), key = lambda x: x[1].start)

        dat0 = datlist2[0][1]
        step = dat0.step
        stop = sorted(datlist.items(), key = lambda x: x[1].stop)[-1][1].stop
        start = dat0.start
        slicer = OrderedDict()
        for i, idat in datlist2:
            istop = idat.stop
            istart = idat.start
            x0 = np.floor((istart - start) / step).astype(int)
            x1 = np.ceil((istop - start) / step).astype(int)
            slicer[i] = slice(x0, x1)
            if istop > stop:
                stop = istop

        index = carange(start, stop, step, name=dat0.name, attrs=attrs,
                        **kwargs)
        return index, slicer


def as_index(obj, name=None, attrs=None):
    """
    Convert an object to an Index.
    """
    if not isinstance(obj, Index):
        if isinstance(obj, Variable):
            obj = obj.to_index()
        elif isinstance(obj, tuple):
            obj = Index(*obj)
        elif name is not None:
            obj = Index(obj, name)
        else:
            raise TypeError

    if name is not None:
        obj.name = name
    if attrs is not None:
        obj.attrs = attrs
    return obj


class TimeScale(Index):
    def __init__(self, data, attrs=None, **kwargs):
        timezone = kwargs.get('timezone', 'UTC')
        if attrs is None:
            attrs = Attributes()
        attrs['quantity'] = 'time'
        attrs['timezone'] = timezone
        attrs['unit'] = 'compound date/time object'
        attrs['calendar'] = 'proleptic_gregorian'

        super().__init__(data, 'time', attrs=attrs, **kwargs)


###############################################################################
class MetaVar(Variable):
    """
    A variable containing only attributes.

    parameters:
        attrs:
    """
    treename = 'variable'
    meta = True

    def __init__(self, pars=None, attrs=None, name=None):
        if pars is not None:
            self.pars = pars
        self.name = name
        self.attrs = Attributes()
        if attrs is not None:
            for key, values in attrs.items():
                self.attrs[key] = values

    def setattrs(self, **kwargs):
        for key, value in kwargs.items():
            self.attrs[key] = value

    def delattrs(self, *args):
        for key in args:
            if key in self.attrs:
                del self.attrs[key]

    def setname(self, name, include=None):
        if include is not None:
            attrs = self.attrs
            strings = [str(attrs[i]) for i in include if i in attrs]
            name = '_'.join([name] + strings)
        self.name = name

    def meta(self, key, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        if key in self.attrs:
            if isinstance(self.attrs[key], float):
                return any([np.isclose(self.attrs[key], x) for x in values])
            return self.attrs[key] in values
        return True

    def __repr__(self):
        string = '<MetaVar> ' + str(self.name) + '\n'
        string += repr(self.pars) + '\n'
        string += repr(self.attrs)
        return string

    def get_printvals(self):
        return self.pars

    def pformat(self, values):
        string = '<MetaVar> ' + str(self.name) + '\n'
        string += repr(values) + '\n'
        string += repr(self.attrs)
        return string

    def store(self, outpath):
        io.to_yaml(self, outpath)

    def to_store(self, outh):
        outh['attrs'] = {}
        self.attrs.to_store(outh['attrs'])
        outh['pars'] = com.np_to_base_type(self.pars)

    @classmethod
    def from_yaml(cls, inh, **kwargs):
        data = cls(attrs=Attributes(inh['attrs']), name=inh['array_id'])
        data.pars = tuple(inh['pars'])
        return data


###############################################################################
def zeros(shape, dtype, name=None, attrs=None, chunksize=com.CHUNKSIZE):
    """
    Return a new variable of given shape and type, filled with zeros.
    """
    dims, shape = zip(*tuple(shape.items()))
    values = da.zeros(shape, dtype=dtype, chunks=chunksize)
    return Variable(values, dims, name, attrs)


def ones(shape, dtype, name=None, attrs=None, chunksize=com.CHUNKSIZE):
    """
    Return a new variable of given shape and type, filled with ones.
    """
    dims, shape = zip(*tuple(shape.items()))
    values = da.ones(shape, dtype=dtype, chunks=chunksize)
    return Variable(values, dims, name, attrs)


def full(shape, fill_value, dtype=None, name=None, attrs=None,
         chunksize=com.CHUNKSIZE):
    """
    Return a new array of given shape and type, with with `fill_value`.
    """
    dims, shape = zip(*tuple(shape.items()))
    values = da.full(shape, fill_value, dtype=dtype, chunks=chunksize)
    return Variable(values, dims, name, attrs)


def linspace(start, stop, num=50, name=None, **kwargs):
    """
    Return evenly spaced numbers over a specified interval.
    """
    step = (stop - start) / num
    return carange(start, stop, step, name=name, **kwargs)


def carange(start, stop, step, name=None, attrs=None, dtype=None,
            chunksize=com.CHUNKSIZE):
    """
    Return evenly spaced values within a given interval.
    """
    start = com.to_nptype(start)
    stop = com.to_nptype(stop)
    step = com.to_nptype(step)

    if dtype is None:
        dtype = stop.dtype

    if dtype.kind == 'M':
        if isinstance(step, str):
            step = com.str_to_td64(step)

        start = None if start is None else start.astype('<M8[us]')
        stop = None if stop is None else stop.astype('<M8[us]')
        step = None if step is None else step.astype('<m8[us]')
        args = (start.astype('i8'), stop.astype('i8'), step.astype('i8'))
    elif dtype.kind == 'm':
        if isinstance(step, str):
            step = com.str_to_td64(step)

        start = None if start is None else start.astype('<m8[us]')
        stop = None if stop is None else stop.astype('<m8[us]')
        step = None if step is None else step.astype('<m8[us]')
        args = (start.astype('i8'), stop.astype('i8'), step.astype('i8'))
    else:
        args = (start, stop, step)
    values = creation.arange(*args, chunks=chunksize)
    if dtype.kind == 'M':
        values = values.astype('<M8[us]')
        index = TimeScale(values, attrs=attrs)
        stop = (-(-args[1] // args[2]) * args[2]).astype('<M8[us]')
    elif dtype.kind == 'm':
        values = values.astype('<m8[us]')
        index = Index(values, name, attrs)
        stop = (-(-args[1] // args[2]) * args[2]).astype('<m8[us]')
    else:
        index = Index(values, name, attrs)
        stop = -(-stop // step) * step

    index.start = start
    index.stop = stop
    index.step = step
    return index


def tarange(start, stop, step, attrs=None, chunksize=com.CHUNKSIZE):
    """
    return evenly spaced temporal values within a given interval.
    """
    if isinstance(step, str):
        step = com.str_to_td64(step)

    start = None if start is None else start.astype('<M8[us]')
    stop = None if stop is None else stop.astype('<M8[us]')
    step = None if step is None else step.astype('<m8[us]')

    args = (start.view('i8'), stop.view('i8'), step.view('i8'))
    values = creation.arange(*args, chunks=chunksize)
    values = values.view('<M8[us]')
    index = TimeScale(values, attrs=attrs)
    if step is not None:
        index.step = step

    index.start = start
    index.stop = stop
    return index


def from_store(inh, chunksize=com.CHUNKSIZE):
    """
    Create a Variable from a given hdf5 node handle.
    """
    if inh is None:
        return

    attrs = Attributes.from_store(inh)
    values = da.from_array(inh, chunksize)
    name = attrs.pop('array_id')
    step = attrs.pop('resolution', None)
    start = attrs.pop('start', None)
    stop = attrs.pop('stop', None)

    if name == 'time':
        values = com.posix_to_dt64(values)
        attrs['unit'] = 'datetime64[us]'
    elif name == 'time_offset':
        values = da.around(values * 1000000) * np.timedelta64(1, 'us')
        attrs['unit'] = 'timedelta64[us]'
    if values.dtype.kind == 'S':
        length = str(values.dtype)[2:]
        values = values.astype('U'+length)
    elif values.dtype == bool:
        values = values.astype(np.uint)

    dims = [idim.label for idim in inh.dims]
    if io.isscale(inh) and name == dims[0]:
        out = Index(values, name, attrs=attrs)
    else:
        out = Variable(values, dims, name=name, attrs=attrs)

    if step is not None:
        if name == 'time' or name == 'time_offset':
            out.step = np.timedelta64(int(step), 'us')
        else:
            out.step = step
    if start is not None:
        if name == 'time':
            out.start = np.datetime64(start, 's')
        elif name == 'time_offset':
            out.start = np.timedelta64(int(start), 'us')
        else:
            out.start = start
    if stop is not None:
        if name == 'time':
            out.stop = np.datetime64(stop, 's')
        elif name == 'time_offset':
            out.stop = np.timedelta64(int(stop), 'us')
        else:
            out.stop = stop
    return out
