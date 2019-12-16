# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:41:37 2015

@author: T.C. van Leth
"""
import logging
from collections import OrderedDict

import dask.array as da
from dask.delayed import Delayed
import numpy as np

from phad import common as com
from phad import container_base as cb
from phad import geometry
from phad import inout_common as io
from phad import variable as var
from phad.attributes import Attributes


class Node:
    """
    Recursive hierarchical data container.
    """
    def __init__(self, children=None, name='NA', attrs=None):
        if children is not None:
            self.children = children
        else:
            self.children = OrderedDict()
        self.name = name
        self.attrs = attrs
        self.handles = set()

    @classmethod
    def _new(cls, children, name, attrs, handles):
        newstat = object.__new__(cls)
        newstat.children = children
        newstat.name = name
        newstat.attrs = attrs
        newstat.handles = handles
        return newstat

    @property
    def attrs(self):
        if not hasattr(self, '_attrs'):
            self._attrs = Attributes()
        return self._attrs

    def setattrs(self, **kwargs):
        for child in self.children.values():
            child.setattrs(**kwargs)

    def delattrs(self, *args):
        for child in self.children.values():
            child.delattrs(*args)

    def getattrs(self, key, fallback=None):
        if key in self.attrs:
            fallback = self.attrs[key]

        newchildren = {}
        for cID, child in self.children.items():
            try:
                newchildren[cID] = child.getattrs(key, fallback=fallback)
            except KeyError:
                pass
        if len(newchildren) == 0:
            return var.Variable(self.attrs[key], (), name=key)
        elif fallback is not None:
            for k in self.children.keys():
                if k not in newchildren:
                    newchildren[k] = var.Variable(fallback, (), name=key)
        else:
            return self._new(newchildren, self.name, self.attrs, self.handles)

    def setname(self, name, **kwargs):
        for child in self.children.values():
            child.setname(name, **kwargs)

    def rename(self, name_dict):
        for child in self.children.values():
            child.rename(name_dict)

    def getstep(self, dim):
        newchildren = {}
        for cID, child in self.children.items():
            newchildren[cID] = child.getstep(dim)
        return self._new(newchildren, self.name, self.attrs, self.handles)

    def __len__(self):
        lens = [len(child) for child in self.children.values()]
        if lens:
            return max(lens)
        else:
            return 0

    @property
    def dtype(self):
        dtypes = [child.dtype for child in self.children.values()]
        if dtypes.count(dtypes[0]) == len(dtypes):
            return dtypes[0]
        else:
            return None

    def astype(self, dtype):
        newchildren = {}
        for cID, child in self.children.items():
            newchildren[cID] = child.astype(dtype)
        return self._new(newchildren, self.name, self.attrs, self.handles)

    @property
    def level(self):
        return max(x.level for x in self.children.values()) + 1

    def __bool__(self):
        return any(bool(child) for child in self.children.values())

    def __repr__(self):
        values = self.get_printvals()
        return self.pformat(*da.compute(values))

    def pformat(self, values):
        x = '<%s> %s\n' % (type(self).__name__, self.name)
        x += self._sub.treename + 's:\n'
        for cID, child in self.children.items():
            x += cID + '    ' + com.indent(child.pformat(values[cID]), 2) + '\n'
        x += 'Attributes:\n'
        for aID, iatt in self.attrs.items():
            x += '    ' + aID + ':  ' + str(iatt) + '\n'
        return x

    def get_printvals(self):
        v_short = {}
        for k, v in self.children.items():
            v_short[k] = v.get_printvals()
        return v_short

    def __getattribute__(self, name):
        if name == 'values':
            return self.children.values
        elif name == 'keys':
            return self.children.keys
        elif name == 'items':
            return self.children.items
        else:
            return object.__getattribute__(self, name)

    def __getitem__(self, key):
        if isinstance(key, (cb.Array, Node)) and key.dtype == bool:
            return self.mask(~key)

        if isinstance(key, tuple):
            if len(key) == 1:
                return self[key[0]]
            elif key[0] is Ellipsis:
                if self.level == len(key) - 1:
                    return self[key[1:]]
                elif self.level >= len(key):
                    return self[(slice(None),) + key]

            elif key[0] == slice(None):
                new = type(self)(name=self.name, attrs=self.attrs)
                for cID, child in self.children.items():
                    try:
                        grandchild = child[key[1:]]
                    except KeyError:
                        continue
                    else:
                        if not hasattr(grandchild, 'keys'):
                            grandchild.attrs = Attributes.merged([grandchild, child])
                        new.children[cID] = grandchild
                return new
            elif isinstance(key[0], list):
                newchildren = OrderedDict((k, v[key[1:]])
                                          for k, v in self.children.items()
                                          if k in key[0])
                return self._new(newchildren, self.name, self.attrs,
                                 self.handles)
            else:
                new = self.children[key[0]][key[1:]]
                return new
        elif isinstance(key, list):
            newchildren = OrderedDict((k, v) for k, v in self.children.items()
                                   if k in key)
            return self._new(newchildren, self.name, self.attrs, self.handles)
        else:
            new = self.children[key]
            #new.attrs = Attributes.merged([new, self])
            return new

    def __setitem__(self, key, item):
        if isinstance(key, tuple):
            if len(key) == 1:
                self.children[key[0]] = item
            elif key[0] == slice(None):
                for cID in self.children.keys():
                    self.children[cID][key[1:]] = item
            else:
                if key[0] not in self.children:
                    self.children[key[0]] = self._sub()
                if len(key[1:]) == 1:
                    self.children[key[0]][key[1]] = item
                else:
                    self.children[key[0]][key[1:]] = item
        else:
            self.children[key] = item

    def __delitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 1:
                del self.children[key[0]]
            elif key[0] == slice(None):
                for cID in self.children.keys():
                    del self.children[cID][key[1:]]
            else:
                del self.children[key[0]][key[1:]]
        else:
            del self.children[key]

    def __contains__(self, key):
        return key in self.children

    def _sele(self, **kwargs):
        """
        make a selection based on group attributes
        return a new Station object
        """
        children = self.children
        for key, value in kwargs.items():
            select = {}
            for cID, child in children.items():
                if key in child.attrs:
                    if isinstance(value, list):
                        if any([child.attrs[key] == x for x in value]):
                            select[cID] = child
                    else:
                        if child.attrs[key] == value:
                            select[cID] = child
                else:
                    selchild = child.select(**{key: value})
                    if selchild:
                        select[cID] = selchild
            children = select
        return children

    def select(self, **kwargs):
        """
        make selection based on group attributes
        """
        return type(self)(children=self._sele(**kwargs), name=self.name,
                          attrs=self.attrs)

    def ssel(self, **kwargs):
        """
        select first contained variable based on group attributes.
        """
        selection = list(self._sele(**kwargs).values())
        if len(selection) > 0:
            return selection[0]

    @staticmethod
    def _make_op(method, **kwargs):
        def func(self, *args, **kwargs2):
            kwargs3 = {**kwargs, **kwargs2}
            return self._invoke_method(method, (self, *args), **kwargs3)
        return func

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        return self._invoke_method(method, inputs, obj=ufunc, **kwargs)

    def _invoke_method(self, method, inputs, obj=None, **kwargs):
        if not all(isinstance(x, self.comp_tup) for x in inputs):
            return NotImplemented
#        if any(hasattr(x, 'degree') and x.degree > self.degree for x in inputs):
#            return NotImplemented

        outchildren = com.defaultdict(OrderedDict)
        for cID in self.children.keys():
            args = self._match(inputs, cID)
            if obj is None:
                obj2, args = args[0], args[1:]
            else:
                obj2 = obj

            outchild = getattr(obj2, method)(*args, **kwargs)
            if not isinstance(outchild, tuple):
                outchild = (outchild,)
            for i, ichild in enumerate(outchild):
                try:
                    n = len(ichild.children)
                except AttributeError:
                    try:
                        n = len(ichild.variables)
                    except AttributeError:
                        n = 1
                if n == 0:
                    continue
                outchildren[i][cID] = ichild

        handles = set()
        for idat in inputs:
            if hasattr(idat, 'handles'):
                handles |= idat.handles

        if len(outchildren) == 0:
            return self._new({}, self.name, self.attrs, handles)
        outdats = []
        for child in outchildren.values():
            outdats.append(self._new(child, self.name, self.attrs, handles))
            if len(child) > 0:
                outdats[-1]._sub = type(next(iter(child.values())))

        if len(outdats) == 1:
            return outdats[0]
        return tuple(outdats)

    def _match(self, inputs, ID):
        invars = []
        for idat in inputs:
            if hasattr(idat, 'degree') and idat.degree == self.degree:
                if ID in idat.children:
                    invars += [idat.children[ID]]
            else:
                invars += [idat]
        return tuple(invars)

    def _compute(self):
        vals = {}
        for iname, ichild in self.children.items():
            vals[iname] = ichild._compute()
        return vals

    def compute(self):
        vals = self._compute()
        outvals = da.compute(vals)[0]
        return self._comp_out(outvals)

    def _comp_out(self, outvals):
        outchildren = OrderedDict()
        for iname, ival in outvals.items():
            ichild = self.children[iname]
            outchildren[iname] = ichild._comp_out(ival)
        return self._new(outchildren, self.name, self.attrs, self.handles)

    def reindex(self, how=None, **indexers):
        indexers = cb.parse_indexes(indexers)
        return self._reindex(indexers, how=how)

    def _reindex(self, indexers, how=None):
        newchildren = {}
        for cID, child in self.children.items():
            newchildren[cID] = child._reindex(indexers, how=how)
        return self._new(newchildren, self.name, self.attrs, self.handles)

    def _apply_index(self, locs, indexers):
        newchildren = {}
        for cID, child in self.children.items():
            newchildren[cID] = child._apply_index(locs, indexers)
        return self._new(newchildren, self.name, self.attrs, self.handles)

    def groupby(self, *indexes):
        """
        groupby based on index
        """
        groups = com.defaultdict(type(self), name=self.name, attrs=self.attrs)
        for cID, child in self.children.items():
            datlist = child.groupby(*indexes)
            for dID, idat in datlist:
                groups[dID].children[cID] = idat
        return sorted(groups.items(), key=lambda x: x[0])

    def where(self, **kwargs):
        """
        make a selection based on value of timeseries variables
        """
        cond = []
        for ivar, values in kwargs.items():
            if isinstance(values, list):
                subcond = []
                for value in values:
                    subcond.append(self.aux[ivar] == value)
                cond.append(np.any(subcond))
            else:
                cond.append(self.aux[ivar] == values)
        cond == np.all(cond)

        newdat = type(self)(name=self.name, attrs=self.attrs)
        for cID, child in self.children.items():
            newdat.children[cID] = child.where(cond)
        newdat.aux = self.aux.where(cond)
        return newdat

#    def flatten(self):
#        """
#        convert dataset to flattened array
#        """
#        new = []
#        for child in self.children.values():
#            for ivar in child.data_vars.values():
#                new.append(ivar._values.flatten())
#        return da.concatenate(new)

    def flatten(self):
        inchildren = list(self.children.values())
        childname = self._sub.treename
        return self._sub.merged(inchildren, xdim=childname+'_id')

    def _create_store(self, outh):
        attrs = self._attrs
        attrs[self.treename + '_id'] = self.name
        attrs.to_store(outh.attrs)

        sources = []
        targets = []
        for cID, child in self.children.items():
            group = outh.require_group(cID)
            source, target = child._create_store(group)
            sources += source
            targets += target
        return sources, targets

    @classmethod
    def from_store(cls, inh, children=None, **kwargs):
        attrs = Attributes.from_store(inh)
        name = attrs.pop(cls.treename + '_id')

        if children is None:
            items = list(inh.items())
        else:
            items = [(key, inh[key]) for key in children]

        data = cls(name=name, attrs=attrs)
        for cID, ichan in items:
            child = cls._sub.from_store(ichan, **kwargs)
            data.children[cID] = child
        return data

    def to_store(self, outh):
        outh['attrs'] = {}
        self.attrs.to_store(outh['attrs'])
        outh['branchname'] = self._sub.treename
        outh['degree'] = self.degree
        outh['branches'] = {}
        for cID, child in self.children.items():
            outh['branches'][cID] = {}
            child.to_store(outh['branches'][cID])

    @classmethod
    def from_yaml(cls, inh, children=None, degree=None, **kwargs):
        if children is None:
            items = list(inh['branches'].items())
        else:
            items = [(key, inh['branches'][key]) for key in children]

        data = cls(attrs=Attributes(inh['attrs']))
        if degree is None:
            degree = inh['degree']
        data.degree = degree
        degree -= 1
        if degree == 0:
            data._sub = cb.MetaChan
        data._sub.treename = inh['branchname']
        for cID, ichan in items:
            child = data._sub.from_yaml(ichan, degree=degree, **kwargs)
            data.children[cID] = child
        return data

    @property
    def meta(self):
        try:
            return self._sub.meta.fget(self._sub)
        except AttributeError:
            return self._sub.meta

    def store(self, outpath):
        if self.meta:
            io.to_yaml(self, outpath)
        else:
            io.to_hdf(self, outpath)

    def merge(self, other, **kwargs):
        return self.merged([self, other], **kwargs)

    @classmethod
    def merged(cls, datlist, xdim=None, **kwargs):
        """
        concatenate sequence of station level containers into one container
        """
        if not datlist:
            return cls()

        datlist = com.prep_merger(datlist)
        dat0 = next(iter(datlist.values()))
        if len(datlist) <= 1:
            return dat0

        dd = com.defaultdict(OrderedDict)
        handles = set()
        for i, idat in datlist.items():
            if not (hasattr(idat, 'degree') and idat.degree == cls.degree):
                continue
            for cID, child in idat.children.items():
                dd[cID][i] = child
            handles |= idat.handles
        for i, idat in datlist.items():
            if hasattr(idat, 'degree') and idat.degree == cls.degree:
                continue
            cID = idat.name
            dd[cID][i] = idat
            handles |= idat.handles

        if xdim is None:
            attrs = Attributes.merged(datlist)
        else:
            attrs, xdim = cb.merge_attrs(datlist, xdim)
        cdat = cls(name=dat0.name, attrs=attrs)
        cdat._sub = dat0._sub
        for cID in dd.keys():
            cdat.children[cID] = cdat._sub.merged(dd[cID], xdim=xdim, **kwargs)
        return cdat

    def update(self, other, **kwargs):
        """
        concatenate sequence of station level containers into one container
        """
        datlist = [self, other]

        dd = com.defaultdict(list)
        handles = set()
        for idat in datlist:
            if not (hasattr(idat, 'degree') and idat.degree == self.degree):
                continue
            for cID, child in idat.children.items():
                dd[cID].append(child)
            handles |= idat.handles
        for cID in dd.keys():
            for idat in datlist:
                if hasattr(idat, 'degree') and idat.degree == self.degree:
                    continue
                dd[cID].append(idat)
            handles |= idat.handles

        attrs = Attributes({**self.attrs, **other.attrs})
        cdat = type(self)(name=self.name, attrs=attrs)
        cdat._sub = self._sub
        for cID in dd.keys():
            if len(dd[cID]) == 1:
                cdat.children[cID] = dd[cID][0]
            else:
                cdat.children[cID] = dd[cID][0].update(dd[cID][1], **kwargs)
        return cdat

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.close()

    def plot(self, label=None, **kwargs):
        for name, child in self.children.items():
            if label is None:
                child.plot(**kwargs)
            else:
                if label == '':
                    label2 = name
                else:
                    label2 = '-'.join([label, name])
                child.plot(label=label2, **kwargs)


def inject_ops(cls):
    other_methods = {'argsort', 'searchsorted', 'item'}
    simple_ops = {'lt', 'le', 'ge', 'gt', 'eq', 'ne', 'neg', 'pos', 'abs',
                  'invert'}
    num_binary_ops = {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                      'pow', 'and', 'xor', 'or', 'matmul'}
    unary_methods = {'clip', 'conj', 'real', 'imag', 'all', 'any'}
    nan_reduce_methods = {'argmax', 'argmin', 'max', 'min', 'mean', 'prod',
                          'sum', 'std', 'var', 'median'}
    con_methods = {'isel', 'sel', 'drop', 'ortho', 'inline', 'resample',
                   'rechunk', 'aselect', 'swap_dims', 'reindex',
                   'reindex_like', 'conform', 'mask', 'upsample'}

    for name in simple_ops | num_binary_ops:
        name = '__'+name+'__'
        setattr(cls, name, cls._make_op(name))

    for name in num_binary_ops:
        # only numeric operations have in-place and reflexive variants
        name = '__r'+name+'__'
        setattr(cls, name, cls._make_op(name))
        name = '__i'+name+'__'
        setattr(cls, name, cls._make_op(name))

    for name in unary_methods | nan_reduce_methods | con_methods:
        setattr(cls, name, cls._make_op(name))

    for name in nan_reduce_methods:
        name = 'nan'+name
        setattr(cls, name, cls._make_op(name))


inject_ops(Node)


###############################################################################
class Station(Node):
    """
    data container for one measuring device/station
    """
    _sub = cb.Channel
    degree = 1
    treename = 'station'

    @Node.attrs.setter
    def attrs(self, values):
        # set default values for standard station attributes
        attrs = Attributes()
        attrs['system_type'] = com.get_fill(str)
        attrs['system_manufacturer'] = com.get_fill(str)
        attrs['system_model'] = com.get_fill(str)

        # only for point device
#        if self.pixeldims == 0:
#            self.attrs['site_id'] = common.get_fill(str)
#            self.attrs['site_latitude'] = common.get_fill(float)
#            self.attrs['site_longitude'] = common.get_fill(float)
#            self.attrs['site_altitude'] = common.get_fill(float)
#
#        elif self.pixeldims == 1:
#            # only for link
#            self.attrs['site_a_id'] = common.get_fill(str)
#            self.attrs['site_a_latitude'] = common.get_fill(float)
#            self.attrs['site_a_longitude'] = common.get_fill(float)
#            self.attrs['site_a_altitude'] = common.get_fill(float)
#            self.attrs['site_b_id'] = common.get_fill(str)
#            self.attrs['site-b_latitude'] = common.get_fill(float)
#            self.attrs['site_b_longitude'] = common.get_fill(float)
#            self.attrs['site_b_altitude'] = common.get_fill(float)

        if values is not None:
            for key, value in values.items():
                attrs[key] = value
        self._attrs = attrs

    def apply(self, func, *args, recurse=False, **kwargs):
        """
        apply function to all channels of the link
        """
        outchildren = {}
        for cID, child in self.children.items():
            logging.info(self.name + ' ' + cID)
            mattrs = Attributes.merged([self, child])
            if recurse is True:
                outchild = child.apply(func, mattrs, *args, **kwargs)
            else:
                outchild = func(child, mattrs, *args, **kwargs)
            if outchild is not None:
                outchildren[cID] = outchild
        return self._new(outchildren, self.name, self.attrs, self.handles)

    @property
    def path(self):
        """
        return path based on GPS coordinates of endpoints
        """
        if not hasattr(self, '_path'):
            y1 = self.attrs['site_a_latitude']
            x1 = self.attrs['site_a_longitude']
            z1 = self.attrs['site_a_altitude']
            y2 = self.attrs['site_b_latitude']
            x2 = self.attrs['site_b_longitude']
            z2 = self.attrs['site_b_altitude']
            self._path = geometry.haversines(x1, y1, z1, x2, y2, z2)
        return self._path

    @property
    def length(self):
        """
        return pathlength based on GPS coordinates of endpoints
        """
        if not hasattr(self, '_length'):
            path = self.path
            self._length = np.sqrt(path @ path)
        return self._length


Station.comp_tup = cb.Channel.comp_tup + (Station,)


class Network(Node):
    """
    describes a set of pointstations. analogue to HDF5 root group
    """
    _sub = Station
    degree = 2
    treename = 'set'

    @Node.attrs.setter
    def attrs(self, values):
        # set default values for standard Network attributes
        attrs = Attributes()
        attrs['file_format'] = 'CMLh5'
        attrs['file_format version'] = '0.1'
        attrs['pro_id'] = com.get_fill(str)
        attrs['level'] = com.get_fill(str)
        attrs['source_type'] = com.get_fill(str)
        attrs['temporal_resolution'] = com.get_fill(str)

        if values is not None:
            for key, value in values.items():
                attrs[key] = value
        self._attrs = attrs

    def apply(self, func, *args, recurse=True, **kwargs):
        """
        use a function on all stations in a set
        """
        outchildren = {}
        for cID, child in self.children.items():
            if recurse is True:
                outchild = child.apply(func, *args, **kwargs)
            else:
                outchild = func(child, *args, **kwargs)
            if outchild is not None:
                outchildren[cID] = outchild
        return self._new(outchildren, self.name, self.attrs, self.handles)


Network.comp_tup = Station.comp_tup + (Network,)


###############################################################################
def convert_to_set(data, scales, meta, **kwargs):
    """
    convert dictionary of arrays with seperate metadata to Network object
    with integrated metadata
    """
    logging.info('converting to data container')
    dset = Network(name=meta['set_id'], attrs=meta['attrs'])
    for sID, smet in meta['stations'].items():
        stat = Station(name=sID, attrs=smet['attrs'])
        for cID, cmet in smet['channels'].items():
            newvars = cb.Variables()
            for vID, vmet in cmet['variables'].items():
                name = sID + cID + vID
                if name not in data:
                    continue
                newvar = data[name]
                if isinstance(newvar, Delayed):
                    shape = tuple(len(scales[key]) for key in vmet['dimensions'])
                    newvar = da.from_delayed(newvar, shape, vmet['dtype'])
                newvars[vID] = var.Variable(newvar, vmet['dimensions'],
                                      name=vID, attrs=vmet['attrs'], **kwargs)
            if not newvars:
                continue
            chan = cb.Channel(variables=newvars, coords=scales, name=cID,
                              attrs=cmet['attrs'])
            stat[cID] = chan
        if not len(stat):
            continue
        dset[sID] = stat
    return dset


def regularize(dset, indexes, step):
    locs = {}
    newindexes = cb.Indexes()
    for i, iindex in enumerate(indexes):
        idim = iindex.name

        if iindex.dtype.kind == 'M':
            newstart = iindex.start.astype('M8[us]').astype(int)
            newstop = iindex.stop.astype('M8[us]').astype(int)
            instep = com.str_to_td64(step[i]).astype('m8[us]').astype(int)
            newstart = (newstart - newstart % instep).astype('<M8[us]')
            newstop = (newstop - newstop % instep + instep).astype('<M8[us]')
        else:
            newstart = newstart - newstart % step[i]
            newstop = newstop - newstop % step[i]

        newindex = var.carange(newstart, newstop, step[i], name=idim)
        n = len(newindex)
        newindexes[idim] = newindex.rechunk(n)
        locs[idim] = iindex.rechunk(n).get_indexer(newindexes[idim], atol=np.timedelta64(2, 'm'))
    return dset._apply_index(locs, newindexes).compute()


def merge(datlist, **kwargs):
    cls = type(datlist[0])
    return cls.merged(datlist, **kwargs)
