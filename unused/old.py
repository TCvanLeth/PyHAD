# -*- coding: utf-8 -*-
from memory_profiler import memory_usage
import psutil
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
            arrays = []
            key_nums = []
            for dID, ikey in enumerate(keys):
                if isinstance(ikey, slice) and ikey != slice(None):
                    ind = ikey.indices(self.shape[dID])
                    arrays.append(creation.arange(*ind, chunks=com.CHUNKSIZE))
                    key_nums.append(dID)
                elif isinstance(ikey, slice) and ikey == slice(None):
                    continue
                else:
                    if not hasattr(ikey, 'chunks'):
                        ikey = da.from_array(np.asarray(ikey), chunks=com.CHUNKSIZE)
                    if ikey.ndim > 1 or ikey.dtype.kind not in ('i', 'b'):
                        raise ValueError(ikey)
                    if ikey.ndim == 0:
                        ikey = ikey.reshape(1)
                    arrays.append(ikey)
                    key_nums.append(dID)

            n = len(arrays)
            array_indexers = []
            for i, array in enumerate(arrays):
                axes = list(range(n))
                axes[i] = n-1
                axes[n-1] = i
                array_indexers += [da.transpose(array[(None,)*(n-1)], axes)]

            for i, dID in enumerate(key_nums):
                keys[dID] = array_indexers[i]
        return tuple(keys)

def chunk(func):
    @wraps(func)
    def nfunc(*args, **kwargs):
        args = np.lib.stride_tricks.broadcast_arrays(*args)
        shp = args[0].shape
        args = tuple(x.ravel() for x in args)
        single = tuple(x[0] for x in args)

        mem, val = memory_usage((func, single, kwargs), retval=True)
        if isinstance(val, tuple):
            nshp = [x.shape for x in val]
        else:
            nshp = val.shape
        del val
        mem = max(mem)*1024*1024/10
        avail_mem = psutil.virtual_memory().available
        chunksize = int(avail_mem//mem)

        ochunks = []
        for slicer in dicer(len(args[0]), chunksize):
            ichunk = tuple(x[slicer] for x in args)
            ochunks += [func(*ichunk, **kwargs)]

        if isinstance(ochunks[0], tuple):
            outvals = com.defaultdict(list)
            for ochunk in ochunks:
                for i, x in enumerate(ochunk):
                    outvals[i] += [x]
            y = outvals.items()
            return tuple(np.concatenate(x).reshape(shp+nshp[i]) for i, x in y)
        return (np.concatenate(ochunks)).reshape(shp+nshp)
    return nfunc


def dicer(length, chunksize):
    start = 0
    stop = 0
    while stop <= length:
        stop = start + chunksize
        slicer = slice(start, stop)
        start = stop
        yield slicer