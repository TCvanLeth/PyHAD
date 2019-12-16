# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:03:57 2016

@author: tcvanleth
"""
from collections import OrderedDict

import numpy as np

from phad import common as com
from phad import printing as pri


class Attributes(OrderedDict):
    """
    """
    def __repr__(self, col_width=None):
        summary = ['Attributes:']
        if self:
            if not col_width:
                max_name_length = max(len(str(vID)) for vID in self)
                col_width = max(max_name_length, 7) + 6
            summary += [pri.pprint('    %s: ' % aID, col_width) + str(iattr)
                        for aID, iattr in self.items()]
        else:
            summary += ['    *empty*']
        return '\n'.join(summary)

    def equivalent(self, other):
        def equivalent(arr1, arr2):
            if isinstance(arr1, np.ndarray) or isinstance(arr2, np.ndarray):
                com.array_equal(arr1, arr2)
            else:
                return arr1 is arr2 or arr1 == arr2

        for key in self:
            if key not in other or not equivalent(self[key], other[key]):
                return False
        for key in other:
            if key not in self:
                return False
        return True

    def to_store(self, outh):
        """
        write attributes to node
        """
        for aID, att in self.items():
            if att is not None:
                if isinstance(att, np.timedelta64):
                    outh[aID] = att.astype('<m8[us]').view(int)
                elif isinstance(att, np.datetime64):
                    outh[aID] = str(att)
                else:
                    outh[aID] = com.np_to_base_type(att)

    @classmethod
    def merged(cls, datlist):
        datlist = com.prep_merger(datlist)

        dd = com.defaultdict(list)
        for i, idat in datlist.items():
            if hasattr(idat, 'attrs'):
                for key, value in idat.attrs.items():
                    dd[key] += [value]

        attrs = cls()
        for key, value in dd.items():
            try:
                attr = [x for x in value if not com.isnull(x)]
            except NotImplementedError as e:
                raise NotImplementedError(key, str(e))
            if len(attr):
                attrs[key] = attr[0]
            else:
                attrs[key] = value[0]
        return attrs

    @classmethod
    def merge_reduce(cls, args):
        """
        delete conflicting/meaningless attributes
        """
        dd = com.defaultdict(list)
        for arg in args:
            if hasattr(arg, '_attrs'):
                for key, value in arg._attrs.items():
                    dd[key] += [value]

        attrs = cls()
        for key, value in dd.items():
            attr = [x for x in value if not com.isnull(x)]
            if (len(attr) == 1 or len(attr) > 1 and
                all(x == attr[0] for x in attr)):
                attrs[key] = attr[0]
        return attrs

    @classmethod
    def from_store(cls, inh):
        attrs = cls()
        for k, v in inh.attrs.items():
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode()
            if type(v) == bool:
                v = int(v)
            attrs[k] = v
        attrs.pop('CLASS', None)
        attrs.pop('DIMENSION_LABELS', None)
        attrs.pop('DIMENSION_LIST', None)
        attrs.pop('REFERENCE_LIST', None)
        attrs.pop('NAME', None)
        return attrs
