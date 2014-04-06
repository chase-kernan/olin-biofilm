'''
Created on Nov 12, 2012

@author: Chase Kernan
'''

import tables as tb
from biofilm import util
import numpy as np
from itertools import product

DEFAULT_PARAMETERS = {
    'stop_on_mass': 2500,
    'stop_on_time': 1000,
    'stop_on_height': 32,
    'stop_on_no_growth': 50,
    'width': 256,
    'height': 32,
    'block_size': 11,
    'boundary_layer': 5,
    'light_penetration': 0,
    'distance_power': 2.0,
    'tension_power': 2.5,
    'initial_cell_spacing': 2,
    'division_rate': 1.0,
    'media_ratio': 0.2,
    'media_monod': 0.5,
    'light_monod': 1.0,
}

INDEX_TO_PARAM = sorted(DEFAULT_PARAMETERS.keys())
PARAM_TO_INDEX = dict((p, i) for i, p in enumerate(INDEX_TO_PARAM))

def spec_from_list(xs):
    return Spec(**dict((INDEX_TO_PARAM[i], x) for i, x in enumerate(xs)))

def spec_to_list(s):
    return [getattr(s, p) for p in INDEX_TO_PARAM]

class Spec(util.TableObject):

    def __init__(self, uuid=None, **kwargs):
        util.TableObject.__init__(self, uuid)
        params = _with_defaults(kwargs, DEFAULT_PARAMETERS)
        for name, value in params.iteritems():
            setattr(self, name, value)

        self.verify()

    @property
    def shape(self): return (self.height, self.width)

    def is_between(self, name, min_value, max_value):
        value = getattr(self, name)
        if value < min_value or value > max_value:
            raise ParameterValueError(name, value, 
                                      "Must be in the range {0} to {1}."\
                                      .format(min_value, max_value))

    def verify(self):
        self.is_between("boundary_layer", 0, 32)
        self.is_between("light_penetration", 0, 1024)
        self.is_between("division_rate", 1e-5, 1e5)
        self.is_between("initial_cell_spacing", 0, self.width-1)
        self.is_between("media_monod", 1e-5, 1e5)
        self.is_between("media_monod", 1e-5, 1e5)
        self.is_between("distance_power", 0.0, 4.0)
        self.is_between("tension_power", 0.0, 4.0)
        self.is_between("block_size", 3, 25)
        if self.block_size % 2 != 1:
            raise ParameterValueError("block_size", self.block_size,
                                      "Must be an odd integer.")
        
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.__dict__)

class SpecTable(tb.IsDescription):
    uuid = util.make_uuid_col()
    stop_on_mass = tb.UInt32Col()
    stop_on_time = tb.UInt32Col()
    stop_on_height = tb.UInt32Col()
    stop_on_no_growth = tb.UInt32Col()
    width = tb.UInt16Col()
    height = tb.UInt16Col()
    block_size = tb.UInt8Col()
    boundary_layer = tb.UInt8Col()
    light_penetration = tb.UInt16Col()
    distance_power = tb.Float32Col()
    tension_power = tb.Float32Col()
    initial_cell_spacing = tb.UInt16Col()
    division_rate = tb.Float32Col()
    media_ratio = tb.Float32Col()
    media_monod = tb.Float32Col()
    light_monod = tb.Float32Col()
Spec.setup_table("specs", SpecTable)

class ParameterValueError(Exception):
    def __init__(self, name, value, reason=None):
        super(ParameterValueError, self).__init__({'name':name, 
                                                   'value':value, 
                                                   'reason':reason})

def _with_defaults(d, defaults):
    new_d = defaults.copy()
    if d:
        for key, value in d.iteritems():
            if key not in defaults:
                raise ValueError("{0} not a valid key.".format(key))
            new_d[key] = value
    return new_d
    
class SpecBuilder(object):
    
    def __init__(self):
        self._all_values = {}
    
    def add(self, column, *values):
        self._all_values.setdefault(column, []).extend(values)
        
    @property
    def num_specs(self):
        return np.product([len(v) for v in self._all_values.itervalues()])

    @property
    def value_sets(self):
        return map(dict, product(*([(name, value) for value in values]
                                   for name, values 
                                   in self._all_values.iteritems())))
    
    def build(self):
        for value_set in self.value_sets:
            spec = Spec(**value_set)
            spec.save(flush=False)
        Spec.table.flush()
        self.clear()

    def clear(self):
        self._all_values = {}

def make_query(use_defaults=False, **conditions):
    specified = set()
    clauses = []
    for column, column_clauses in conditions.iteritems():
        specified.add(column)
        if isinstance(column_clauses, basestring):
            column_clauses = (column_clauses,)
        clauses.extend((column + clause) for clause in column_clauses)
    
    if use_defaults:
        for param, value in DEFAULT_PARAMETERS.iteritems():
            if param not in specified:
                clauses.append("{0}=={1}".format(param, value))
    
    return "&".join("({0})".format(clause) for clause in clauses)
