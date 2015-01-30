#!/usr/bin/python

from biofilm.model import spec, runner
import numpy as np
import sys
from boto.dynamodb2.table import Table
from boto.dynamodb2 import connect_to_region
import uuid

param_ranges = {
    'stop_on_mass': (0, 5000),
    'stop_on_time': 100000,
    'stop_on_no_growth': 500,
    'boundary_layer': 5,
    'tension_power': (0.01, 2.0),
    'distance_power': (0.01, 2.0),
    'initial_cell_spacing': (1, 50),
    'media_ratio': (0.01, 1.2),
    'media_monod': (0.01, 1.0),
    'light_monod': (0.01, 1.0),
    'division_rate': 2.0,
    'stop_on_height': 40,
    'stop_on_no_growth': 1000,
    'width': 128,
    'height': 40,
}

table = Table('biofilm-parameter-space', connect_to_region('us-west-2'))

def sample_spec():
    params = {}
    for param, r in param_ranges.iteritems():
        if isinstance(r, tuple):
            if all(isinstance(x, int) for x in r):
                params[param] = int(round(np.random.uniform(*r)))
            else:
                params[param] = np.random.uniform(*r)
        else:
            params[param] = r

    if np.random.random() < 0.5:
        params['light_penetration'] = 0
    else:
        params['light_penetration'] = np.random.uniform(3.0, 20.0)

    return spec.Spec(**params)

spec_fields = ['stop_on_mass', 'boundary_layer', 'light_penetration', 'distance_power',
	   'tension_power', 'distance_power', 'initial_cell_spacing', 'media_ratio',
	   'media_monod', 'light_monod', 'division_rate']
result_fields = ['biomass', 'coverage', 'time']

for i in range(1000000):
    s = sample_spec()
    result = runner.run(s)
    
    row = { 'id': uuid.uuid4().hex }
    for field in spec_fields: row[field] = str(getattr(s, field))
    row['biomass'] = str(result.mass/float(s.width))
    row['coverage'] = str(result.cells.sum(axis=1)/float(s.width))
    row['time'] = str(result.time)
    print table.put_item(data=row), i



