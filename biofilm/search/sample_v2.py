#!/usr/bin/python

from biofilm.model import spec, runner
import numpy as np
import sys
import csv

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


num_to_run = int(sys.argv[1])
results_path = sys.argv[2]


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

with open(results_path, 'w') as f:
    spec_fields = ['stop_on_mass', 'boundary_layer', 'light_penetration', 'distance_power',
                   'tension_power', 'distance_power', 'initial_cell_spacing', 'media_ratio',
                   'media_monod', 'light_monod', 'division_rate']
    result_fields = ['biomass', 'coverage', 'time']

    writer = csv.DictWriter(f, fieldnames=spec_fields+result_fields)
    writer.writeheader()

    for i in range(num_to_run):
        print i,

        s = sample_spec()
        result = runner.run(s)
        
        row = {}
        for field in spec_fields: row[field] = s[field]
        row['biomass'] = result.mass/float(spec['width'])
        row['coverage'] = result.cells.sum(axis=1)/float(spec['width'])
        row['time'] = result.time
        writer.writerow(row)



