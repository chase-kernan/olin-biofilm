#!/usr/bin/python

param_ranges = {
    'stop_on_mass': 4000,
    'stop_on_time': 60000,
    'stop_on_no_growth': 1000,
    'boundary_layer': 7,
    'light_penetration': (0.1, 50),
    'tension_power': (0.1, 3),
    'distance_power': (0.1, 3),
    'initial_cell_spacing': 2,
    'media_ratio': (0.01, 0.60),
    'media_monod': (0.01, 1),
    'light_monod': (0.01, 1)
}

def run_sample(runner_id):
    from biofilm import util
    util.set_h5(util.results_h5_path('sample_{0}'.format(runner_id)))

    from biofilm.model import spec, runner, result
    import numpy as np

    def sample_spec():
        params = {}
        for param, r in param_ranges.iteritems():
            if isinstance(r, tuple):
                params[param] = np.random.uniform(*r)
            else:
                params[param] = r
        return spec.Spec(**params)

    for i in range(10000):
        s = sample_spec()
        s.save()
        result.from_model(runner.run(s)).save()
        print i,

import sys
arg = sys.argv[1]

if arg == 'analyze':
    print 'analyzing'
    from biofilm import util
    util.set_h5(util.results_h5_path('merged_sample_new'))

    import numpy as np
    from biofilm.model import spec, result, analysis
    from sklearn.linear_model import LinearRegression

    param_features = []
    for param, r in param_ranges.iteritems():
        if isinstance(r, tuple):
            param_features.append(param)

    num_results = min(1e6, result.Result.count())
    features = np.empty((num_results, len(param_features)), float)
    print features.shape
    values = np.empty(features.shape[0], float)
    for i, res in enumerate(result.Result.all()):
        if i % 100 == 0: print i,
        if i >= num_results: break
        for j, param in enumerate(param_features):
            features[i, j] = getattr(res.spec, param)
        #values[i] = analysis.horizontal_surface_area.compute_by_result(res, flush=False)
        values[i] = analysis.horizontal_surface_area.func(res)
    print
    regression = LinearRegression()
    regression.fit(features, values, n_jobs=-1)
    print 'Features', param_features
    print 'Coeff', regression.coef_
    print 'Intercept', regression.intercept_
    print 'Params', regression.get_params()
    print 'R2', regression.score(features, values)
elif arg == 'dump':
    from biofilm import util
    path = util.results_h5_path(sys.argv[2])
    util.set_h5(path)

    import numpy as np
    from biofilm.model import spec, result, analysis

    param_features = []
    for param, r in param_ranges.iteritems():
        if isinstance(r, tuple):
            param_features.append(param)

    with open(path + '.dump', 'w') as dump:
        dump.write(','.join(param_features + ['mass', 'perimeter', 'hsa']) + '\n')

        for i, res in enumerate(result.Result.all()):
            for param in param_features:
                dump.write('{0},'.format(getattr(res.spec, param)))
            dump.write('{},{},{},\n'.format(analysis.mass.func(res),
                                            analysis.perimeter.func(res),
                                            analysis.horizontal_surface_area.func(res)))
elif arg == 'analyze_dump':
    import numpy as np
    import os
    import csv
    from sklearn.linear_model import LinearRegression

    param_features = []
    for param, r in param_ranges.iteritems():
        if isinstance(r, tuple):
            param_features.append(param)

    features = []
    values = []
    for name in os.listdir('dump'):
        print name
        with open('dump/'+name, 'r') as csv_file:
            for row in csv.DictReader(csv_file):
                #row['media_ratio'] = 1/(25*float(row['media_ratio']))
                features.append([float(row[param]) for param in param_features])
                values.append(float(row['hsa']))

    print len(features)
    features = np.array(features, dtype=float)
    print features.shape
    values = np.array(values, dtype=float)

    regression = LinearRegression()
    regression.fit(features, values, n_jobs=-1)
    print 'Features', param_features
    print 'Coeff', regression.coef_
    print 'Intercept', regression.intercept_
    print 'Params', regression.get_params()
    print 'R2', regression.score(features, values)

else:
    run_sample(arg)
