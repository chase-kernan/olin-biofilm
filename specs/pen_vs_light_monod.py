
import sys
pen = int(sys.argv[1])
print 'penetration = {}'.format(pen)

from biofilm import util
util.set_h5(util.results_h5_path('pen_vs_light_monod_pen{}'.format(pen)))

from biofilm.model.spec import Spec, SpecBuilder
from biofilm.model import runner, result
import numpy as np

builder = SpecBuilder()
builder.add('stop_on_mass', 4000)
builder.add('stop_on_time', 50000)
builder.add('stop_on_no_growth', 1000)
builder.add('boundary_layer', 5)
builder.add('light_penetration', *np.linspace(pen+1e-4, pen+1, 5))
builder.add('tension_power', 1)
builder.add('distance_power', 2)
builder.add('initial_cell_spacing', 2)
builder.add('media_ratio', 1)
builder.add('media_monod', 0.25)
builder.add('light_monod', *np.linspace(0.01, 1, 50))
#builder.add('division_rate', 2.0)
total = builder.num_specs
builder.build()

for i, spec in enumerate(Spec.all()):
    print i, 'of', total
    for _ in range(5):
        result.from_model(runner.run(spec)).save()

print 'done!'
