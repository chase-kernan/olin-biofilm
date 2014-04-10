
from biofilm import util
util.set_h5(util.results_h5_path('final_tension_vs_distance'))

from biofilm.model.spec import Spec, SpecBuilder
from biofilm.model import runner, result
import numpy as np

builder = SpecBuilder()
builder.add('stop_on_mass', 4000)
builder.add('stop_on_time', 10000)
builder.add('stop_on_no_growth', 500)
builder.add('boundary_layer', 5)
builder.add('light_penetration', 0)
builder.add('tension_power', *np.linspace(0.1, 4.0, 40))
builder.add('distance_power', *np.linspace(0.1, 4.0, 40))
builder.add('initial_cell_spacing', 2)
builder.add('media_ratio', 2)
builder.add('media_monod', 0.35)
builder.add('division_rate', 3.0)
total = builder.num_specs
builder.build()

for i, spec in enumerate(Spec.all()):
    print i, 'of', total
    for _ in range(3):
        result.from_model(runner.run(spec)).save()

print 'done!'