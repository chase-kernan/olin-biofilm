
from biofilm import util
util.set_h5(util.results_h5_path('light0_s_b_grid'))

from biofilm.model.spec import Spec, SpecBuilder
from biofilm.model import runner, result
import numpy as np

builder = SpecBuilder()
builder.add('stop_on_mass', 5000)
builder.add('boundary_layer', *range(1, 11))
builder.add('light_penetration', 0)
builder.add('initial_cell_spacing', *range(2, 48, 2))
builder.add('media_ratio', 2)
builder.add('media_monod', 0.5)
builder.add('light_monod', 0)
total = builder.num_specs
builder.build()

for i, spec in enumerate(Spec.all()):
    print i, 'of', total
    result.from_model(runner.run(spec)).save()

print 'done!'