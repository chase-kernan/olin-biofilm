
from biofilm import util
util.set_h5(util.results_h5_path('light_s2_b5_grid'))

from biofilm.model.spec import Spec, SpecBuilder, make_query
from biofilm.model import runner, result
import numpy as np
import sys

if sys.argv[1] == 'analyze':
    from biofilm.model import analysis as an
    from biofilm.classify.score import Scorer

    scorer = Scorer('aerobic', target_mass=5000)
    anaerobic_score = an.Field(func=lambda r: scorer.score(r.image),
                               path='anaerobic_score')
    for i, spec in enumerate(Spec.all()):
        print i
        #anaerobic_score.compute_by_spec(spec)
    anaerobic_score.phase_diagram_2d('light_penetration', 'light_monod', 
                                     num_cells=10, 
                                     spec_query=make_query(media_monod='>=0.8', media_ratio=('>=2', '<=3')),
                                     show=True)

else:
    builder = SpecBuilder()
    builder.add('stop_on_mass', 5000)
    builder.add('boundary_layer', 5)
    builder.add('light_penetration', 4, 6, 8, 10, 12)
    builder.add('tension_power', 2)
    builder.add('initial_cell_spacing', 2)
    builder.add('media_ratio', *np.linspace(0.1, 5.0, 10))
    builder.add('media_monod', *np.linspace(0.1, 1, 5))
    builder.add('light_monod', *np.linspace(0.1, 1, 10))
    total = builder.num_specs
    builder.build()

    for i, spec in enumerate(Spec.all()):
        print i, 'of', total
        result.from_model(runner.run(spec)).save()

    print 'done!'
