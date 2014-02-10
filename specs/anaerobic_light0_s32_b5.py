
from biofilm import util
util.set_h5(utils.results_h5_path('anaerobic_light0_s32_b5'))

from biofilm.search import byscore
from biofilm.classify import score

scorer = score.Scorer('anaerobic', target_mass=3500)
params = {
    'stop_on_mass': scorer.target_mass,
    'boundary_layer': 5,
    'light_penetration': 0,
    'tension_power': (0.1, 3),
    'initial_cell_spacing': 32,
    'media_ratio': (0.1, 5),
    'media_monod': (0.01, 1),
    'light_monod': (0.01, 1)
}
print 'anaerobic_light0_s32_b5'
result = maximize(scorer, params)
print result
