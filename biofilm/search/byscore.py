
from biofilm.model import spec, runner, result
import numpy as np
from scipy.optimize import anneal

def merge_params(params):
    merged = spec.DEFAULT_PARAMETERS.copy()
    merged.update(params)
    return merged

def maximize(scorer, params):
    params = merge_params(params)
    param_list = np.array([params[p] for p in spec.INDEX_TO_PARAM], object)

    used = np.array([isinstance(p, tuple) for p in param_list])
    used_indices = np.where(used)
    not_used = np.logical_not(used)
    ndim = used.sum()

    x0 = np.array([sum(p)/2.0 for u, p in zip(used, param_list) if u])

    base_full_x = param_list.copy()
    base_full_x[used] = x0
    
    def objective(x):
        full_x = base_full_x.copy()
        full_x[used] = x

        s = spec.spec_from_list(full_x)
        print s
        s.save()
        m = runner.run(s)
        result.from_model(m).save()

        score = scorer.score(m.render())
        print score
        return 1.0 - score # invert score to maximize

    return anneal(objective, x0, full_output=True, Tf=1e-8, dwell=10, disp=True,
                  lower=[lower for lower, upper in param_list[used]],
                  upper=[upper for lower, upper in param_list[used]])

if __name__ == '__main__':
    from biofilm.classify import score
    scorer = score.Scorer('anaerobic', target_mass=3500)
    params = {
        'stop_on_mass': scorer.target_mass,
        'boundary_layer': 5,
        'light_penetration': (0.1, 32),
        'tension_power': (0.1, 3),
        'initial_cell_spacing': 16,
        'media_ratio': (0.2, 5),
        'media_monod': (0.01, 1),
        'light_monod': (0.01, 1)
    }
    print maximize(scorer, params)


