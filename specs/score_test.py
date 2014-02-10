
from biofilm.model import spec, runner, result
from biofilm.classify import score

def build():
    builder = spec.SpecBuilder()
    builder.add('stop_on_mass', 2500)
    builder.add('boundary_layer', 5)
    builder.add('light_penetration', 0, 6)
    builder.add('initial_cell_spacing', 2, 32)
    builder.add('division_rate', 1)
    builder.add('media_ratio', 0.2, 2.0)
    builder.add('media_monod', 0.1, 0.3, 0.7)
    builder.add('light_monod', 0.1, 0.3, 0.7)
    print builder.num_specs
    builder.build()

def run():
    aerobic = score.Scorer('aerobic')
    anaerobic = score.Scorer('anaerobic')
    flat = score.Scorer('flat')
    for i, sp in enumerate(spec.Spec.all()):
        print i
        model = runner.run(sp)
        result.save_model(model)
        print 'Spec', sp
        print 'Aerobic', aerobic.score(model.render())
        print 'Anaerobic', anaerobic.score(model.render())
        print 'Flat', flat.score(model.render())

if __name__ == '__main__':
    build()
    run()

