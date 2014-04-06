
from biofilm.model import spec as sp, runner, analysis, result as rs
from biofilm import util

from matplotlib import pyplot as plt

STOP = {}
INVALID = {}

def search(h5_file=None):
    if h5_file:
        util.set_h5(h5_file)

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.ion()

    params = sp.DEFAULT_PARAMETERS.copy()
    while True:
        params = read_command(params)
        if params is STOP: break
        run_model(params)

    util.get_h5().close()

COMMAND_STR = '(v)iew current parameters, (c)hange a parameter, (r)un, (q)uit => '

def read_command(params):
    while True:
        command = raw_input(COMMAND_STR)
        if command == 'v':
            show_params(params)
        elif command == 'c':
            new_params = do_change(params)
            if new_params is not INVALID:
                params = new_params
        elif command == 'r':
            return params
        elif command == 'q':
            return STOP
        else:
            print 'Unknown command.'

def run_model(params):
    spec = sp.Spec(**params)

    def on_step(model):
        if model.time % 10 == 0:
            print model.time,
            plt.imshow(model.cells)
            plt.pause(0.0001)
        if model.time % 200 == 0:
            command = raw_input('(s)top, anything else to continue => ')
            return command == 's' 
        return False

    try:
        model = runner.run(spec, on_step=on_step)
    except KeyboardInterrupt:
        print 'Stopping model.'
    else:
        rs.save_model(model)

def show_params(params):
    for i, param_name in enumerate(sp.INDEX_TO_PARAM):
        print '({0}) {1}: {2}'.format(i, param_name, params[param_name])

def do_change(params):
    try:
        index = int(raw_input('Input a parameter index => '))
        if index < 0 or index >= len(sp.INDEX_TO_PARAM):
            raise ValueError(index)
    except ValueError:
        print 'Invalid index.'
        return INVALID

    print 'Current value is', params[sp.INDEX_TO_PARAM[index]]

    try:
        value = input('Input a new value => ')
    except (ValueError, SyntaxError):
        print 'Invalid value.'
        return INVALID

    new_params = params.copy()
    new_params[sp.INDEX_TO_PARAM[index]] = value
    try:
        sp.Spec(**new_params)
    except sp.ParameterValueError as e:
        print e
        return INVALID

    return new_params

if __name__ == '__main__':
    import sys
    h5_file = sys.argv[1]
    search(util.results_h5_path(h5_file))