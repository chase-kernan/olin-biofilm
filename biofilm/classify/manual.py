
from matplotlib import pyplot as plt
from tables.misc.enum import Enum
import tables as tb
from biofilm import util
from biofilm.model.result import Result

CLASSES = tb.misc.enum.Enum(['column', 'mushroom', 'flat', 'no growth', 
                             'cactus', 'other', 'late stage'])

class Classification(util.EasyTableObject): pass
class ClassificationTable(tb.IsDescription):
    uuid = util.make_uuid_col()
    spec_uuid = util.make_uuid_col()
    result_uuid = util.make_uuid_col()
    class_ = tb.EnumCol(CLASSES, 'other', tb.UInt8Atom())

Classification.setup_table("classifications", ClassificationTable,
                           sorted_indices=['uuid', 'result_uuid', 'spec_uuid'])

STOP = {}
IGNORE = {}

def classify(h5_file=None):
    if h5_file:
        util.set_h5(h5_file)

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.ion()

    for i, result in enumerate(Result.all(random_order=True)):
        print 'Result {0} of {1}'.format(i, Result.table.raw.nrows)

        query = 'result_uuid=="{0}"'.format(result.uuid)
        if Classification.find_single(query):
            print 'Already classified'
            continue

        response = classify_result(result)
        if response is STOP:
            break

    # any tear down?

def classify_result(result):
    image = result.image

    mass = image.sum()
    if image.sum() < 1000:
        return CLASSES['no growth']
    print 'Mass', mass

    for row in range(image.shape[0]):
        if image[row, :].sum() > 0:
            max_row = row
        else:
            break

    plt.imshow(image[:max_row, :], interpolation='nearest')
    plt.pause(0.01)

    response = ask_for_class()
    if response is STOP:
        return STOP
    elif response is IGNORE:
        pass
    else:
        Classification(result_uuid=result.uuid, 
                       spec_uuid=result.spec_uuid,
                       class_=response).save()

def ask_for_class(num_tries=5):
    prompt = ', '.join('({1}) {0}'.format(*c) for c in CLASSES) + '=>'
    for _ in range(num_tries):
        raw = raw_input(prompt)
        if raw is 'q' or raw is 'quit':
            return STOP
        elif raw is 'i' or raw is 'ignore':
            return IGNORE

        try:
            index = int(raw)
            CLASSES(index) # throws on non-existance
            return index
        except ValueError:
            continue
        
    return IGNORE

if __name__ == '__main__':
    import sys
    h5_file = sys.argv[1]
    classify(util.results_h5_path(h5_file))