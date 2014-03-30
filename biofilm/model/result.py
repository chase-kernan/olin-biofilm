
from biofilm import util
from biofilm.model import spec as sp
import tables as tb

def from_model(model):
    return Result(spec_uuid=model.spec.uuid, image=model.render(), 
                  mass=model.mass_history)

def save_model(model):
    from_model(model).save()

class Result(util.TableObject):

    def __init__(self, uuid=None, spec_uuid=None, image=None, mass=None):
        util.TableObject.__init__(self, uuid)
        self.spec_uuid = spec_uuid
        self.image = image; self._exclude.add('image')
        self.mass = mass; self._exclude.add('mass')

    @property
    @util.memoized
    def spec(self): return sp.Spec.get(self.spec_uuid)

    @property
    @util.memoized
    def int_image(self): return self.image.astype(np.uint8)

    def _on_get(self, row):
        self.image = _get_image(self.uuid, self.spec.shape)
        self.mass = _get_mass(self.uuid, self.spec.stop_on_time)

    def _fill_row(self, row):
        util.TableObject._fill_row(self, row)
        _save_image(self.uuid, self.image)
        _save_mass(self.uuid, self.mass)

    def _on_delete(self):
        util.TableObject._on_delete(self)
        _delete_image(self.uuid, self.spec.shape)
        _delete_mass(self.uuid, self.spec.stop_on_time)

class ResultTable(tb.IsDescription):
    uuid = util.make_uuid_col()
    spec_uuid = tb.StringCol(32)
Result.setup_table("results", ResultTable, 
                   sorted_indices=['uuid', 'spec_uuid'])

_get_image, _save_image, _delete_image \
        = util.make_variable_data("_results_image", tb.BoolCol,
                                   filters=tb.Filters(complib='zlib', 
                                                      complevel=9))
_get_mass, _save_mass, _delete_mass \
        = util.make_variable_data("_results_mass", tb.UInt16Col,
                                   filters=tb.Filters(complib='zlib', 
                                                      complevel=9))
@util.memoized
def _get_image_table(size):
    class Image(util.EasyTableObject): pass
    class ImageTable(tb.IsDescription):
        uuid = util.make_uuid_col()
        image = tb.BoolCol(shape=size)

    name = "_result_images_{0}x{1}".format(*size)
    Image.setup_table(name, ImageTable,
                      filters=tb.Filters(complib='zlib', complevel=9))
    return Image

@util.memoized
def _get_mass_table(length):
    class Mass(util.EasyTableObject): pass
    class MassTable(tb.IsDescription):
        uuid = util.make_uuid_col()
        mass = tb.UInt16Col(shape=(1, length))

    name = "_result_mass_{0}".format(length)
    Mass.setup_table(name, MassTable,
                     filters=tb.Filters(complib='zlib', complevel=9))
    return Mass