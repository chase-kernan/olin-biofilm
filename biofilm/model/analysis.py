'''
Created on Dec 30, 2012

@author: chase_000
'''

import numpy as np
import cv2
from biofilm.model import spec as sp
import tables as tb
from biofilm import util
from biofilm.model import result
from scipy import interpolate
from itertools import chain
from itertools import groupby

try:
    from matplotlib import pyplot as plt
    plt.set_cmap('hot') # ugh, rainbow cmaps...
except ImportError: pass

class Field(object):

    def __init__(self, func=None, path='', column=tb.Float32Col(), 
                 **table_args):
        self.func = func
        self.path = path
        self._make_tables(column, table_args)
    
    def _make_tables(self, column, table_args):
        self._make_by_result(column, table_args)
        self._make_by_spec()

    def _make_by_result(self, column, table_args):
        class ByResult(util.EasyTableObject): pass
        class ByResultTable(tb.IsDescription):
            uuid = util.make_uuid_col()
            data = column
        ByResult.setup_table(self.path, ByResultTable, 
                             expectedrows=result.Result.table.raw.nrows,
                             **table_args)
        self._by_result = ByResult

    def _make_by_spec(self):
        class BySpec(util.EasyTableObject): pass
        class BySpecTable(tb.IsDescription):
            uuid = util.make_uuid_col()
            mean = tb.Float32Col()
            median = tb.Float32Col()
            std = tb.Float32Col()
            max = tb.Float32Col()
            min = tb.Float32Col()
        BySpec.setup_table(self.path + "_by_spec", BySpecTable, 
                           filters=tb.Filters(complib='blosc', complevel=1),
                           expectedrows=sp.Spec.table.raw.nrows)
        self._by_spec = BySpec

    def reset(self):
        self._by_result.table.reset()
        self._by_spec.table.reset()

    def get_by_result(self, result, **compute_args):
        try:
            return self._by_result.get(result.uuid).data
        except KeyError:
            return self.compute_by_result(result, **compute_args)

    def compute_by_result(self, result, flush=True):
        data = self.func(result)
        self._by_result(uuid=result.uuid, data=data).save(flush=flush)
        return data

    def delete_by_result(self, result):
        self._by_result(uuid=result.uuid).delete()

    def get_by_spec(self, spec, **compute_args):
        try:
            data = self._by_spec.get(spec.uuid)
            return dict(mean=data.mean, median=data.median, std=data.std,
                        max=data.max, min=data.min)
        except KeyError:
            return self.compute_by_spec(spec, **compute_args)

    def _wrap(self, data):
        summary = util.row_to_dict(data)
        del summary['uuid']
        return summary

    def compute_by_spec(self, spec, recompute=False, flush=True):
        matching = "spec_uuid=='{0}'".format(spec.uuid)
        func = self.compute_by_result if recompute else self.get_by_result
        data = np.array([func(res) for res in result.Result.where(matching)])
        summary = self._summarize(data)
        self._by_spec(uuid=spec.uuid, **summary).save(flush=flush)
        return summary

    def delete_by_spec(self, spec):
        self._by_spec(uuid=spec.uuid).delete()

    def compute_specs(self, query=None, print_interval=100, **compute_kwargs):
        if query is None:
            print sp.Spec.table.raw.nrows
            specs = sp.Spec.all()
        else:
            specs = sp.Spec.where(query)
        for i, spec in enumerate(specs):
            if print_interval and i % print_interval == 0:
                print i,
            self.compute_by_spec(spec, **compute_kwargs)

    def _summarize(self, data):
        if data.size > 0:
            return dict(mean=data.mean(), median=np.median(data),
                        std=data.std(), max=data.max(), min=data.min())
        else:
            print "WARNING: Empty data!"
            return dict(mean=0.0, median=0.0, std=0.0, max=0.0, min=0.0)

    def plot(self, parameter, spec_query=None, statistic='mean', show=False, 
             **plot_args):
        specs = self._query_specs(spec_query)
        
        shape = len(specs), 1
        xs = np.empty(shape, float)
        ys = np.empty(shape, float)
        
        for i, spec in enumerate(specs):
            xs[i] = float(getattr(spec, parameter))
            ys[i] = self._get_statistic(spec, statistic)

        plt.plot(xs, ys, 'rx', **plot_args)
        plt.xlabel(parameter)
        plt.ylabel(self.path)
        if show: plt.show()

    def phase_diagram_2d(self, parameter1, parameter2, num_cells=50, 
                         spec_query=None, statistic='mean', show=False, 
                         **plot_args):
        specs = self._query_specs(spec_query)
        
        shape = len(specs), 1
        xs = np.empty(shape, float)
        ys = np.empty(shape, float)
        values = np.empty(shape, float)
        
        for i, spec in enumerate(specs):
            xs[i] = float(getattr(spec, parameter1))
            ys[i] = float(getattr(spec, parameter2))
            values[i] = self._get_statistic(spec, statistic)
        
        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()
        
        assert xMin != xMax
        assert yMin != yMax

        try:
            num_x, num_y = num_cells
        except TypeError:
            num_x, num_y = num_cells, num_cells
        
        grid = np.mgrid[xMin:xMax:num_x*1j, 
                        yMin:yMax:num_y*1j]
        interp = interpolate.griddata(np.hstack((xs, ys)), 
                                      values, 
                                      np.vstack((grid[0].flat, grid[1].flat)).T, 
                                      'linear')
        valueGrid = np.reshape(interp, grid[0].shape)
        
        plt.pcolormesh(grid[0], grid[1], valueGrid, **plot_args)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.xlabel(parameter1)
        plt.ylabel(parameter2)
        plt.colorbar()
        plt.title(self.path)
        if show: plt.show()

    def contour_plot(self, parameter1, parameter2, num_cells=50, 
                     spec_query=None, statistic='mean', show=False, 
                     smoothing=None, **plot_args):
        specs = self._query_specs(spec_query)
        
        shape = len(specs), 1
        xs = np.empty(shape, float)
        ys = np.empty(shape, float)
        values = np.empty(shape, float)
        
        for i, spec in enumerate(specs):
            xs[i] = float(getattr(spec, parameter1))
            ys[i] = float(getattr(spec, parameter2))
            values[i] = self._get_statistic(spec, statistic)
        
        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()
        
        assert xMin != xMax
        assert yMin != yMax
        
        grid = np.mgrid[xMin:xMax:num_cells*1j, 
                        yMin:yMax:num_cells*1j]
        interp = interpolate.griddata(np.hstack((xs, ys)), 
                                      values, 
                                      np.vstack((grid[0].flat, grid[1].flat)).T, 
                                      'cubic')
        valueGrid = np.reshape(interp, grid[0].shape)

        #try:
        #    valueGrid.clip(plot_args['vmin'], plot_args['vmax'], out=valueGrid)
        #except: KeyError

        if smoothing is not None:
            #from scipy.ndimage.filters import gaussian_filter
            #gaussian_filter(valueGrid, smoothing, output=valueGrid)
            from scipy.ndimage.interpolation import zoom
            gx = zoom(grid[0], smoothing)
            gy = zoom(grid[1], smoothing)
            valueGrid = zoom(valueGrid, smoothing)
        else:
            gx, gy = grid[0], grid[1]
        
        contour = plt.contour(gx, gy, valueGrid, **plot_args)
        plt.clabel(contour, inline=True, fontsize=10)
        plt.grid(True)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.xlabel(parameter1)
        plt.ylabel(parameter2)
        #plt.colorbar()
        plt.title(self.path)
        if show: plt.show()

    def scatter_plot(self, y_field, spec_query=None, statistic='mean',
                     show=False, **plot_args):
        specs = self._query_specs(spec_query)
        print '# specs = ', len(specs)

        if statistic == 'all':
            xs = []
            ys = []

            for spec in specs:
                matching = "spec_uuid=='{0}'".format(spec.uuid)
                for res in result.Result.where(matching):
                    xs.append(self.get_by_result(res))
                    ys.append(y_field.get_by_result(res))

            xs = np.array(xs)
            ys = np.array(ys)
        else:
            xs = np.empty(len(specs), float)
            ys = np.empty_like(xs)

            for i, spec in enumerate(specs):
                xs[i] = self._get_statistic(spec, statistic)
                ys[i] = y_field._get_statistic(spec, statistic)

        plt.plot(xs, ys, '.', **plot_args)
        plt.xlabel(self.path)
        plt.ylabel(y_field.path)
        if show: plt.show()

        return xs, ys

    def _query_specs(self, spec_query):
        if spec_query:
            return list(sp.Spec.where(spec_query))
        else:
            return list(sp.Spec.all())

    def _get_statistic(self, spec, statistic):
        return self.get_by_spec(spec)[statistic]

class VariableField(Field):

    def __init__(self, shape=lambda r: 1, **field_args):
        Field.__init__(self, **field_args)
        self.shape = shape

    def _make_by_result(self, column, table_args):
        self._get_by_result, self._save_by_result, self._delete_by_result \
                = util.make_variable_data(self.path, column, **table_args)

    def reset(self):
        raise NotImplemented()

    def get_by_result(self, result):
        try:
            return self._get_by_result(result.uuid, self.shape(result))
        except KeyError:
            return self.compute_by_result(result)

    def compute_by_result(self, result):
        data = self.func(result)
        self._save_by_result(result.uuid, data)
        return data

    def delete_by_result(self, result):
        self._delete_by_result(result.uuid, self.shape(result))

    def _summarize(self, data):
        combined = list(chain.from_iterable(data))
        return Field._summarize(self, np.array(combined))

class CurveAveragingField(VariableField):

    def __init__(self, spec_shape=lambda s: 1, **var_args):
        self.spec_shape = spec_shape
        VariableField.__init__(self, **var_args)

    def _make_by_spec(self):
        self._get_by_spec, self._save_by_spec, self._delete_by_spec \
                = util.make_variable_data(self.path + "_by_spec", 
                                           tb.Float32Col)

    def get_by_spec(self, spec):
        try:
            return self._get_by_spec(spec.uuid, self.spec_shape(spec))
        except KeyError:
            return self.compute_by_spec(spec)

    def compute_by_spec(self, spec):
        matching = "spec_uuid=='{0}'".format(spec.uuid)
        data = [self.compute_by_result(res) 
                for res in result.Result.where(matching)]
        summary = self._summarize(data)
        self._save_by_spec(spec.uuid, summary)
        return summary

    def delete_by_result(self, spec):
        self._delete_by_result(result.uuid, self.spec_shape(spec))

    def _summarize(self, data):
        if not data: return None

        averaged = np.copy(data[0])
        for i in range(1, len(data)):
            averaged += data[i]
        return averaged/float(len(data))

    def curve_plot(self, spec_query=None, show=False):
        specs = self._query_specs(spec_query)

        plt.hold(True)
        for spec in specs:
            plt.plot(self.get_by_spec(spec))

        plt.ylabel(self.path)
        if show: plt.show()

    def average_curve_plot(self, spec_query=None, show=False, **plot_args):
        specs = self._query_specs(spec_query)
        data = [self.get_by_spec(spec) for spec in specs]
        plt.plot(self._summarize(data), **plot_args)
        plt.ylabel(self.path)
        if show: plt.show()

# TODO: convert Field into a function @descriptor

def _compute_mass(result):
    return result.image.sum()
mass = Field(func=_compute_mass, path="mass", column=tb.UInt32Col())


def _compute_heights(result):
    return util.compute_heights(result.image)
heights = VariableField(func=_compute_heights,
                        shape=lambda r: r.image.shape[1],
                        path="heights",
                        column=tb.UInt16Col)

def _compute_mean_horizontal_runs(result):
    image = result.image
    max_height = heights.get_by_result(result).max()

    runs = np.zeros(image.shape[0], float)
    for row in range(max_height):
        row_runs = [len(list(group)) for cell, group in groupby(image[row, :])
                    if cell > 0]
        runs[row] = np.mean(row_runs)
    return runs
mean_horizontal_runs = VariableField(func=_compute_mean_horizontal_runs,
                                     shape=lambda r: r.image.shape[0],
                                     path="mean_horizontal_runs",
                                     column=tb.Float32Col)

def _compute_run_ratio(result):
    max_height = heights.get_by_result(result).max()
    step = int(max_height/8.0)
    if step < 1: return 0.0

    runs = mean_horizontal_runs.get_by_result(result)
    norm = runs[2*step:3*step].mean()
    if norm < 1: return 0.0

    canopy = runs[5*step:6*step].mean()
    return canopy/norm
run_ratio = Field(func=_compute_run_ratio,  path="run_ratio",
                  column=tb.Float32Col())

def _compute_canopy_coverage(result):
    max_height = heights.get_by_result(result).max()
    #step = int(max_height/8.0)
    #if step < 1: return 0.0

    covs = coverages.get_by_result(result)
    neck = covs[:4].mean()
    canopy = covs[8:16].mean()
    return canopy/neck
canopy_coverage = Field(func=_compute_canopy_coverage,  path="canopy_coverage",
                        column=tb.Float32Col())

def _compute_dropoff(result):
    cvrg = coverages.get_by_result(result)
    max_coverage = cvrg.max()
    max_coverage_height = cvrg.argmax()
    max_height = heights.get_by_result(result).max()
    return max_coverage/(max_height - max_coverage_height)
dropoff = Field(func=_compute_dropoff, path="dropoff", column=tb.Float32Col())

def _compute_mean_coverage(result):
    cvrg = coverages.get_by_result(result)
    max_height = heights.get_by_result(result).max()
    return cvrg[:max_height].mean()
mean_coverage = Field(func=_compute_mean_coverage, 
                      path="mean_coverage", 
                      column=tb.Float32Col())

def _compute_first_order_coverage(result):
    cvrg = coverages.get_by_result(result)
    max_height = heights.get_by_result(result).max()
    return np.polyfit(range(max_height), cvrg[:max_height], 1)[0]
first_order_coverage = Field(func=_compute_first_order_coverage, 
                             path="first_order_coverage", 
                             column=tb.Float32Col())

def _compute_second_order_coverage(result):
    cvrg = coverages.get_by_result(result)
    max_height = heights.get_by_result(result).max()
    return np.polyfit(range(max_height), cvrg[:max_height], 2)[0]
second_order_coverage = Field(func=_compute_second_order_coverage, 
                              path="second_order_coverage", 
                              column=tb.Float32Col())

def _compute_contours(int_image):
    # findContours clips the borders of an image
    larger_image = np.zeros([x+2 for x in int_image.shape], int_image.dtype)
    larger_image[1:-1, 1:-1] = int_image
    return cv2.findContours(larger_image, 
                            cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_SIMPLE)[0]

def _compute_perimeter(result):
    return sum(cv2.arcLength(c, True) for c in 
               _compute_contours(result.int_image))\
           /float(result.spec.width)

perimeter = Field(func=_compute_perimeter, path="perimeter")

def _compute_roughness(result):
    #h = heights.get_by_result(result)
    #height_diff = abs(h - h.mean()).sum()
    height_diff = heights.get_by_result(result).std()
    if height_diff > 1e-4:
        return perimeter.get_by_result(result)/height_diff
    else:
        return 0
roughness = Field(func=_compute_roughness, path="roughness")

def _compute_coverages(result):
    return result.image.sum(axis=1)/float(result.spec.width)
coverages = CurveAveragingField(func=_compute_coverages,
                                shape=lambda r: r.spec.height,
                                spec_shape=lambda s: s.height,
                                path="coverages",
                                column=tb.Float32Col)

def _compute_mean_cell_height(result):
    height_sums = result.image.sum(axis=1)
    return np.average(np.arange(len(height_sums)), weights=height_sums)
mean_cell_height = Field(func=_compute_mean_cell_height, path="mean_cell_height")

def _compute_mean_cell_height_ratio(result):
    cell_height = mean_cell_height.get_by_result(result)
    return cell_height/heights.get_by_result(result).mean()
mean_cell_height_ratio = Field(func=_compute_mean_cell_height_ratio, path="mean_cell_height_ratio")

def _compute_overhangs(result):
    oh = np.zeros(result.spec.width, np.uint16)
    empty_count = np.zeros_like(oh)
    max_height = util.compute_heights(result.image).max()
    for row in range(max_height):
        alive = result.image[row]
        oh += empty_count*alive
        empty_count += 1
        empty_count[alive] = 0

    return oh
overhangs = VariableField(func=_compute_overhangs,
                          shape=lambda r: r.spec.width,
                          path="overhangs",
                          column=tb.UInt16Col)

def _compute_mean_overhang(result):
    oh = overhangs.get_by_result(result)
    return oh.mean()/result.image.sum()
mean_overhang = Field(func=_compute_mean_overhang,
                      path="mean_overhang",
                      column=tb.Float32Col())

def _compute_x_correlations(result):
    distances = range(1, result.spec.width/2)
    x_correlations = np.zeros((result.spec.height, len(distances)), float)
    found = np.empty(len(distances), int)
    count = np.empty_like(found)

    for row in range(get_height_dist(result)['max']):
        found.fill(0)
        count.fill(0)

        for col in range(result.spec.width):
            cell = result.image[row, col]

            for i, distance in enumerate(distances):
                for direction in [-1, 1]:
                    offset = col + distance*direction
                    if offset < 0 or offset >= result.spec.width:
                        continue

                    count[i] += 1
                    if result.image[row, offset] == cell:
                        found[i] += 1

        x_correlations[row, :] = found.astype(float)/count

    return x_correlations
x_correlations = VariableField(func=_compute_x_correlations,
                               shape=lambda r: (r.spec.height, r.spec.width/2),
                               path="x_correlations",
                               column=tb.Float32Col)

def _compute_convex_hull_area(result):
    area = 0.0
    for contour in _compute_contours(result.int_image):
        try:
            hull = cv2.convexHull(contour, returnPoints=True)
            area += cv2.contourArea(hull)
        except:
            continue
    return area
convex_hull_area = Field(func=_compute_convex_hull_area,
                         path="convex_hull_area")


def _compute_convex_density(result):
    area = float(convex_hull_area.get_by_result(result))
    if area <= 1.0:
        return 0.0 
    else:
        return min(1.0, mass.get_by_result(result)/area)
convex_density = Field(func=_compute_convex_density, path="convex_density")

def _compute_density(result):
    area = float(result.spec.width*heights.get_by_result(result).max())
    return 0.0 if area <= 1.0 else mass.get_by_result(result)/area
density = Field(func=_compute_density, path="density")

def _compute_light_exposure(result):
    penetration_depth = 6.0 #result.spec.light_penetration
    cum_sum = result.image.cumsum(axis=0)
    light = np.exp(-cum_sum/penetration_depth)
    return (light*result.image).sum()
light_exposure = Field(func=_compute_light_exposure, path="light_exposure")

def _compute_horizontal_surface_area(result):
    # contours = [cv2.approxPolyDP(c, 2, True) for c in _compute_contours(result.int_image)]
    contours = _compute_contours(result.int_image)
    vectors = [np.diff(c, axis=0) for c in contours]

    positive = sum(np.abs(v[:, 0, 0]).sum() for v in vectors)
    total = sum(v[:, 0, 0].sum() for v in vectors)
    return float(positive - total)/result.int_image.shape[1]

horizontal_surface_area = Field(func=_compute_horizontal_surface_area, 
                                path='horizontal_surface_area')


def _compute_b2r(result):
    return (result.spec.boundary_layer**2)*result.spec.media_ratio
b2r = Field(func=_compute_b2r, path='b2r')

def _compute_recip_b2r(result):
    return 1.0/((result.spec.boundary_layer**2)*result.spec.media_ratio)
recip_b2r = Field(func=_compute_recip_b2r, path='recip_b2r')

def _compute_br(result):
    return (result.spec.boundary_layer)*result.spec.media_ratio
br = Field(func=_compute_br, path='br')

def _compute_growth_time(result):
    indices = np.nonzero(result.mass == 0)[0]
    return indices[0] if indices.size > 0 else result.mass.size
growth_time = Field(func=_compute_growth_time, 
                    path='growth_time', 
                    column=tb.UInt32Col())
