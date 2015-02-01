'''
Created on Dec 29, 2012

@author: Chase Kernan
'''

import numpy as np
import cv2
import re
import random as rdm

from biofilm.model.media import probability as media_probability
from biofilm.model.light import probability as light_probability


def run(spec, **run_kwargs):
    return Model(spec).run(**run_kwargs)

ALIVE = 1
DEAD = 0

class Model(object):
    
    def __init__(self, spec):
        self.spec = spec
        self.num_cells = self.spec.shape
        self.min_dimension = min(self.num_cells)
        self.last_growth = 0
        self.time = 0
    
    def run(self, on_step=None):
        self.reset()
        self._place_cells_regularly()
        self.__max_height = 1

        should_stop = make_stopping_function(self.spec)
        while not should_stop(self):
            self.mass_history[self.time] = self.mass
            self.step()
            self.time += 1
            if on_step and on_step(self): break

        return self
    
    def reset(self):
        self.time = 0
        self.cells = np.zeros(self.num_cells, np.uint8)
        self.light = np.zeros(self.num_cells, float) # no need to reallocate
        self.division_probability = np.zeros(self.num_cells, float)
        self.dividing = np.zeros(self.num_cells, bool)
        self.surface_tension = np.zeros(self.num_cells, float)
        self.mass_history = np.zeros(self.spec.stop_on_time, np.uint16)
        
        self.__max_row = self.num_cells[0]-1
        self.__max_column = self.num_cells[1]-1
        self.__mass = 0
        self.__max_height = 1

        self._distance_kernel = generate_distance_kernel(self.spec.block_size)
        self._distance_kernel **= self.spec.distance_power
        self._distance_kernel /= self._distance_kernel.sum()
        self._tension_kernel = np.array([[1, 2, 1],
                                         [2, 0, 2],
                                         [1, 2, 1]], float)
        self._tension_kernel /= self._tension_kernel.sum()
        self._tension_min = self._tension_kernel[0:1, 0].sum()
        
        shape = self.spec.block_size, self.spec.block_size
        self._probability = np.empty(shape, np.float32)
        self._cumulative = np.empty(self._probability.size, np.float32)
        self._indices = np.arange(self._probability.size)
        self._cell_block = np.empty(shape, np.uint8)

    def render(self):
        return self.cells.astype(bool)
    
    @property
    def mass(self):
        return self.__mass
    
    @property
    def max_height(self):
        return self.__max_height
    
    def set_alive(self, row, column):
        row = max(0, min(self.__max_row, row))
        column = max(0, min(self.__max_column, column))
        
        if self.cells[row, column] == ALIVE:
            return
        
        self.cells[row, column] = ALIVE
        self.__mass += 1
        if row > self.__max_height:
            self.__max_height = row

        self.last_growth = self.time

    def step(self):
        if self.time - self.last_growth <= 1:
            self._calculate_division_probability()
        self._calculate_dividing_cells()
        self._divide()

    def _place_cells_regularly(self, spacing=None):
        if not spacing:
            spacing = self.spec.initial_cell_spacing
        
        start = int(spacing/2)
        end = self.num_cells[1]-int(spacing/2)
        for column in range(start, end+1, spacing):
            self.set_alive(0, column)

    def _calculate_surface_tension(self, center_factor=0):
        k = center_factor
        tension_kernel = np.array([[1, 2, 1],
                                   [2, k, 2],
                                   [1, 2, 1]], dtype=np.uint8)
        local_sum = cv2.filter2D(self.cells, -1, tension_kernel)
        self.surface_tension = local_sum/np.float(tension_kernel.sum())

    def _calculate_division_probability(self):
        prob = self.spec.division_rate*light_probability(self)

        media_prob = media_probability(self)
        prob *= media_prob[:prob.shape[0], :]

        prob[np.logical_not(self.cells[:prob.shape[0], :])] = 0
        self.division_probability = prob

    def _calculate_dividing_cells(self):
        self.dividing = np.random.ranf(self.division_probability.shape) <= \
                        self.division_probability

    def _divide(self):        
        block_size = self.spec.block_size # shortcut
        half_block = (block_size-1)/2
        
        rows, columns = self.dividing.nonzero()
        for i in range(len(rows)):
            row = rows[i]
            column = columns[i]

            write_block(self._cell_block, self.cells, row, column, block_size)
            cv2.filter2D(self._cell_block, cv2.CV_32F, self._tension_kernel,
                         self._probability, borderType=cv2.BORDER_CONSTANT)
            cv2.threshold(self._probability, self._tension_min, 0, 
                          cv2.THRESH_TOZERO, self._probability)
            self._probability[self._cell_block] = 0
            self._probability **= self.spec.tension_power
            self._probability *= self._distance_kernel
            
            # optimized version of np.random.choice
            np.cumsum(self._probability.flat, out=self._cumulative)
            total = self._cumulative[-1]
            if total < 1.0e-12:
                # no viable placements, we'll have precision problems anyways
                continue 
            self._cumulative /= total
            
            index = self._indices[np.searchsorted(self._cumulative, 
                                                  rdm.random())]
            local_row, local_column = np.unravel_index(index, 
                                                       self._probability.shape)
            self.set_alive(row+(local_row-half_block), 
                           column+(local_column-half_block))


def make_stopping_function(spec):
    clauses = []

    for full_name, value in spec.__dict__.iteritems():
        if value == 0: continue
        match = re.match(r"stop_on_(\w+)", full_name)
        if not match: continue
        name = match.group(1)

        if name == "mass":
            max_mass = int(value)
            clauses.append(lambda m: m.mass >= max_mass)
        elif name == "time":
            max_time = int(value)
            clauses.append(lambda m: m.time >= max_time)
        elif name == "height":
            max_height = int(value)
            clauses.append(lambda m: m.max_height >= max_height)
        elif name == 'no_growth':
            no_growth = int(value)
            clauses.append(lambda m: m.time >= m.last_growth+no_growth)
        else:
            raise sp.ParameterValueError("stop_on", spec.stop_on,
                    "No such stopping function {0}.".format(name))
    
    return lambda m: any(clause(m) for clause in clauses)


def generate_distance_kernel(size=7):
    kernel = np.empty((size, size), dtype=float)
    center = (size - 1)/2
    for row in range(size):
        for column in range(size):
            dx = row - center
            dy = column - center
            kernel[row, column] = dx**2 + dy**2
    kernel = np.sqrt(kernel)
    
    # avoid a 0 divide
    kernel[center, center] = 1.0
    kernel = 1.0/kernel
    kernel[center, center] = 0.0

    return kernel # we don't need to normalize here, we'll do it later

def write_block(block, matrix, row, column, block_size, filler=0):
    x = (block_size-1)/2
    left = max(0, column-x)
    right = min(matrix.shape[1]-1, column+x+1)
    top = max(0, row-x)
    bottom = min(matrix.shape[0]-1, row+x+1)
    
    block.fill(filler)
    block[x-(row-top) : x+(bottom-row),
          x-(column-left) : x+(right-column)] = matrix[top:bottom, left:right]

if __name__ == '__main__':
    from biofilm.model import spec
    from matplotlib import pyplot as plt
    run(spec.Spec())
