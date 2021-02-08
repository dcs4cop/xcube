# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union, Tuple

import pyproj
import xarray as xr

from .base import GridMapping
from .cfconv import get_dataset_grid_mappings
from .coords import new_grid_mapping_from_coords


def new_grid_mapping_from_dataset(dataset: xr.Dataset,
                                  *,
                                  xy_var_names: Tuple[str, str] = None,
                                  tile_size: Union[int, Tuple[str, str]] = None,
                                  prefer_crs: pyproj.crs.CRS = None,
                                  prefer_is_regular: bool = None,
                                  emit_warnings: bool = False) -> Optional[GridMapping]:
    if xy_var_names is not None:
        x_var_name, y_var_name = xy_var_names
        if x_var_name not in dataset or y_var_name not in dataset:
            raise ValueError(f'coordinate variables "{x_var_name}" '
                             f'or "{y_var_name}" not found in dataset')
        # TODO: create new instance using named coordinate variables
        raise NotImplementedError('xy_var_names not yet supported')

    grid_mappings = get_dataset_grid_mappings(dataset,
                                              emit_warnings=emit_warnings).values()
    grid_mappings = [new_grid_mapping_from_coords(x_coords=gm.coords.x,
                                                  y_coords=gm.coords.y,
                                                  crs=gm.crs,
                                                  tile_size=tile_size)
                     for gm in grid_mappings]

    if len(grid_mappings) > 1:
        if prefer_crs is not None and prefer_is_regular is not None:
            for gm in grid_mappings:
                if gm.crs == prefer_crs and gm.is_regular == prefer_is_regular:
                    return gm

        if prefer_crs is not None:
            for gm in grid_mappings:
                if gm.crs == prefer_crs:
                    return gm

        if prefer_is_regular is not None:
            for gm in grid_mappings:
                if gm.is_regular == prefer_is_regular:
                    return gm

    # Get arbitrary one (here: first)
    if grid_mappings:
        return grid_mappings[0]

    raise ValueError('cannot find any grid mapping in dataset')
