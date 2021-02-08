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
from typing import Tuple, Union

import dask.array as da
import pyproj
import xarray as xr

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from .base import GridMapping
from .helpers import _default_xy_dim_names
from .helpers import _default_xy_var_names
from .helpers import _normalize_int_pair
from .helpers import _normalize_number_pair
from .helpers import _to_int_or_float


class RegularGridMapping(GridMapping):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xy_coords = None

    def _new_xy_coords(self) -> xr.DataArray:
        self._assert_regular()
        x_res_05, y_res_05 = self.x_res / 2, self.y_res / 2
        x1, x2 = self.x_min + x_res_05, self.x_max - x_res_05
        y1, y2 = self.y_min + y_res_05, self.y_max - y_res_05
        if not self.is_j_axis_up:
            y1, y2 = y2, y1
        x_name, y_name = self.xy_dim_names
        x_coords_1d = xr.DataArray(da.linspace(x1, x2, self.width, chunks=self.tile_width), dims=x_name)
        y_coords_1d = xr.DataArray(da.linspace(y1, y2, self.height, chunks=self.tile_height), dims=y_name)
        y_coords_2d, x_coords_2d = xr.broadcast(y_coords_1d, x_coords_1d)
        xy_coords = xr.concat([x_coords_2d, y_coords_2d], dim='coord').chunk((2, self.tile_height, self.tile_width))
        xy_coords.name = 'xy_coords'
        return xy_coords


def new_regular_grid_mapping(size: Union[int, Tuple[int, int]],
                             xy_min: Tuple[float, float],
                             xy_res: Union[float, Tuple[float, float]],
                             crs: pyproj.crs.CRS,
                             *,
                             tile_size: Union[int, Tuple[int, int]] = None,
                             is_j_axis_up: bool = False) -> GridMapping:
    width, height = _normalize_int_pair(size, name='size')
    assert_condition(width > 1 and height > 1, 'invalid size')

    x_min, y_min = _normalize_number_pair(xy_min, name='xy_min')

    x_res, y_res = _normalize_number_pair(xy_res, name='xy_res')
    assert_condition(x_res > 0 and y_res > 0, 'invalid xy_res')

    assert_instance(crs, pyproj.crs.CRS, name='crs')

    x_min = _to_int_or_float(x_min)
    y_min = _to_int_or_float(y_min)
    x_max = _to_int_or_float(x_min + x_res * width)
    y_max = _to_int_or_float(y_min + y_res * height)

    if crs.is_geographic:
        # TODO: don't do that.
        #  Instead set NaN in coord vars returned by to_coords()
        if y_min < -90:
            raise ValueError('invalid y_min')
        if y_max > 90:
            raise ValueError('invalid size, y_min combination')

    return RegularGridMapping(crs=crs,
                              size=(width, height),
                              tile_size=tile_size or (width, height),
                              xy_bbox=(x_min, y_min, x_max, y_max),
                              xy_res=(x_res, y_res),
                              xy_var_names=_default_xy_var_names(crs),
                              xy_dim_names=_default_xy_dim_names(crs),
                              is_regular=True,
                              is_lon_360=x_max > 180,
                              is_j_axis_up=is_j_axis_up)


def to_regular_grid_mapping(grid_mapping: GridMapping,
                            *,
                            tile_size: Union[int, Tuple[int, int]] = None,
                            is_j_axis_up: bool = False) -> GridMapping:
    if grid_mapping.is_regular:
        if tile_size is not None or is_j_axis_up != grid_mapping.is_j_axis_up:
            return grid_mapping.derive(tile_size=tile_size, is_j_axis_up=is_j_axis_up)
        return grid_mapping

    x_min, y_min, x_max, y_max = grid_mapping.xy_bbox
    x_res, y_res = grid_mapping.xy_res
    # x_digits = 2 + abs(round(math.log10(x_res)))
    # y_digits = 2 + abs(round(math.log10(y_res)))
    # x_min = floor_to_fraction(x_min, x_digits)
    # y_min = floor_to_fraction(y_min, y_digits)
    # x_max = ceil_to_fraction(x_max, x_digits)
    # y_max = ceil_to_fraction(y_max, y_digits)
    xy_res = min(x_res, y_res) or max(x_res, y_res)
    width = round((x_max - x_min + xy_res) / xy_res)
    height = round((y_max - y_min + xy_res) / xy_res)
    width = width if width >= 2 else 2
    height = height if height >= 2 else 2

    return new_regular_grid_mapping(size=(width, height),
                                    xy_min=(x_min, y_min),
                                    xy_res=xy_res,
                                    crs=grid_mapping.crs,
                                    tile_size=tile_size,
                                    is_j_axis_up=is_j_axis_up)
