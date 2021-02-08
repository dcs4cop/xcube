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

import abc
import math
from typing import Tuple, Union, Dict

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from .base import GridMapping
from .helpers import _assert_valid_xy_names
from .helpers import _default_xy_var_names
from .helpers import _normalize_int_pair
from .helpers import _to_int_or_float
from .helpers import from_lon_360
from .helpers import round_to_fraction
from .helpers import to_lon_360


class CoordsGridMapping(GridMapping, abc.ABC):
    """Grid mapping constructed from 1D/2D coordinate variables and a CRS."""

    def __init__(self,
                 /,
                 x_coords: xr.DataArray,
                 y_coords: xr.DataArray,
                 **kwargs):
        self._x_coords = x_coords
        self._y_coords = y_coords
        super().__init__(**kwargs)


class Coords1DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from 1D coordinate variables and a CRS."""

    def _new_xy_coords(self) -> xr.DataArray:
        y, x = xr.broadcast(self._y_coords, self._x_coords)
        return xr.concat([x, y], dim='coord') \
            .chunk(self.xy_coords_chunks)


class Coords2DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from 2D coordinate variables and a CRS."""

    def _new_xy_coords(self) -> xr.DataArray:
        return xr.concat([self._x_coords, self._y_coords], dim='coord') \
            .chunk(self.xy_coords_chunks)


def new_grid_mapping_from_coords(x_coords: xr.DataArray,
                                 y_coords: xr.DataArray,
                                 crs: pyproj.crs.CRS,
                                 *,
                                 tile_size: Union[int, Tuple[int, int]] = None) -> GridMapping:
    assert_instance(x_coords, xr.DataArray, name='x_coords')
    assert_instance(y_coords, xr.DataArray, name='y_coords')
    assert_condition(x_coords.ndim in (1, 2),
                     'x_coords and y_coords must be either 1D or 2D arrays')
    assert_instance(crs, pyproj.crs.CRS, name='crs')

    if x_coords.name and y_coords.name:
        xy_var_names = str(x_coords.name), str(y_coords.name)
    else:
        xy_var_names = _default_xy_var_names(crs)

    tile_size = _normalize_int_pair(tile_size, default=None)
    is_lon_360 = None
    if crs.is_geographic:
        is_lon_360 = np.any(x_coords > 180)

    if x_coords.ndim == 1:
        assert_condition(x_coords.size >= 2 and y_coords.size >= 2,
                         'sizes of x_coords and y_coords 1D arrays must be >= 2')

        cls = Coords1DGridMapping
        size = x_coords.size, y_coords.size

        x_dim, y_dim = x_coords.dims[0], y_coords.dims[0]

        x_diff = _abs_no_zero(x_coords.diff(dim=x_dim).values)
        y_diff = _abs_no_zero(y_coords.diff(dim=y_dim).values)

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = np.any(np.nanmax(x_diff) > 180)
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x_diff = _abs_no_zero(x_coords.diff(dim=x_dim))
                is_lon_360 = True

        x_res, y_res = x_diff[0], y_diff[0]
        x_diff_equal = np.allclose(x_diff, x_res, atol=1.e-5)
        y_diff_equal = np.allclose(y_diff, y_res, atol=1.e-5)
        is_regular = x_diff_equal and y_diff_equal
        if is_regular:
            x_res = round_to_fraction(x_res, 5, 0.25)
            y_res = round_to_fraction(y_res, 5, 0.25)
        else:
            x_res = round_to_fraction(float(np.nanmedian(x_diff)), 2, 0.5)
            y_res = round_to_fraction(float(np.nanmedian(y_diff)), 2, 0.5)

        if tile_size is None \
                and x_coords.chunks is not None \
                and y_coords.chunks is not None:
            tile_size = max(0, *x_coords.chunks[0]), max(0, *y_coords.chunks[0])

        # Guess j axis direction
        is_j_axis_up = bool(y_coords[0] < y_coords[-1])

    else:
        assert_condition(x_coords.shape == y_coords.shape,
                         'shapes of x_coords and y_coords 2D arrays must be equal')
        assert_condition(x_coords.dims == y_coords.dims,
                         'dimensions of x_coords and y_coords 2D arrays must be equal')

        y_dim, x_dim = x_coords.dims

        cls = Coords2DGridMapping

        height, width = x_coords.shape
        size = width, height

        x = da.asarray(x_coords)
        y = da.asarray(y_coords)

        x_x_diff = _abs_no_nan(da.diff(x, axis=1))
        x_y_diff = _abs_no_nan(da.diff(x, axis=0))
        y_x_diff = _abs_no_nan(da.diff(y, axis=1))
        y_y_diff = _abs_no_nan(da.diff(y, axis=0))

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = da.any(da.max(x_x_diff) > 180) or \
                                       da.any(da.max(x_y_diff) > 180)
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x = da.asarray(x_coords)
                x_x_diff = _abs_no_nan(da.diff(x, axis=1))
                x_y_diff = _abs_no_nan(da.diff(x, axis=0))
                is_lon_360 = True

        is_regular = False

        if da.all(x_y_diff == 0) and da.all(y_x_diff == 0):
            x_res = x_x_diff[0, 0]
            y_res = y_y_diff[0, 0]
            is_regular = da.allclose(x_x_diff[0, :], x_res) and \
                         da.allclose(x_x_diff[-1, :], x_res) and \
                         da.allclose(y_y_diff[:, 0], y_res) and \
                         da.allclose(y_y_diff[:, -1], y_res)

        if not is_regular:
            # Let diff arrays have same shape as original by
            # doubling last rows and columns.
            x_x_diff_c = da.concatenate([x_x_diff, x_x_diff[:, -1:]], axis=1)
            y_x_diff_c = da.concatenate([y_x_diff, y_x_diff[:, -1:]], axis=1)
            x_y_diff_c = da.concatenate([x_y_diff, x_y_diff[-1:, :]], axis=0)
            y_y_diff_c = da.concatenate([y_y_diff, y_y_diff[-1:, :]], axis=0)
            # Find resolution via area
            x_abs_diff = da.sqrt(da.square(x_x_diff_c) + da.square(x_y_diff_c))
            y_abs_diff = da.sqrt(da.square(y_x_diff_c) + da.square(y_y_diff_c))
            xy_areas = (x_abs_diff * y_abs_diff).flatten()
            xy_areas = da.where(xy_areas > 0, xy_areas, np.inf)
            xy_area_index = da.argmin(xy_areas)
            xy_res = math.sqrt(xy_areas[xy_area_index])
            xy_res = round_to_fraction(xy_res, digits=2, resolution=0.5)
            x_res, y_res = float(xy_res), float(xy_res)

        if tile_size is None and x_coords.chunks is not None:
            j_chunks, i_chunks = x_coords.chunks
            tile_size = max(0, *i_chunks), max(0, *j_chunks)

        if tile_size is not None:
            tile_width, tile_height = tile_size
            x_coords = x_coords.chunk((tile_height, tile_width))
            y_coords = y_coords.chunk((tile_height, tile_width))

        # Guess j axis direction
        is_j_axis_up = np.all(y_coords[0, :] < y_coords[-1, :]) or None

    x_res, y_res = _to_int_or_float(x_res), _to_int_or_float(y_res)
    x_res_05, y_res_05 = x_res / 2, y_res / 2
    x_min = _to_int_or_float(x_coords.min() - x_res_05)
    y_min = _to_int_or_float(y_coords.min() - y_res_05)
    x_max = _to_int_or_float(x_coords.max() + x_res_05)
    y_max = _to_int_or_float(y_coords.max() + y_res_05)

    return cls(x_coords=x_coords,
               y_coords=y_coords,
               crs=crs,
               size=size,
               tile_size=tile_size,
               xy_bbox=(x_min, y_min, x_max, y_max),
               xy_res=(x_res, y_res),
               xy_var_names=xy_var_names,
               xy_dim_names=(str(x_dim), str(y_dim)),
               is_regular=is_regular,
               is_lon_360=is_lon_360,
               is_j_axis_up=is_j_axis_up)


def _abs_no_zero(array: Union[xr.DataArray, da.Array, np.ndarray]):
    array = np.fabs(array)
    return np.where(np.isclose(array, 0), np.nan, array)


def _abs_no_nan(array: Union[xr.DataArray, da.Array, np.ndarray]):
    array = np.fabs(array)
    return np.where(np.logical_or(np.isnan(array), np.isclose(array, 0)), 0, array)


def grid_mapping_to_coords(grid_mapping: GridMapping,
                           xy_var_names: Tuple[str, str] = None,
                           xy_dim_names: Tuple[str, str] = None,
                           exclude_bounds: bool = False) -> Dict[str, xr.DataArray]:
    """
    Get CF-compliant axis coordinate variables and cell boundary coordinate variables.

    Defined only for grid mappings with regular x,y coordinates.

    :param grid_mapping: A regular grid mapping.
    :param xy_var_names: Optional coordinate variable names (x_var_name, y_var_name).
    :param xy_dim_names: Optional coordinate dimensions names (x_dim_name, y_dim_name).
    :param exclude_bounds: If True, do not create bounds coordinates. Defaults to False.
    :return: dictionary with coordinate variables
    """

    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name='xy_var_names')
    if xy_dim_names:
        _assert_valid_xy_names(xy_dim_names, name='xy_dim_names')

    x_name, y_name = xy_var_names or grid_mapping.xy_var_names
    x_dim_name, y_dim_name = xy_dim_names or grid_mapping.xy_dim_names
    w, h = grid_mapping.size
    x1, y1, x2, y2 = grid_mapping.xy_bbox
    x_res, y_res = grid_mapping.xy_res
    x_res_05 = x_res / 2
    y_res_05 = y_res / 2

    dtype = np.float64

    x_data = np.linspace(x1 + x_res_05, x2 - x_res_05, w, dtype=dtype)
    if grid_mapping.is_lon_360:
        x_data = from_lon_360(x_data)

    if grid_mapping.is_j_axis_up:
        y_data = np.linspace(y1 + y_res_05, y2 - y_res_05, h, dtype=dtype)
    else:
        y_data = np.linspace(y2 - y_res_05, y1 + y_res_05, h, dtype=dtype)

    if grid_mapping.crs.is_geographic:
        x_attrs = dict(
            long_name='longitude coordinate',
            standard_name='longitude',
            units='degrees_east'
        )
        y_attrs = dict(
            long_name='latitude coordinate',
            standard_name='latitude',
            units='degrees_north'
        )
    else:
        x_attrs = dict(
            long_name="x coordinate of projection",
            standard_name="projection_x_coordinate"
        )
        y_attrs = dict(
            long_name="y coordinate of projection",
            standard_name="projection_y_coordinate"
        )

    x_coords = xr.DataArray(x_data, dims=x_dim_name, attrs=x_attrs)
    y_coords = xr.DataArray(y_data, dims=y_dim_name, attrs=y_attrs)
    coords = {
        x_name: x_coords,
        y_name: y_coords,
    }
    if not exclude_bounds:
        x_bnds_0_data = np.linspace(x1, x2 - x_res, w, dtype=dtype)
        x_bnds_1_data = np.linspace(x1 + x_res, x2, w, dtype=dtype)

        if grid_mapping.is_lon_360:
            x_bnds_0_data = from_lon_360(x_bnds_0_data)
            x_bnds_1_data = from_lon_360(x_bnds_1_data)

        if grid_mapping.is_j_axis_up:
            y_bnds_0_data = np.linspace(y1, y2 - y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y1 + y_res, y2, h, dtype=dtype)
        else:
            y_bnds_0_data = np.linspace(y2, y1 + y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y2 - y_res, y1, h, dtype=dtype)

        bnds_dim_name = 'bnds'
        x_bnds_name = f'{x_name}_{bnds_dim_name}'
        y_bnds_name = f'{y_name}_{bnds_dim_name}'
        # Note, according to CF, bounds variables are not required to have
        # any attributes, so we don't pass any.
        x_bnds_coords = xr.DataArray(list(zip(x_bnds_0_data, x_bnds_1_data)),
                                     dims=[x_dim_name, bnds_dim_name])
        y_bnds_coords = xr.DataArray(list(zip(y_bnds_0_data, y_bnds_1_data)),
                                     dims=[y_dim_name, bnds_dim_name])
        x_coords.attrs.update(bounds=x_bnds_name)
        y_coords.attrs.update(bounds=y_bnds_name)
        coords.update({
            x_bnds_name: x_bnds_coords,
            y_bnds_name: y_bnds_coords,
        })

    return coords
