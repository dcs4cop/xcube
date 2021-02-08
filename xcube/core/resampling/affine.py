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

import math
from typing import Union, Callable, Optional, Sequence, Tuple, Mapping, Hashable, Any

import numpy as np
import xarray as xr
from dask import array as da
from dask_image import ndinterp

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping import assert_regular_grid_mapping
from xcube.util.assertions import assert_condition

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]


def affine_transform_dataset(source_ds: xr.Dataset,
                             source_gm: GridMapping = None,
                             target_cm: GridMapping = None,
                             var_configs: Mapping[Hashable, Mapping[str, Any]] = None):
    if source_gm.crs != target_cm.crs:
        raise ValueError(f'CRS of source_gm and target_cm must be equal, '
                         f'was "{source_gm.crs.name}" and "{target_cm.crs.name}"')
    assert_regular_grid_mapping(source_gm, name='source_gm')
    assert_regular_grid_mapping(target_cm, name='target_cm')
    var_configs = var_configs or {}
    matrix = target_cm.ij_transform_to(source_gm)
    ((i_scale, _, i_off), (_, j_scale, j_off)) = matrix
    x_dim, y_dim = source_gm.xy_dim_names
    width, height = target_cm.size
    tile_width, tile_height = target_cm.tile_size
    yx_dims = (y_dim, x_dim)
    coords = dict()
    data_vars = dict()
    for k, var in source_ds.variables.items():
        new_var = None
        if var.ndim >= 2 and var.dims[-2:] == yx_dims:
            var_config = var_configs.get(k, dict())
            if np.issubdtype(var.dtype, int) or np.issubdtype(var.dtype, bool):
                spline_order = 0
                aggregator = None
                recover_nan = False
            else:
                spline_order = 1
                aggregator = np.nanmean
                recover_nan = True
            var_data = resample_ndimage(
                var.data,
                scale=(j_scale, i_scale),
                offset=(j_off, i_off),
                shape=(height, width),
                chunks=(tile_height, tile_width),
                spline_order=var_config.get('spline_order', spline_order),
                aggregator=var_config.get('aggregator', aggregator),
                recover_nan=var_config.get('recover_nan', recover_nan),
            )
            new_var = xr.DataArray(var_data, dims=var.dims, attrs=var.attrs)
        elif x_dim not in var.dims and y_dim not in var.dims:
            new_var = var.copy()
        if new_var is not None:
            if k in source_ds.coords:
                coords[k] = new_var
            elif k in source_ds.data_vars:
                data_vars[k] = new_var
    exclude_bounds = not source_ds[source_gm.xy_var_names[0]].attrs.get('bounds')
    coords.update(target_cm.to_coords(xy_var_names=source_gm.xy_var_names,
                                      xy_dim_names=source_gm.xy_dim_names,
                                      exclude_bounds=exclude_bounds))
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=source_ds.attrs)


def resample_ndimage(im: NDImage,
                     scale: Union[float, Tuple[float, float]] = 1,
                     offset: Union[float, Tuple[float, float]] = None,
                     shape: Union[int, Tuple[int, int]] = None,
                     chunks: Sequence[int] = None,
                     spline_order: int = 1,
                     aggregator: Optional[Aggregator] = np.nanmean,
                     recover_nan: bool = False) -> da.Array:
    im = da.asarray(im)
    offset = _normalize_offset(offset, im.ndim)
    scale = _normalize_scale(scale, im.ndim)
    if shape is None:
        shape = resize_shape(im.shape, scale)
    else:
        shape = _normalize_shape(shape, im)
    chunks = _normalize_chunks(chunks, shape)
    scale_y, scale_x = scale[-2], scale[-1]
    divisor_x = math.ceil(abs(scale_x))
    divisor_y = math.ceil(abs(scale_y))
    if (divisor_x >= 2 or divisor_y >= 2) and aggregator is not None:
        # Downsampling
        # ------------
        axes = {im.ndim - 2: divisor_y, im.ndim - 1: divisor_x}
        elongation = _normalize_scale((scale_y / divisor_y, scale_x / divisor_x), im.ndim)
        larger_shape = resize_shape(shape, (divisor_y, divisor_x),
                                    divisor_x=divisor_x, divisor_y=divisor_y)
        # print('Downsampling: ', scale)
        # print('  divisor:', (divisor_y, divisor_x))
        # print('  elongation:', elongation)
        # print('  shape:', shape)
        # print('  larger_shape:', larger_shape)
        divisible_chunks = _make_divisible_tiles(larger_shape, divisor_x, divisor_y)
        im = _transform_array(im,
                              elongation, offset,
                              larger_shape, divisible_chunks,
                              spline_order, recover_nan)
        im = da.coarsen(aggregator, im, axes)
        if shape != im.shape:
            im = im[..., 0:shape[-2], 0:shape[-1]]
        if chunks is not None:
            im = im.rechunk(chunks)
    else:
        # Upsampling
        # ----------
        # print('Upsampling: ', scale)
        im = _transform_array(im,
                              scale, offset,
                              shape, chunks,
                              spline_order, recover_nan)
    return im


def _transform_array(im: da.Array,
                     scale: Tuple[float, ...],
                     offset: Tuple[float, ...],
                     shape: Tuple[int, ...],
                     chunks: Optional[Tuple[int, ...]],
                     spline_order: int,
                     recover_nan: bool) -> da.Array:
    assert_condition(len(scale) == im.ndim, 'invalid scale')
    assert_condition(len(offset) == im.ndim, 'invalid offset')
    assert_condition(len(shape) == im.ndim, 'invalid shape')
    assert_condition(chunks is None or len(chunks) == im.ndim, 'invalid chunks')
    if _is_no_op(im, scale, offset, shape):
        return im
    matrix = scale
    at_kwargs = dict(
        offset=offset,
        order=spline_order,
        output_shape=shape,
        output_chunks=chunks,
        mode='constant',
    )
    if recover_nan and spline_order > 0:
        # We can "recover" values that are neighbours to NaN values
        # that would otherwise become NaN too.
        mask = da.isnan(im)
        # First check if there are NaN values ar all
        if da.any(mask):
            # Yes, then
            # 1. replace NaN by zero
            filled_im = da.where(mask, 0.0, im)
            # 2. transform the zeo-filled image
            scaled_im = ndinterp.affine_transform(filled_im, matrix, **at_kwargs, cval=0.0)
            # 3. transform the inverted mask
            scaled_norm = ndinterp.affine_transform(1.0 - mask, matrix, **at_kwargs, cval=0.0)
            # 4. put back NaN where there was zero, otherwise decode using scaled mask
            return da.where(da.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm)

    # No dealing with NaN required
    return ndinterp.affine_transform(im, matrix, **at_kwargs, cval=np.nan)


def resize_shape(shape: Sequence[int],
                 scale: Union[float, Tuple[float, ...]],
                 divisor_x: int = 1,
                 divisor_y: int = 1) -> Tuple[int, ...]:
    scale = _normalize_scale(scale, len(shape))
    height, width = shape[-2], shape[-1]
    scale_y, scale_x = scale[-2], scale[-1]
    wf = width * abs(scale_x)
    hf = height * abs(scale_y)
    w = divisor_x * math.ceil(wf / divisor_x)
    h = divisor_y * math.ceil(hf / divisor_y)
    return tuple(shape[0:-2]) + (h, w)


def _make_divisible_tiles(larger_shape: Tuple[int, ...],
                          divisor_x: int,
                          divisor_y: int) -> Tuple[int, ...]:
    w = min(larger_shape[-1], divisor_x * ((2048 + divisor_x - 1) // divisor_x))
    h = min(larger_shape[-2], divisor_y * ((2048 + divisor_y - 1) // divisor_y))
    return (len(larger_shape) - 2) * (1,) + (h, w)


def _normalize_image(im: NDImage) -> da.Array:
    return da.asarray(im)


def _normalize_offset(offset: Optional[Sequence[float]], ndim: int) -> Tuple[int, ...]:
    return _normalize_pair(offset, 0.0, ndim, 'offset')


def _normalize_scale(scale: Optional[Sequence[float]], ndim: int) -> Tuple[int, ...]:
    return _normalize_pair(scale, 1.0, ndim, 'scale')


def _normalize_pair(pair: Optional[Sequence[float]], default: float, ndim: int, name: str) -> Tuple[int, ...]:
    if pair is None:
        pair = [default, default]
    elif isinstance(pair, (int, float)):
        pair = [pair, pair]
    elif len(pair) != 2:
        raise ValueError(f'illegal image {name}')
    return (ndim - 2) * (default,) + tuple(pair)


def _normalize_shape(shape: Optional[Sequence[int]], im: NDImage) -> Tuple[int, ...]:
    if shape is None:
        return im.shape
    if len(shape) != 2:
        raise ValueError('illegal image shape')
    return im.shape[0:-2] + tuple(shape)


def _normalize_chunks(chunks: Optional[Sequence[int]],
                      shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if chunks is None:
        return None
    if len(chunks) < 2 or len(chunks) > len(shape):
        raise ValueError('illegal image chunks')
    return (len(shape) - len(chunks)) * (1,) + tuple(chunks)


def _is_no_op(im: NDImage,
              scale: Sequence[float],
              offset: Sequence[float],
              shape: Tuple[int, ...]):
    return shape == im.shape \
           and all(math.isclose(s, 1) for s in scale) \
           and all(math.isclose(o, 0) for o in offset)
