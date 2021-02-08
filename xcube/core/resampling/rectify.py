# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from typing import Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.select import select_spatial_subset
from xcube.util.dask import compute_array_from_func


def rectify_dataset(source_ds: xr.Dataset,
                    *,
                    var_names: Union[str, Sequence[str]] = None,
                    source_gm: GridMapping = None,
                    xy_var_names: Tuple[str, str] = None,
                    target_gm: GridMapping = None,
                    tile_size: Union[int, Tuple[int, int]] = None,
                    is_j_axis_up: bool = None,
                    output_ij_names: Tuple[str, str] = None,
                    compute_subset: bool = True,
                    uv_delta: float = 1e-3) -> Optional[xr.Dataset]:
    """
    Reproject dataset *source_ds* using its per-pixel x,y coordinates or the
    given *source_gm*.

    The function expects *source_ds* or the given *source_gm* to have either
    one- or two-dimensional coordinate variables that provide spatial x,y
    coordinates for every data variable with the same spatial dimensions.

    For example, a dataset may comprise variables with spatial dimensions
    ``var(..., y_dim, x_dim)``, then one the function expects coordinates to
    be provided in two forms:

    1. One-dimensional ``x_var(x_dim)`` and ``y_var(y_dim)`` (coordinate)
       variables.
    2. Two-dimensional ``x_var(y_dim, x_dim)`` and ``y_var(y_dim, x_dim)``
       (coordinate) variables.

    If *target_gm* is given and it defines a tile size or *tile_size* is
    given, and the number of tiles is greater than one in the output's
    x- or y-direction, then the returned dataset will be composed of lazy,
    chunked dask arrays. Otherwise the returned dataset will be composed
    of ordinary numpy arrays.

    :param source_ds: Source dataset grid mapping.
    :param var_names: Optional variable name or sequence of variable names.
    :param source_gm: Target dataset grid mapping.
    :param xy_var_names: Optional tuple of the x- and y-coordinate variables in *source_ds*.
        Ignored if *source_gm* is given.
    :param target_gm: Optional output geometry. If not given, output geometry will be computed
        to spatially fit *dataset* and to retain its spatial resolution.
    :param tile_size: Optional tile size for the output.
    :param is_j_axis_up: Whether y coordinates are increasing with positive image j axis.
    :param output_ij_names: If given, a tuple of variable names in which to store the computed source pixel
        coordinates in the returned output.
    :param compute_subset: Whether to compute a spatial subset from *dataset* using *output_geom*. If set,
        The function may return ``None`` in case there is no overlap.
    :param uv_delta: A normalized value that is used to determine whether x,y coordinates in the output are contained
        in the triangles defined by the input x,y coordinates.
        The higher this value, the more inaccurate the rectification will be.
    :return: a reprojected dataset, or None if the requested output does not intersect with *dataset*.
    """
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds, xy_var_names=xy_var_names)

    src_attrs = dict(source_ds.attrs)

    if target_gm is None:
        target_gm = source_gm.to_regular(tile_size=tile_size)
    elif compute_subset:
        source_ds_subset = select_spatial_subset(source_ds,
                                                 xy_bbox=target_gm.xy_bbox,
                                                 ij_border=1,
                                                 xy_border=0.5 * (target_gm.x_res + target_gm.y_res),
                                                 geo_coding=source_gm)
        if source_ds_subset is None:
            return None
        if source_ds_subset is not source_ds:
            # TODO: GridMapping.from_dataset() may be expensive. Find a more effective way.
            source_gm = GridMapping.from_dataset(source_ds_subset)
            source_ds = source_ds_subset

    # if src_geo_coding.xy_var_names != output_geom.xy_var_names:
    #     output_geom = output_geom.derive(xy_var_names=src_geo_coding.xy_var_names)
    # if src_geo_coding.xy_dim_names != output_geom.xy_dim_names:
    #     output_geom = output_geom.derive(xy_dim_names=src_geo_coding.xy_dim_names)
    if tile_size is not None or is_j_axis_up is not None:
        target_gm = target_gm.derive(tile_size=tile_size, is_j_axis_up=is_j_axis_up)

    src_vars = _select_variables(source_ds, source_gm, var_names)

    if target_gm.is_tiled:
        compute_dst_src_ij_images = _compute_ij_images_xarray_dask
        compute_dst_var_image = _compute_var_image_xarray_dask
    else:
        compute_dst_src_ij_images = _compute_ij_images_xarray_numpy
        compute_dst_var_image = _compute_var_image_xarray_numpy

    dst_src_ij_array = compute_dst_src_ij_images(source_gm,
                                                 target_gm,
                                                 uv_delta)

    dst_x_dim, dst_y_dim = target_gm.xy_dim_names
    dst_dims = dst_y_dim, dst_x_dim
    dst_ds_coords = target_gm.to_coords()
    dst_vars = dict()
    for src_var_name, src_var in src_vars.items():
        dst_var_dims = src_var.dims[0:-2] + dst_dims
        dst_var_coords = {d: src_var.coords[d] for d in dst_var_dims if d in src_var.coords}
        dst_var_coords.update({d: dst_ds_coords[d] for d in dst_var_dims if d in dst_ds_coords})
        dst_var_array = compute_dst_var_image(src_var,
                                              dst_src_ij_array,
                                              fill_value=np.nan)
        dst_var = xr.DataArray(dst_var_array,
                               dims=dst_var_dims,
                               coords=dst_var_coords,
                               attrs=src_var.attrs)
        dst_vars[src_var_name] = dst_var

    if output_ij_names:
        output_i_name, output_j_name = output_ij_names
        dst_ij_coords = {d: dst_ds_coords[d] for d in dst_dims if d in dst_ds_coords}
        dst_vars[output_i_name] = xr.DataArray(dst_src_ij_array[0], dims=dst_dims, coords=dst_ij_coords)
        dst_vars[output_j_name] = xr.DataArray(dst_src_ij_array[1], dims=dst_dims, coords=dst_ij_coords)

    return xr.Dataset(dst_vars, coords=dst_ds_coords, attrs=src_attrs)


def _select_variables(source_ds: xr.Dataset,
                      source_gm: GridMapping,
                      var_names: Union[None, str, Sequence[str]]) -> Mapping[str, xr.DataArray]:
    """
    Select variables from *dataset*.

    :param source_ds: Source dataset.
    :param source_gm: Optional dataset geo-coding.
    :param var_names: Optional variable name or sequence of variable names.
    :return: The selected variables as a variable name to ``xr.DataArray`` mapping
    """
    spatial_var_names = source_gm.xy_var_names
    spatial_shape = tuple(reversed(source_gm.size))
    spatial_dims = tuple(reversed(source_gm.xy_dim_names))
    if var_names is None:
        var_names = [var_name for var_name, var in source_ds.data_vars.items()
                     if var_name not in spatial_var_names and _is_2d_spatial_var(var, spatial_shape, spatial_dims)]
    elif isinstance(var_names, str):
        var_names = (var_names,)
    elif len(var_names) == 0:
        raise ValueError(f'empty var_names')
    src_vars = {}
    for var_name in var_names:
        src_var = source_ds[var_name]
        if not _is_2d_spatial_var(src_var, spatial_shape, spatial_dims):
            raise ValueError(
                f"cannot rectify variable {var_name!r} as its shape or dimensions "
                f"do not match those of {spatial_var_names[0]!r} and {spatial_var_names[1]!r}")
        src_vars[var_name] = src_var
    return src_vars


def _is_2d_spatial_var(var: xr.DataArray, shape, dims) -> bool:
    return var.ndim >= 2 and var.shape[-2:] == shape and var.dims[-2:] == dims


def _compute_ij_images_xarray_numpy(src_geo_coding: GridMapping,
                                    output_geom: GridMapping,
                                    uv_delta: float) -> np.ndarray:
    """Compute numpy.ndarray destination image with source pixel i,j coords from xarray.DataArray x,y sources """
    dst_width = output_geom.width
    dst_height = output_geom.height
    dst_shape = 2, dst_height, dst_width
    dst_src_ij_images = np.full(dst_shape, np.nan, dtype=np.float64)
    dst_x_offset = output_geom.x_min
    dst_y_offset = output_geom.y_min if output_geom.is_j_axis_up else output_geom.y_max
    dst_x_scale = output_geom.x_res
    dst_y_scale = output_geom.y_res if output_geom.is_j_axis_up else -output_geom.y_res
    # TODO: GridMapping: this will cause out-of-memory problems!
    x_values, y_values = src_geo_coding.xy_coords.values
    _compute_ij_images_numpy_parallel(x_values,
                                      y_values,
                                      0,
                                      0,
                                      dst_src_ij_images,
                                      dst_x_offset,
                                      dst_y_offset,
                                      dst_x_scale,
                                      dst_y_scale,
                                      uv_delta)
    return dst_src_ij_images


def _compute_ij_images_xarray_dask(src_geo_coding: GridMapping,
                                   output_geom: GridMapping,
                                   uv_delta: float) -> da.Array:
    """Compute dask.array.Array destination image with source pixel i,j coords from xarray.DataArray x,y sources """
    dst_width = output_geom.width
    dst_height = output_geom.height
    dst_tile_width = output_geom.tile_width
    dst_tile_height = output_geom.tile_height
    dst_var_shape = 2, dst_height, dst_width
    dst_var_chunks = 2, dst_tile_height, dst_tile_width

    dst_x_min, dst_y_min, dst_x_max, dst_y_max = output_geom.xy_bbox
    dst_x_res, dst_y_res = output_geom.xy_res
    dst_is_j_axis_up = output_geom.is_j_axis_up

    # Compute an empirical xy_border as a function of the number of tiles, because the more tiles we have
    # the smaller the destination xy-bboxes and the higher the risk to not find any source ij-bbox for
    # a given xy-bbox.
    # xy_border will not be larger than half of the coverage of a tile.
    #
    num_tiles_x = dst_width / dst_tile_width
    num_tiles_y = dst_height / dst_tile_height
    xy_border = min(min(2 * num_tiles_x * output_geom.x_res,
                        2 * num_tiles_y * output_geom.y_res),
                    min(0.5 * (dst_x_max - dst_x_min),
                        0.5 * (dst_y_max - dst_y_min)))

    dst_xy_bboxes = output_geom.xy_bboxes
    src_ij_bboxes = src_geo_coding.ij_bboxes_from_xy_bboxes(dst_xy_bboxes,
                                                            xy_border=xy_border, ij_border=1)

    return compute_array_from_func(_compute_ij_images_xarray_dask_block,
                                   dst_var_shape,
                                   dst_var_chunks,
                                   np.float64,
                                   ctx_arg_names=[
                                       'dtype',
                                       'block_id',
                                       'block_shape',
                                       'block_slices',
                                   ],
                                   args=(
                                       src_geo_coding.xy_coords,
                                       src_ij_bboxes,
                                       dst_x_min,
                                       dst_y_min,
                                       dst_y_max,
                                       dst_x_res,
                                       dst_y_res,
                                       dst_is_j_axis_up,
                                       uv_delta
                                   ),
                                   name='ij_pixels')


def _compute_ij_images_xarray_dask_block(dtype: np.dtype,
                                         block_id: int,
                                         block_shape: Tuple[int, int],
                                         block_slices: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                                         src_xy_coords: xr.DataArray,
                                         src_ij_bboxes: np.ndarray,
                                         dst_x_min: float,
                                         dst_y_min: float,
                                         dst_y_max: float,
                                         dst_x_res: float,
                                         dst_y_res: float,
                                         dst_is_j_axis_up: bool,
                                         uv_delta: float) -> np.ndarray:
    """Compute dask.array.Array destination block with source pixel i,j coords from xarray.DataArray x,y sources """
    dst_src_ij_block = np.full(block_shape, np.nan, dtype=dtype)
    _, (dst_y_slice_start, _), (dst_x_slice_start, _) = block_slices
    src_ij_bbox = src_ij_bboxes[block_id]
    src_i_min, src_j_min, src_i_max, src_j_max = src_ij_bbox
    if src_i_min == -1:
        return dst_src_ij_block
    src_xy_values = src_xy_coords[:, src_j_min:src_j_max + 1, src_i_min:src_i_max + 1].values
    src_x_values = src_xy_values[0]
    src_y_values = src_xy_values[1]
    dst_x_offset = dst_x_min + dst_x_slice_start * dst_x_res
    if dst_is_j_axis_up:
        dst_y_offset = dst_y_min + dst_y_slice_start * dst_y_res
    else:
        dst_y_offset = dst_y_max - dst_y_slice_start * dst_y_res
    _compute_ij_images_numpy_sequential(src_x_values,
                                        src_y_values,
                                        src_i_min,
                                        src_j_min,
                                        dst_src_ij_block,
                                        dst_x_offset,
                                        dst_y_offset,
                                        dst_x_res,
                                        dst_y_res if dst_is_j_axis_up else -dst_y_res,
                                        uv_delta)
    return dst_src_ij_block


@nb.njit(nogil=True, parallel=True, cache=True)
def _compute_ij_images_numpy_parallel(src_x_image: np.ndarray,
                                      src_y_image: np.ndarray,
                                      src_i_min: int,
                                      src_j_min: int,
                                      dst_src_ij_images: np.ndarray,
                                      dst_x_offset: float,
                                      dst_y_offset: float,
                                      dst_x_scale: float,
                                      dst_y_scale: float,
                                      uv_delta: float):
    """Compute numpy.ndarray destination image with source pixel i,j coords
    from numpy.ndarray x,y sources in parallel mode."""
    src_height = src_x_image.shape[-2]
    dst_src_ij_images[:, :, :] = np.nan
    for src_j0 in nb.prange(src_height - 1):
        _compute_ij_images_for_source_line(src_j0,
                                           src_x_image,
                                           src_y_image,
                                           src_i_min,
                                           src_j_min,
                                           dst_src_ij_images,
                                           dst_x_offset,
                                           dst_y_offset,
                                           dst_x_scale,
                                           dst_y_scale,
                                           uv_delta)


# Extra dask version, because if we use parallel=True and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_ij_images_numpy_sequential(src_x_image: np.ndarray,
                                        src_y_image: np.ndarray,
                                        src_i_min: int,
                                        src_j_min: int,
                                        dst_src_ij_images: np.ndarray,
                                        dst_x_offset: float,
                                        dst_y_offset: float,
                                        dst_x_scale: float,
                                        dst_y_scale: float,
                                        uv_delta: float):
    """Compute numpy.ndarray destination image with source pixel i,j coords
    from numpy.ndarray x,y sources NOT in parallel mode."""
    src_height = src_x_image.shape[-2]
    dst_src_ij_images[:, :, :] = np.nan
    for src_j0 in range(src_height - 1):
        _compute_ij_images_for_source_line(src_j0,
                                           src_x_image,
                                           src_y_image,
                                           src_i_min,
                                           src_j_min,
                                           dst_src_ij_images,
                                           dst_x_offset,
                                           dst_y_offset,
                                           dst_x_scale,
                                           dst_y_scale,
                                           uv_delta)


@nb.njit(nogil=True, cache=True)
def _compute_ij_images_for_source_line(src_j0: int,
                                       src_x_image: np.ndarray,
                                       src_y_image: np.ndarray,
                                       src_i_min: int,
                                       src_j_min: int,
                                       dst_src_ij_images: np.ndarray,
                                       dst_x_offset: float,
                                       dst_y_offset: float,
                                       dst_x_scale: float,
                                       dst_y_scale: float,
                                       uv_delta: float):
    """Compute numpy.ndarray destination image with source pixel i,j coords
    from a numpy.ndarray x,y source line."""
    src_width = src_x_image.shape[-1]

    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]

    dst_px = np.zeros(4, dtype=src_x_image.dtype)
    dst_py = np.zeros(4, dtype=src_y_image.dtype)

    u_min = v_min = -uv_delta
    uv_max = 1.0 + 2 * uv_delta

    for src_i0 in range(src_width - 1):
        src_i1 = src_i0 + 1
        src_j1 = src_j0 + 1

        dst_px[0] = dst_p0x = src_x_image[src_j0, src_i0]
        dst_px[1] = dst_p1x = src_x_image[src_j0, src_i1]
        dst_px[2] = dst_p2x = src_x_image[src_j1, src_i0]
        dst_px[3] = dst_p3x = src_x_image[src_j1, src_i1]

        dst_py[0] = dst_p0y = src_y_image[src_j0, src_i0]
        dst_py[1] = dst_p1y = src_y_image[src_j0, src_i1]
        dst_py[2] = dst_p2y = src_y_image[src_j1, src_i0]
        dst_py[3] = dst_p3y = src_y_image[src_j1, src_i1]

        dst_pi = np.floor((dst_px - dst_x_offset) / dst_x_scale).astype(np.int64)
        dst_pj = np.floor((dst_py - dst_y_offset) / dst_y_scale).astype(np.int64)

        dst_i_min = np.min(dst_pi)
        dst_i_max = np.max(dst_pi)
        dst_j_min = np.min(dst_pj)
        dst_j_max = np.max(dst_pj)

        if dst_i_max < 0 \
                or dst_j_max < 0 \
                or dst_i_min >= dst_width \
                or dst_j_min >= dst_height:
            continue

        if dst_i_min < 0:
            dst_i_min = 0

        if dst_i_max >= dst_width:
            dst_i_max = dst_width - 1

        if dst_j_min < 0:
            dst_j_min = 0

        if dst_j_max >= dst_height:
            dst_j_max = dst_height - 1

        # u from p0 right to p1, v from p0 down to p2
        det_a = _fdet(dst_p0x, dst_p0y, dst_p1x, dst_p1y, dst_p2x, dst_p2y)
        # u from p3 left to p2, v from p3 up to p1
        det_b = _fdet(dst_p3x, dst_p3y, dst_p2x, dst_p2y, dst_p1x, dst_p1y)

        if np.isnan(det_a) or np.isnan(det_b):
            # print('no plane at:', src_i0, src_j0)
            continue

        for dst_j in range(dst_j_min, dst_j_max + 1):
            dst_y = dst_y_offset + (dst_j + 0.5) * dst_y_scale
            for dst_i in range(dst_i_min, dst_i_max + 1):
                dst_x = dst_x_offset + (dst_i + 0.5) * dst_x_scale

                # TODO: use two other combinations,
                #       if one of the dst_px<n>,dst_py<n> pairs is missing.

                src_i = src_j = -1

                if det_a != 0.0:
                    u = _fu(dst_x, dst_y, dst_p0x, dst_p0y, dst_p2x, dst_p2y) / det_a
                    v = _fv(dst_x, dst_y, dst_p0x, dst_p0y, dst_p1x, dst_p1y) / det_a
                    if u >= u_min and v >= v_min and u + v <= uv_max:
                        src_i = src_i0 + _fclamp(u, 0.0, 1.0)
                        src_j = src_j0 + _fclamp(v, 0.0, 1.0)
                if src_i == -1 and det_b != 0.0:
                    u = _fu(dst_x, dst_y, dst_p3x, dst_p3y, dst_p1x, dst_p1y) / det_b
                    v = _fv(dst_x, dst_y, dst_p3x, dst_p3y, dst_p2x, dst_p2y) / det_b
                    if u >= u_min and v >= v_min and u + v <= uv_max:
                        src_i = src_i1 - _fclamp(u, 0.0, 1.0)
                        src_j = src_j1 - _fclamp(v, 0.0, 1.0)
                if src_i != -1:
                    dst_src_ij_images[0, dst_j, dst_i] = src_i_min + src_i
                    dst_src_ij_images[1, dst_j, dst_i] = src_j_min + src_j


def _compute_var_image_xarray_numpy(src_var: xr.DataArray,
                                    dst_src_ij_images: np.ndarray,
                                    fill_value: Union[int, float, complex] = np.nan) -> np.ndarray:
    """Extract source pixels from xarray.DataArray source with numpy.ndarray data"""
    return _compute_var_image_numpy(src_var.values, dst_src_ij_images, fill_value)


def _compute_var_image_xarray_dask(src_var: xr.DataArray,
                                   dst_src_ij_images: np.ndarray,
                                   fill_value: Union[int, float, complex] = np.nan) -> da.Array:
    """Extract source pixels from xarray.DataArray source with dask.array.Array data"""
    return da.map_blocks(_compute_var_image_xarray_dask_block,
                         src_var.values,
                         dst_src_ij_images,
                         fill_value,
                         dtype=src_var.dtype,
                         drop_axis=0)


@nb.njit(nogil=True, cache=True)
def _compute_var_image_numpy(src_var: np.ndarray,
                             dst_src_ij_images: np.ndarray,
                             fill_value: Union[int, float, complex] = np.nan) -> np.ndarray:
    """Extract source pixels from numpy.ndarray source with numba in parallel mode"""
    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]
    dst_shape = src_var.shape[:-2] + (dst_height, dst_width)
    dst_values = np.full(dst_shape, fill_value, dtype=src_var.dtype)
    _compute_var_image_numpy_parallel(src_var, dst_src_ij_images, dst_values)
    return dst_values


@nb.njit(nogil=True, cache=True)
def _compute_var_image_xarray_dask_block(src_var_image: np.ndarray,
                                         dst_src_ij_images: np.ndarray,
                                         fill_value: Union[int, float, complex] = np.nan) -> np.ndarray:
    """Extract source pixels from np.ndarray source and return a block of a dask array"""
    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]
    dst_shape = src_var_image.shape[:-2] + (dst_height, dst_width)
    dst_values = np.full(dst_shape, fill_value, dtype=src_var_image.dtype)
    _compute_var_image_numpy_sequential(src_var_image, dst_src_ij_images, dst_values)
    return dst_values


@nb.njit(nogil=True, parallel=True, cache=True)
def _compute_var_image_numpy_parallel(src_var_image: np.ndarray,
                                      dst_src_ij_images: np.ndarray,
                                      dst_var_image: np.ndarray):
    """Extract source pixels from np.ndarray source using numba parallel mode"""
    dst_height = dst_var_image.shape[-2]
    for dst_j in nb.prange(dst_height):
        _compute_var_image_for_dest_line(dst_j, src_var_image, dst_src_ij_images, dst_var_image)


# Extra dask version, because if we use parallel=True and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_var_image_numpy_sequential(src_var_image: np.ndarray,
                                        dst_src_ij_images: np.ndarray,
                                        dst_var_image: np.ndarray):
    """Extract source pixels from np.ndarray source NOT using numba parallel mode"""
    dst_height = dst_var_image.shape[-2]
    for dst_j in range(dst_height):
        _compute_var_image_for_dest_line(dst_j, src_var_image, dst_src_ij_images, dst_var_image)


@nb.njit(nogil=True, cache=True)
def _compute_var_image_for_dest_line(dst_j: int,
                                     src_var_image: np.ndarray,
                                     dst_src_ij_images: np.ndarray,
                                     dst_var_image: np.ndarray):
    """Extract source pixels from *src_values* np.ndarray and write into dst_values np.ndarray"""
    src_width = src_var_image.shape[-1]
    src_height = src_var_image.shape[-2]
    dst_width = dst_var_image.shape[-1]
    src_i_min = 0
    src_j_min = 0
    src_i_max = src_width - 1
    src_j_max = src_height - 1
    for dst_i in range(dst_width):
        src_i_f = dst_src_ij_images[0, dst_j, dst_i]
        src_j_f = dst_src_ij_images[1, dst_j, dst_i]
        if np.isnan(src_i_f) or np.isnan(src_j_f):
            continue
        # Note int() is 2x faster than math.floor() and should yield the same results for only positive i,j.
        src_i = int(src_i_f)
        src_j = int(src_j_f)
        u = src_i_f - src_i
        v = src_j_f - src_j
        if u > 0.5:
            src_i = _iclamp(src_i + 1, src_i_min, src_i_max)
        if v > 0.5:
            src_j = _iclamp(src_j + 1, src_j_min, src_j_max)
        dst_var_image[..., dst_j, dst_i] = src_var_image[..., src_j, src_i]


@nb.njit('float64(float64, float64, float64, float64, float64, float64)', nogil=True, inline='always')
def _fdet(px0: float, py0: float, px1: float, py1: float, px2: float, py2: float) -> float:
    return (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)


@nb.njit('float64(float64, float64, float64, float64, float64, float64)', nogil=True, inline='always')
def _fu(px: float, py: float, px0: float, py0: float, px2: float, py2: float) -> float:
    return (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)


@nb.njit('float64(float64, float64, float64, float64, float64, float64)', nogil=True, inline='always')
def _fv(px: float, py: float, px0: float, py0: float, px1: float, py1: float) -> float:
    return (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)


@nb.njit('float64(float64, float64, float64)', nogil=True, inline='always')
def _fclamp(x: float, x_min: float, x_max: float) -> float:
    return x_min if x < x_min else (x_max if x > x_max else x)


@nb.njit('int64(int64, int64, int64)', nogil=True, inline='always')
def _iclamp(x: int, x_min: int, x_max: int) -> int:
    return x_min if x < x_min else (x_max if x > x_max else x)


def _millis(seconds: float) -> int:
    return round(1000 * seconds)
