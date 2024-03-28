# The MIT License (MIT)
# Copyright (c) 2023 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Union, Callable, Mapping, Hashable, Any, Optional

import numpy as np
import xarray as xr
from dask import array as da

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.helpers import scale_xy_res_and_size
from .affine import affine_transform_dataset
from .affine import resample_dataset
from .rectify import rectify_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]

# If _SCALE_LIMIT is exceeded, we don't need
# to downscale source image before we can
# rectify it.
_SCALE_LIMIT = 0.95


def resample_in_space(
    dataset: xr.Dataset,
    source_gm: GridMapping = None,
    target_gm: GridMapping = None,
    var_configs: Mapping[Hashable, Mapping[str, Any]] = None,
    encode_cf: bool = True,
    gm_name: Optional[str] = None,
):
    """
    Resample a dataset in the spatial dimensions.

    If the source grid mapping *source_gm* is not given,
    it is derived from *dataset*:
    `source_gm = GridMapping.from_dataset(dataset)`.

    If the target grid mapping *target_gm* is not given,
    it is derived from *source_gm*:
    `target_gm = source_gm.to_regular()`.

    If *source_gm* is almost equal to *target_gm*, this
    function is a no-op and *dataset* is returned unchanged.

    Otherwise the function computes a spatially
    resampled version of *dataset* and returns it.

    Using *var_configs*, the resampling of individual
    variables can be configured. If given, *var_configs*
    must be a mapping from variable names to configuration
    dictionaries which can have the following properties:

    * `spline_order` (int) - The order of spline polynomials
        used for interpolating. It is used for upsampling only.
        Possible values are 0 to 5.
        Default is 1 (bi-linear) for floating point variables,
        and 0 (= nearest neighbor) for integer and bool variables.
    * `aggregator` (str) - An optional aggregating
        function. It is used for downsampling only.
        Examples are numpy.nanmean, numpy.nanmin, numpy.nanmax.
        Default is numpy.nanmean for floating point variables,
        and None (= nearest neighbor) for integer and bool variables.
    * `recover_nan` (bool) - whether a special algorithm
        shall be used that is able to recover values that would
        otherwise yield NaN during resampling.
        Default is True for floating point variables,
        and False for integer and bool variables.

    Note that *var_configs* is only used if the resampling involves
    an affine transformation. This is true if the CRS of
    *source_gm* and CRS of *target_gm* are equal and one of two
    cases is given:

    1. *source_gm* is regular.
       In this case the resampling is the affine transformation.
       and the result is returned directly.
    2. *source_gm* is not regular and has a lower resolution
       than *target_cm*.
       In this case *dataset* is downsampled first using an affine
       transformation. Then the result is rectified.

    In all other cases, no affine transformation is applied and
    the resampling is a direct rectification.

    Args:
        dataset: The source dataset.
        source_gm: The source grid mapping.
        target_gm: The target grid mapping. Must be regular.
        var_configs: Optional resampling configurations
            for individual variables.
        encode_cf: Whether to encode the target grid mapping
            into the resampled dataset in a CF-compliant way.
            Defaults to `True`.
        gm_name: Name for the grid mapping variable.
            Defaults to "crs". Used only if *encode_cf* is `True`.
    Returns:
         The spatially resampled dataset.
    """
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset
        source_gm = GridMapping.from_dataset(dataset)

    if target_gm is None:
        # No target grid mapping given, so do derive it from source
        target_gm = source_gm.to_regular()

    if source_gm.is_close(target_gm):
        # If source and target grid mappings are almost equal
        return dataset

    # target_gm must be regular
    GridMapping.assert_regular(target_gm, name="target_gm")

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic

    if both_geographic or source_gm.crs == target_gm.crs:
        # If CRSes are both geographic or their CRSes are equal:
        if source_gm.is_regular:
            # If also the source is regular, then resampling reduces
            # to an affine transformation.
            return affine_transform_dataset(
                dataset,
                source_gm=source_gm,
                target_gm=target_gm,
                var_configs=var_configs,
                encode_cf=encode_cf,
                gm_name=gm_name,
            )

        # If the source is not regular, we need to rectify it,
        # so the target is regular. Our rectification implementation
        # works only correctly if source pixel size >= target pixel
        # size. Therefore check if we must downscale source first.
        x_scale = source_gm.x_res / target_gm.x_res
        y_scale = source_gm.y_res / target_gm.y_res
        if x_scale > _SCALE_LIMIT and y_scale > _SCALE_LIMIT:
            # Source pixel size >= target pixel size.
            # We can rectify.
            return rectify_dataset(
                dataset,
                source_gm=source_gm,
                target_gm=target_gm,
                encode_cf=encode_cf,
                gm_name=gm_name,
            )

        # Source has higher resolution than target.
        # Downscale first, then rectify
        if source_gm.is_regular:
            # If source is regular
            downscaled_gm = source_gm.scale((x_scale, y_scale))
            downscaled_dataset = resample_dataset(
                dataset,
                ((x_scale, 1, 0), (1, y_scale, 0)),
                size=downscaled_gm.size,
                tile_size=source_gm.tile_size,
                xy_dim_names=source_gm.xy_dim_names,
                var_configs=var_configs,
            )
        else:
            _, downscaled_size = scale_xy_res_and_size(
                source_gm.xy_res, source_gm.size, (x_scale, y_scale)
            )
            downscaled_dataset = resample_dataset(
                dataset,
                ((x_scale, 1, 0), (1, y_scale, 0)),
                size=downscaled_size,
                tile_size=source_gm.tile_size,
                xy_dim_names=source_gm.xy_dim_names,
                var_configs=var_configs,
            )
            downscaled_gm = GridMapping.from_dataset(
                downscaled_dataset,
                tile_size=source_gm.tile_size,
                prefer_crs=source_gm.crs,
            )
        return rectify_dataset(
            downscaled_dataset,
            source_gm=downscaled_gm,
            target_gm=target_gm,
            encode_cf=encode_cf,
            gm_name=gm_name,
        )

    # If CRSes are not both geographic and their CRSes are different
    # transform the source_gm so its CRS matches the target CRS:
    transformed_source_gm = source_gm.transform(target_gm.crs)
    transformed_x, transformed_y = transformed_source_gm.xy_coords
    return resample_in_space(
        dataset.assign(transformed_x=transformed_x, transformed_y=transformed_y),
        source_gm=transformed_source_gm,
        target_gm=target_gm,
        gm_name=gm_name,
    )
