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

from typing import Optional

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.assertions import assert_instance


def encode_grid_mapping(
    ds: xr.Dataset,
    gm: GridMapping,
    gm_name: Optional[str] = None,
    force: Optional[bool] = None,
) -> xr.Dataset:
    """Encode the given grid mapping *gm* into a
    copy of *ds* in a CF-compliant way and return the dataset copy.
    The function removes any existing grid mappings.

    If the CRS of *gm* is geographic and the spatial dimension and coordinate
    names are "lat", "lon" and *force* is `False`, or *force* is `None`
    and no former grid mapping was encoded in *ds*, then nothing else is
    done and the dataset copy is returned without further action.

    Otherwise, for every spatial data variable with dims=(..., y, x),
    the function sets the attribute "grid_mapping" to *gm_name*.
    The grid mapping CRS is encoded in a new 0-D variable named *gm_name*.

    Args:
        ds: The dataset.
        gm: The dataset's grid mapping.
        gm_name: Name for the grid mapping variable. Defaults to "crs".
        force: Whether to force encoding of grid mapping even if CRS is
            geographic and spatial dimension names are "lon", "lat".
            Optional value, if not provided, *force* will be assumed
            `True` if a former grid mapping was encoded in *ds*.

    Returns:
        A copy of *ds* with *gm* encoded into it.
    """
    assert_instance(ds, xr.Dataset, "ds")
    assert_instance(gm, GridMapping, "gm")
    if gm_name is not None:
        assert_instance(gm_name, str, "gm_name")

    ds_copy = ds.copy()

    x_dim_name, y_dim_name = gm.xy_dim_names
    spatial_vars = [
        (var_name, var)
        for var_name, var in ds.data_vars.items()
        if (var.ndim >= 2 and var.dims[-1] == x_dim_name and var.dims[-2] == y_dim_name)
    ]

    old_gm_names = set(
        old_gm_name
        for old_gm_name in (
            var.attrs.get("grid_mapping") for var_name, var in spatial_vars
        )
        if old_gm_name and old_gm_name in ds_copy
    )
    if old_gm_names:
        force = True if force is None else force
        gm_name = gm_name or next(iter(old_gm_names))
        ds_copy = ds_copy.drop_vars(old_gm_names)

    is_geographic = (
        gm.xy_var_names == gm.xy_dim_names
        and gm.xy_dim_names == ("lon", "lat")
        and gm.crs.is_geographic
    )

    if force or not is_geographic:
        gm_name = gm_name or "crs"
        for var_name, var in spatial_vars:
            ds_copy[var_name] = var.assign_attrs(grid_mapping=gm_name)
        ds_copy[gm_name] = xr.DataArray(0, attrs=gm.crs.to_cf())

    return ds_copy


def maybe_encode_grid_mapping(
    encode_cf: bool, ds: xr.Dataset, gm: GridMapping, gm_name: Optional[str]
) -> xr.Dataset:
    """Internal helper."""
    if encode_cf:
        return encode_grid_mapping(
            ds, gm, gm_name=gm_name, force=True if gm_name else None
        )
    return ds
