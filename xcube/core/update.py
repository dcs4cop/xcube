# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import datetime
from typing import Any
from typing import Dict

import xarray as xr

from xcube.constants import FORMAT_NAME_NETCDF4
from xcube.constants import FORMAT_NAME_ZARR
from xcube.core.gridmapping import GridMapping
from xcube.util.config import NameDictPairList

_TIME_ATTRS_DATA = (
    "time",
    "time_bnds",
    ("time_coverage_start", "time_coverage_end"),
    str,
)


def update_dataset_attrs(
    dataset: xr.Dataset,
    global_attrs: Dict[str, Any] = None,
    update_existing: bool = False,
    in_place: bool = False,
) -> xr.Dataset:
    """Update spatio-temporal CF/THREDDS attributes
    given *dataset* according to spatio-temporal coordinate
    variables time, lat, and lon.

    Args:
        dataset: The dataset.
        global_attrs: Optional global attributes.
        update_existing: If `True`, any existing attributes will be updated.
        in_place: If `True`, *dataset* will be modified in place and returned.
    Returns:
         A new dataset, if *in_place* if `False` (default), else the passed and modified *dataset*.
    """
    if not in_place:
        dataset = dataset.copy()

    if global_attrs:
        dataset.attrs.update(global_attrs)

    dataset = update_dataset_spatial_attrs(
        dataset, update_existing=update_existing, in_place=in_place
    )
    dataset = update_dataset_temporal_attrs(
        dataset, update_existing=update_existing, in_place=in_place
    )
    return dataset


def update_dataset_spatial_attrs(
    dataset: xr.Dataset, update_existing: bool = False, in_place: bool = False
) -> xr.Dataset:
    """Update spatial CF/THREDDS attributes of given *dataset*.

    Args:
        dataset: The dataset.
        update_existing: If `True`, any existing attributes will be
            updated.
        in_place: If `True`, *dataset* will be modified in place and
            returned.

    Returns:
        A new dataset, if *in_place* if `False` (default), else the
        passed and modified *dataset*.
    """
    if not in_place:
        dataset = dataset.copy()
    gm = GridMapping.from_dataset(dataset)
    gs_attrs = {
        "geospatial_lon_min",
        "geospatial_lon_max",
        "geospatial_lat_min",
        "geospatial_lat_max",
    }
    if update_existing or not gs_attrs.issubset(dataset.attrs):
        # Update dataset with newly retrieved attributes
        dataset.attrs.update(gm.to_dataset_attrs())
        dataset.attrs["date_modified"] = datetime.datetime.now().isoformat()
    return dataset


def update_dataset_temporal_attrs(
    dataset: xr.Dataset, update_existing: bool = False, in_place: bool = False
) -> xr.Dataset:
    """Update temporal CF/THREDDS attributes of given *dataset*.

    Args:
        dataset: The dataset.
        update_existing: If `True`, any existing attributes will be
            updated.
        in_place: If `True`, *dataset* will be modified in place and
            returned.

    Returns:
        A new dataset, if *in_place* is `False` (default), else the
        passed and modified *dataset*.
    """
    coord_data = [_TIME_ATTRS_DATA]
    if not in_place:
        dataset = dataset.copy()

    for coord_name, coord_bnds_name, coord_attr_names, cast in coord_data:
        coord_min_attr_name, coord_max_attr_name = coord_attr_names
        if (
            update_existing
            or coord_min_attr_name not in dataset.attrs
            or coord_max_attr_name not in dataset.attrs
        ):
            coord = None
            coord_bnds = None
            coord_res = None
            if coord_name in dataset:
                coord = dataset[coord_name]
                coord_bnds_name = coord.attrs.get("bounds", coord_bnds_name)
            if coord_bnds_name in dataset:
                coord_bnds = dataset[coord_bnds_name]
            if (
                coord_bnds is not None
                and coord_bnds.ndim == 2
                and coord_bnds.shape[0] > 0
                and coord_bnds.shape[1] == 2
            ):
                coord_v1 = coord_bnds[0][0]
                coord_v2 = coord_bnds[-1][1]
                coord_res = (coord_v2 - coord_v1) / coord_bnds.shape[0]
                coord_res = float(coord_res.values)
                coord_min, coord_max = (
                    (coord_v1, coord_v2) if coord_res > 0 else (coord_v2, coord_v1)
                )
                dataset.attrs[coord_min_attr_name] = cast(coord_min.values)
                dataset.attrs[coord_max_attr_name] = cast(coord_max.values)
            elif coord is not None and coord.ndim == 1 and coord.shape[0] > 0:
                coord_v1 = coord[0]
                coord_v2 = coord[-1]
                if coord.shape[0] > 1:
                    coord_res = (coord_v2 - coord_v1) / (coord.shape[0] - 1)
                    coord_v1 -= coord_res / 2
                    coord_v2 += coord_res / 2
                    coord_res = float(coord_res.values)
                    coord_min, coord_max = (
                        (coord_v1, coord_v2) if coord_res > 0 else (coord_v2, coord_v1)
                    )
                else:
                    coord_min, coord_max = coord_v1, coord_v2
                dataset.attrs[coord_min_attr_name] = cast(coord_min.values)
                dataset.attrs[coord_max_attr_name] = cast(coord_max.values)

    dataset.attrs["date_modified"] = datetime.datetime.now().isoformat()
    return dataset


def update_dataset_var_attrs(
    dataset: xr.Dataset, var_attrs_list: NameDictPairList
) -> xr.Dataset:
    """Update the attributes of variables in given *dataset*.
    Optionally rename variables according to a given attribute named "name".

    *var_attrs_list* must be a sequence of pairs of the form (<var_name>, <var_attrs>) where <var_name> is a string
    and <var_attrs> is a dictionary representing the attributes to be updated , including an optional "name" attribute.
    If <var_attrs> contains an attribute "name", the variable named <var_name> will be renamed to that attribute's
    value.

    Args:
        dataset: A dataset.
        var_attrs_list: List of tuples of the form (variable name,
            properties dictionary).

    Returns:
        A shallow copy of *dataset* with updated / renamed variables.
    """
    if not var_attrs_list:
        return dataset

    var_name_attrs = dict()
    var_renamings = dict()
    new_var_names = set()

    # noinspection PyUnusedLocal,PyShadowingNames
    for var_name, var_attrs in var_attrs_list:
        if not var_attrs:
            continue
        # noinspection PyShadowingNames
        var_attrs = dict(var_attrs)
        if "name" in var_attrs:
            new_var_name = var_attrs.pop("name")
            if new_var_name in new_var_names:
                raise ValueError(
                    f"variable {var_name!r} cannot be renamed into {new_var_name!r} "
                    "because the name is already in use"
                )
            new_var_names.add(new_var_name)
            var_attrs["original_name"] = var_name
            var_renamings[var_name] = new_var_name
            var_name = new_var_name
        var_name_attrs[var_name] = var_attrs

    if var_renamings:
        dataset = dataset.rename(var_renamings)
    elif var_name_attrs:
        dataset = dataset.copy()

    if var_name_attrs:
        for var_name, var_attrs in var_name_attrs.items():
            var = dataset[var_name]
            var.attrs.update(var_attrs)

    return dataset


def update_dataset_chunk_encoding(
    dataset: xr.Dataset,
    chunk_sizes: Dict[str, int] = None,
    format_name: str = None,
    in_place: bool = False,
) -> xr.Dataset:
    """Update each variable's encoding in *dataset* with respect to *chunk_sizes*
    so *dataset* is written in chunks for given *format_name*.

    Args:
        dataset: input dataset.
        chunk_sizes: the chunk sizes to be used for the encoding. If
            None, any chunking encoding is removed.
        format_name: format name, e.g. "zarr" or "netcdf4".
        in_place: If `True`, *dataset* will be modified in place and
            returned.
    """
    if format_name == FORMAT_NAME_ZARR:
        chunk_sizes_attr_name = "chunks"
    elif format_name == FORMAT_NAME_NETCDF4:
        chunk_sizes_attr_name = "chunksizes"
    else:
        return dataset
    if not in_place:
        dataset = dataset.copy()
    for var_name in dataset.variables:
        var = dataset[var_name]
        if chunk_sizes is not None:

            def get_size(i):
                dim_name = var.dims[i]
                size = chunk_sizes.get(dim_name)
                if isinstance(size, int):
                    return size
                if var.chunks:
                    size = var.chunks[i]
                    if isinstance(size, int):
                        return size
                    if len(size):
                        return size[0]
                return var.shape[i]

            var.encoding.update(
                {chunk_sizes_attr_name: tuple(map(get_size, range(var.ndim)))}
            )
        elif chunk_sizes_attr_name in var.encoding:
            # Remove any explicit and possibly unintended specification
            del var.encoding[chunk_sizes_attr_name]
    return dataset
