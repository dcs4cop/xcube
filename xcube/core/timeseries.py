# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from typing import AbstractSet, Optional, Union
from collections.abc import Sequence

import numpy as np
import pyproj
import shapely.geometry
import shapely.ops
import shapely.wkt
import xarray as xr

from xcube.core.geom import GeometryLike
from xcube.core.geom import normalize_geometry
from xcube.core.geom import get_dataset_geometry
from xcube.core.geom import mask_dataset_by_geometry
from xcube.core.gridmapping import GridMapping
from xcube.core.varexpr import VarExprContext
from xcube.core.varexpr import split_var_assignment
from xcube.util.timeindex import ensure_time_index_compatible
from xcube.util.assertions import assert_instance
from xcube.constants import CRS_CRS84

Date = Union[np.datetime64, str]

AGG_MEAN = "mean"
AGG_MEDIAN = "median"
AGG_STD = "std"
AGG_MIN = "min"
AGG_MAX = "max"
AGG_COUNT = "count"

MUST_LOAD = True
CAN_COMPUTE = False

AGG_METHODS = {
    AGG_MEAN: CAN_COMPUTE,
    AGG_MEDIAN: MUST_LOAD,
    AGG_STD: CAN_COMPUTE,
    AGG_MIN: CAN_COMPUTE,
    AGG_MAX: CAN_COMPUTE,
    AGG_COUNT: CAN_COMPUTE,
}


def get_time_series(
    dataset: xr.Dataset,
    grid_mapping: Optional[GridMapping] = None,
    geometry: Optional[GeometryLike] = None,
    var_names: Optional[Sequence[str]] = None,
    start_date: Optional[Date] = None,
    end_date: Optional[Date] = None,
    agg_methods: Union[str, Sequence[str], AbstractSet[str]] = AGG_MEAN,
    use_groupby: bool = False,
    cube_asserted: Optional[bool] = None,
) -> Optional[xr.Dataset]:
    """Get a time series dataset from a data *cube*.

    *geometry* may be provided as a (shapely) geometry object, a valid
    GeoJSON object, a valid WKT string,
    a sequence of box coordinates (x1, y1, x2, y2), or point coordinates
    (x, y). If *geometry* covers an area,
    i.e. is not a point, the function aggregates the variables to compute a
    mean value and if desired,
    the number of valid observations and the standard deviation.

    *start_date* and *end_date* may be provided as a numpy.datetime64 or an
    ISO datetime string.

    Returns a time-series dataset whose data variables have a time dimension
    but no longer have spatial dimensions,
    hence the resulting dataset's variables will only have N-2 dimensions.
    A global attribute ``max_number_of_observations`` will be set to the
    maximum number of observations
    that could have been made in each time step.
    If the given *geometry* does not overlap the cube's boundaries, or if not
    output variables remain,
    the function returns ``None``.

    Args:
        dataset: The dataset
        grid_mapping: Grid mapping of *cube*.
        geometry: Optional geometry
        var_names: Optional sequence of names of variables to be
            included.
        start_date: Optional start date.
        end_date: Optional end date.
        agg_methods: Aggregation methods. May be single string or
            sequence of strings. Possible values are 'mean', 'median',
            'min', 'max', 'std', 'count'. Defaults to 'mean'. Ignored if
            geometry is a point.
        use_groupby: Use group-by operation. May increase or decrease
            runtime performance and/or memory consumption.
        cube_asserted: Deprecated and ignored since xcube 0.11.0. No
            replacement.
    """
    if cube_asserted is not None:
        warnings.warn(
            "cube_asserted has been deprecated" " and will be removed soon.",
            DeprecationWarning,
        )
    assert_instance(dataset, xr.Dataset)
    if grid_mapping is not None:
        assert_instance(grid_mapping, GridMapping)
    else:
        grid_mapping = GridMapping.from_dataset(dataset)

    geometry = normalize_geometry(geometry)
    if geometry is not None and not grid_mapping.crs.is_geographic:
        project = pyproj.Transformer.from_crs(
            CRS_CRS84, grid_mapping.crs, always_xy=True
        ).transform
        geometry = shapely.ops.transform(project, geometry)

    if var_names is not None:
        data_vars: dict[str, xr.DataArray] = {}
        for var_name_or_assign in var_names:
            var_name, var_expr = split_var_assignment(var_name_or_assign)
            if var_expr:
                variable = VarExprContext(dataset).evaluate(var_expr)
            else:
                var_name = var_name_or_assign
                variable = dataset[var_name]

            if isinstance(variable, xr.DataArray) and "time" in variable.dims:
                data_vars[var_name] = variable

        dataset = xr.Dataset(data_vars)

    if len(dataset.data_vars) == 0:
        return None

    if start_date is not None or end_date is not None:
        date_slice = slice(start_date, end_date)
        safe_slice = ensure_time_index_compatible(dataset, date_slice)
        dataset = dataset.sel(time=safe_slice)

    x_name, y_name = grid_mapping.xy_dim_names
    if isinstance(geometry, shapely.geometry.Point):
        bounds = get_dataset_geometry(dataset)
        if not bounds.contains(geometry):
            return None
        indexers = {x_name: geometry.x, y_name: geometry.y}
        dataset = dataset.sel(**indexers, method="Nearest")
        return dataset.assign_attrs(max_number_of_observations=1)

    agg_methods = normalize_agg_methods(agg_methods)

    if geometry is not None:
        dataset = mask_dataset_by_geometry(
            dataset,
            geometry,
            update_attrs=False,
            save_geometry_mask="__mask__",
        )
        if dataset is None:
            return None
        mask = dataset["__mask__"]
        max_number_of_observations = np.count_nonzero(mask)
        dataset = dataset.drop_vars(["__mask__"])
    else:
        max_number_of_observations = dataset[y_name].size * dataset[x_name].size

    must_load = len(agg_methods) > 1 or any(
        AGG_METHODS[agg_method] == MUST_LOAD for agg_method in agg_methods
    )
    if must_load:
        dataset.load()

    agg_datasets = []
    if use_groupby:
        time_group = dataset.groupby("time")
        for agg_method in agg_methods:
            method = getattr(time_group, agg_method)
            if agg_method == "count":
                agg_dataset = method(dim=xr.ALL_DIMS)
            else:
                agg_dataset = method(dim=xr.ALL_DIMS, skipna=True)
            agg_datasets.append(agg_dataset)
    else:
        for agg_method in agg_methods:
            method = getattr(dataset, agg_method)
            if agg_method == "count":
                agg_dataset = method(dim=(y_name, x_name))
            else:
                agg_dataset = method(dim=(y_name, x_name), skipna=True)
            agg_datasets.append(agg_dataset)

    agg_datasets = [
        agg_dataset.rename(
            name_dict={v: f"{v}_{agg_method}" for v in agg_dataset.data_vars}
        )
        for agg_method, agg_dataset in zip(agg_methods, agg_datasets)
    ]

    ts_dataset = xr.merge(agg_datasets)
    ts_dataset = ts_dataset.assign_attrs(
        max_number_of_observations=max_number_of_observations
    )

    return ts_dataset


def normalize_agg_methods(
    agg_methods: Union[str, Sequence[str]], exception_type=ValueError
) -> set[str]:
    agg_methods = agg_methods or [AGG_MEAN]
    if isinstance(agg_methods, str):
        agg_methods = [agg_methods]
    agg_methods = set(agg_methods)
    invalid_agg_methods = agg_methods - set(AGG_METHODS.keys())
    if invalid_agg_methods:
        s = "s" if len(invalid_agg_methods) > 1 else ""
        raise exception_type(
            f"invalid aggregation method{s}:"
            f' {", ".join(sorted(list(invalid_agg_methods)))}'
        )
    return agg_methods
