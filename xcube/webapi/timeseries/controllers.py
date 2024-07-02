# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Optional, Union
from collections.abc import Sequence

import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr

from xcube.constants import LOG
from xcube.core import timeseries
from xcube.core.ancvar import find_ancillary_var_names
from xcube.core.gridmapping import GridMapping
from xcube.core.varexpr import split_var_assignment
from xcube.server.api import ApiError
from xcube.util.geojson import GeoJSON
from xcube.util.perf import measure_time
from .context import TimeSeriesContext

TimeSeriesValue = dict[str, Union[str, bool, int, float, None]]
TimeSeries = list[TimeSeriesValue]
TimeSeriesCollection = list[TimeSeries]
GeoJsonObj = dict[str, Any]
GeoJsonFeature = GeoJsonObj
GeoJsonGeometry = GeoJsonObj


def get_time_series(
    ctx: TimeSeriesContext,
    ds_name: str,
    var_name: str,
    geo_json: GeoJsonObj,
    agg_methods: Union[str, Sequence[str]] = None,
    start_date: Optional[np.datetime64] = None,
    end_date: Optional[np.datetime64] = None,
    tolerance: Optional[float] = 1.0,
    max_valids: Optional[int] = None,
    incl_ancillary_vars: bool = False,
) -> Union[TimeSeries, TimeSeriesCollection]:
    """Get the time-series for a given GeoJSON object *geo_json*.

    If *geo_json* is a single geometry or feature a list of
    time-series values is returned.
    If *geo_json* is a single geometry collection
    or feature collection a collection of lists of time-series values
    is returned so that geometry/feature collection and
    time-series collection elements are corresponding
    at same indices.

    A time series value always contains the key "time" whose
    value is an ISO date/time string. Other entries
    are values with varying keys depending on the geometry
    type and *agg_methods*. Their values may be
    either a bool, int, float or None.
    For point geometries the second key is "value".
    For non-point geometries that cover spatial areas, there
    will be values for all keys given by *agg_methods*.

    Args:
        ctx: Service context object
        ds_name: The dataset identifier.
        var_name: The variable name.
        geo_json: The GeoJSON object that is a or has a geometry or
            collection of geometries.
        agg_methods: Spatial aggregation methods for geometries that
            cover a spatial area.
        start_date: An optional start date.
        end_date: An optional end date.
        tolerance: Time tolerance in seconds that expands the given time
            range. Defaults to one second.
        max_valids: Optional number of valid points. If it is None
            (default), also missing values are returned as NaN; if it is
            -1, only valid values are returned; if it is a positive
            integer, the most recent valid values are returned.
        incl_ancillary_vars: For point geometries, include values of
            ancillary variables, if any.

    Returns:
        Time-series data structure.
    """
    if tolerance:
        timedelta = pd.Timedelta(tolerance, unit="seconds")
        if start_date is not None:
            start_date -= timedelta
        if end_date is not None:
            end_date += timedelta

    agg_methods = timeseries.normalize_agg_methods(
        agg_methods, exception_type=ApiError.BadRequest
    )

    ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_name)
    dataset = ctx.datasets_ctx.get_time_series_dataset(
        ds_name,
        # Check if var_name is an expression
        var_name=var_name if "=" not in var_name else None,
    )
    geo_json_geometries, is_collection = _to_geo_json_geometries(geo_json)
    geometries = _to_shapely_geometries(geo_json_geometries)

    with measure_time() as time_result:
        results = _get_time_series_for_geometries(
            dataset,
            var_name,
            geometries,
            start_date=start_date,
            end_date=end_date,
            agg_methods=agg_methods,
            grid_mapping=ml_dataset.grid_mapping,
            max_valids=max_valids,
            incl_ancillary_vars=incl_ancillary_vars,
        )

    if ctx.datasets_ctx.trace_perf:
        LOG.info(
            f"get_time_series: dataset id {ds_name},"
            f" variable {var_name}, "
            f"{len(results)} x {len(results[0])} values,"
            f" took {time_result.duration} seconds"
        )

    return results[0] if not is_collection and len(results) == 1 else results


def _get_time_series_for_geometries(
    dataset: xr.Dataset,
    var_name: str,
    geometries: list[shapely.geometry.base.BaseGeometry],
    agg_methods: set[str],
    grid_mapping: Optional[GridMapping] = None,
    start_date: Optional[np.datetime64] = None,
    end_date: Optional[np.datetime64] = None,
    max_valids: Optional[int] = None,
    incl_ancillary_vars: bool = False,
) -> TimeSeriesCollection:
    time_series_collection = []
    for geometry in geometries:
        time_series = _get_time_series_for_geometry(
            dataset,
            var_name,
            geometry,
            agg_methods,
            grid_mapping=grid_mapping,
            start_date=start_date,
            end_date=end_date,
            max_valids=max_valids,
            incl_ancillary_vars=incl_ancillary_vars,
        )
        time_series_collection.append(time_series)
    return time_series_collection


def _get_time_series_for_geometry(
    dataset: xr.Dataset,
    var_name: str,
    geometry: shapely.geometry.base.BaseGeometry,
    agg_methods: set[str],
    grid_mapping: Optional[GridMapping] = None,
    start_date: Optional[np.datetime64] = None,
    end_date: Optional[np.datetime64] = None,
    max_valids: Optional[int] = None,
    incl_ancillary_vars: bool = False,
) -> TimeSeries:
    if isinstance(geometry, shapely.geometry.Point):
        return _get_time_series_for_point(
            dataset,
            var_name,
            geometry,
            agg_methods,
            grid_mapping=grid_mapping,
            start_date=start_date,
            end_date=end_date,
            max_valids=max_valids,
            incl_ancillary_vars=incl_ancillary_vars,
        )

    time_series_ds = timeseries.get_time_series(
        dataset,
        grid_mapping=grid_mapping,
        geometry=geometry,
        var_names=[var_name],
        agg_methods=agg_methods,
        start_date=start_date,
        end_date=end_date,
        cube_asserted=True,
    )
    if time_series_ds is None:
        return []

    var_name, _ = split_var_assignment(var_name)
    key_to_var_names = {
        agg_method: f"{var_name}_{agg_method}" for agg_method in agg_methods
    }

    return collect_timeseries_result(
        time_series_ds, key_to_var_names, max_valids=max_valids
    )


def _get_time_series_for_point(
    dataset: xr.Dataset,
    var_name_or_assign: str,
    point: shapely.geometry.Point,
    agg_methods: set[str],
    grid_mapping: Optional[GridMapping] = None,
    start_date: Optional[np.datetime64] = None,
    end_date: Optional[np.datetime64] = None,
    max_valids: Optional[int] = None,
    incl_ancillary_vars: bool = False,
) -> TimeSeries:
    var_key = None
    if timeseries.AGG_MEAN in agg_methods:
        var_key = timeseries.AGG_MEAN
    elif timeseries.AGG_MEDIAN in agg_methods:
        var_key = timeseries.AGG_MEDIAN
    elif timeseries.AGG_MIN in agg_methods or timeseries.AGG_MAX in agg_methods:
        var_key = timeseries.AGG_MIN
    if not var_key:
        raise ApiError.BadRequest(
            "Aggregation methods must include one of" ' "mean", "median", "min", "max"'
        )

    var_names = [var_name_or_assign]
    var_name, var_expr = split_var_assignment(var_name_or_assign)
    key_to_var_names = {var_key: var_name}

    if incl_ancillary_vars and not var_expr:
        roles_to_anc_var_name_sets = find_ancillary_var_names(
            dataset, var_name, same_shape=True, same_dims=True
        )
        roles_to_anc_var_names = dict()
        for role, roles_to_anc_var_name_sets in roles_to_anc_var_name_sets.items():
            if role:
                roles_to_anc_var_names[role] = roles_to_anc_var_name_sets.pop()

        var_names += list(set(roles_to_anc_var_names.values()))
        for role, anc_var_name in roles_to_anc_var_names.items():
            key_to_var_names[role] = anc_var_name

    time_series_ds = timeseries.get_time_series(
        dataset,
        grid_mapping=grid_mapping,
        geometry=point,
        var_names=var_names,
        start_date=start_date,
        end_date=end_date,
        cube_asserted=True,
    )
    if time_series_ds is None:
        return []

    return collect_timeseries_result(
        time_series_ds, key_to_var_names, max_valids=max_valids
    )


def collect_timeseries_result(
    time_series_ds: xr.Dataset, key_to_var_names: dict[str, str], max_valids: int = None
) -> TimeSeries:
    _check_max_valids(max_valids)

    var_values_map = dict()
    for key, var_name in key_to_var_names.items():
        values = time_series_ds[var_name].values
        if np.issubdtype(values.dtype, np.floating):
            num_type = float
        elif np.issubdtype(values.dtype, np.integer):
            num_type = int
        elif np.issubdtype(values.dtype, np.dtype(bool)):
            num_type = bool
        else:
            raise ValueError(
                f"cannot convert {values.dtype}" f" into JSON-convertible value"
            )
        var_values_map[key] = [
            (num_type(v) if f else None) for f, v in zip(np.isfinite(values), values)
        ]

    time_values = [
        t.isoformat() + "Z"
        for t in pd.DatetimeIndex(time_series_ds.time.values).round("s")
    ]

    max_number_of_observations = time_series_ds.attrs.get(
        "max_number_of_observations", 1
    )
    num_times = len(time_values)
    time_series = []

    max_valids_is_positive = max_valids is not None and max_valids > 0
    if max_valids_is_positive:
        time_indexes = range(num_times - 1, -1, -1)
    else:
        time_indexes = range(num_times)

    for time_index in time_indexes:
        if len(time_series) == max_valids:
            break

        time_series_value = dict()
        all_null = True
        for key, var_values in var_values_map.items():
            var_value = var_values[time_index]
            if var_value is not None:
                all_null = False
            time_series_value[key] = var_value

        has_count = "count" in time_series_value
        no_obs = all_null or (has_count and time_series_value["count"] == 0)
        if no_obs and max_valids is not None:
            continue

        time_series_value["time"] = time_values[time_index]
        if has_count:
            time_series_value["count_tot"] = max_number_of_observations

        time_series.append(time_series_value)

    if max_valids_is_positive:
        time_series = time_series[::-1]

    return time_series


def _to_shapely_geometries(
    geo_json_geometries: list[GeoJsonGeometry],
) -> list[shapely.geometry.base.BaseGeometry]:
    geometries = []
    for geo_json_geometry in geo_json_geometries:
        try:
            geometry = shapely.geometry.shape(geo_json_geometry)
        except (TypeError, ValueError) as e:
            raise ApiError.BadRequest("Invalid GeoJSON geometry encountered") from e
        geometries.append(geometry)
    return geometries


def _to_geo_json_geometries(geo_json: GeoJsonObj) -> tuple[list[GeoJsonGeometry], bool]:
    is_collection = False
    if GeoJSON.is_feature(geo_json):
        geometry = _get_feature_geometry(geo_json)
        geometries = [geometry]
    elif GeoJSON.is_feature_collection(geo_json):
        is_collection = True
        features = GeoJSON.get_feature_collection_features(geo_json)
        geometries = (
            [_get_feature_geometry(feature) for feature in features] if features else []
        )
    elif GeoJSON.is_geometry_collection(geo_json):
        is_collection = True
        geometries = GeoJSON.get_geometry_collection_geometries(geo_json)
    elif GeoJSON.is_geometry(geo_json):
        geometries = [geo_json]
    else:
        raise ApiError.BadRequest("GeoJSON object expected")
    return geometries, is_collection


def _get_feature_geometry(feature: GeoJsonFeature) -> GeoJsonGeometry:
    geometry = GeoJSON.get_feature_geometry(feature)
    if geometry is None or not GeoJSON.is_geometry(geometry):
        raise ApiError.BadRequest("GeoJSON feature without geometry")
    return geometry


def _get_float_value(values: Optional[np.ndarray], index: int) -> Optional[float]:
    if values is None:
        return None
    value = float(values[index])
    return None if np.isnan(value) else value


def _check_max_valids(max_valids):
    if not (max_valids is None or max_valids == -1 or max_valids > 0):
        raise ApiError.BadRequest(
            f"max_valids must be either None, -1 or positive," f" was {max_valids}"
        )
