from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
import shapely

from xcube.constants import LOG
from xcube.core.geom import get_dataset_geometry
from xcube.core.geom import mask_dataset_by_geometry
from xcube.core.varexpr import VarExprContext
from xcube.core.varexpr import split_var_assignment
from xcube.server.api import ApiError
from xcube.util.perf import measure_time_cm
from .context import StatisticsContext


NAN_RESULT = {"count": 0}
DEFAULT_BIN_COUNT = 100


def compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name: str,
    geo_json: dict[str, Any],
    params: Mapping[str, str],
):
    params = dict(params)
    try:
        time_label = params.pop("time")
    except KeyError:
        raise ApiError.BadRequest("Missing query parameter 'time'")
    trace_perf = params.pop("debug", "1" if ctx.datasets_ctx.trace_perf else "0") == "1"
    measure_time = measure_time_cm(logger=LOG, disabled=not trace_perf)
    with measure_time("Computing statistics"):
        return _compute_statistics(
            ctx, ds_id, var_name, time_label, geo_json, DEFAULT_BIN_COUNT
        )


def _compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name_or_assign: str,
    time_label: str,
    geo_json: dict[str, Any],
    bin_count: int,
):
    ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_id)
    dataset = ml_dataset.get_dataset(0)
    grid_mapping = ml_dataset.grid_mapping

    try:
        time = np.array(time_label, dtype=dataset.time.dtype)
    except (TypeError, ValueError) as e:
        raise ApiError.BadRequest("Invalid 'time'") from e

    try:
        geometry = shapely.geometry.shape(geo_json)
    except (TypeError, ValueError, AttributeError) as e:
        raise ApiError.BadRequest("Invalid GeoJSON geometry encountered") from e

    dataset = dataset.sel(time=time, method="nearest")

    x_name, y_name = grid_mapping.xy_dim_names
    if isinstance(geometry, shapely.geometry.Point):
        bounds = get_dataset_geometry(dataset)
        if not bounds.contains(geometry):
            return NAN_RESULT
        indexers = {x_name: geometry.x, y_name: geometry.y}
        variable = _get_dataset_variable(var_name_or_assign, dataset)
        value = variable.sel(**indexers, method="Nearest").values
        if np.isnan(value):
            return NAN_RESULT
        return {
            "count": 1,
            "minimum": float(value),
            "maximum": float(value),
            "mean": float(value),
            "deviation": 0.0,
        }

    dataset = mask_dataset_by_geometry(dataset, geometry)
    if dataset is None:
        return NAN_RESULT

    variable = _get_dataset_variable(var_name_or_assign, dataset)

    count = int(np.count_nonzero(~np.isnan(variable)))
    if count == 0:
        return NAN_RESULT

    # note, casting to float forces intended computation
    minimum = float(variable.min())
    maximum = float(variable.max())
    h_values, h_edges = np.histogram(
        variable, bin_count, range=(minimum, maximum), density=True
    )

    return {
        "count": count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(variable.mean()),
        "deviation": float(variable.std()),
        "histogram": {
            "values": [float(v) for v in h_values],
            "edges": [float(v) for v in h_edges],
        },
    }


def _get_dataset_variable(var_name_or_assign: str, dataset: xr.Dataset) -> xr.DataArray:
    var_name, var_expr = split_var_assignment(var_name_or_assign)
    if var_expr:
        variable = VarExprContext(dataset).evaluate(var_expr)
        variable.name = var_name
    else:
        var_name = var_name_or_assign
        variable = dataset[var_name]

    return variable
