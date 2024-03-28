# The MIT License (MIT)
# Copyright (c) 2021-2022 by the xcube team and contributors
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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from typing import Collection, Optional, Tuple, Callable, Dict, Any, List, Mapping
from typing import Union

import cftime
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.assertions import assert_given
from xcube.util.timeindex import ensure_time_index_compatible

Bbox = Tuple[float, float, float, float]
TimeRange = Union[
    Tuple[Optional[str], Optional[str]],
    Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
]


def select_subset(
    dataset: xr.Dataset,
    *,
    var_names: Optional[Collection[str]] = None,
    bbox: Optional[Bbox] = None,
    time_range: Optional[TimeRange] = None,
    grid_mapping: Optional[GridMapping] = None,
):
    """Create a subset from *dataset* given *var_names*,
    *bbox*, *time_range*.

    This is a high-level convenience function that may invoke

    * `select_variables_subset()`
    * `select_spatial_subset()`
    * `select_temporal_subset()`

    Args:
        dataset: The dataset.
        var_names: Optional variable names.
        bbox: Optional bounding box in the dataset's CRS coordinate
            units.
        time_range: Optional time range
        grid_mapping: Optional dataset grid mapping.

    Returns:
        a subset of *dataset*, or unchanged *dataset* if no keyword-
        arguments are used.
    """
    if var_names is not None:
        dataset = select_variables_subset(dataset, var_names=var_names)
    if bbox is not None:
        dataset = select_spatial_subset(
            dataset, xy_bbox=bbox, grid_mapping=grid_mapping
        )
    if time_range is not None:
        dataset = select_temporal_subset(dataset, time_range=time_range)
    return dataset


def select_variables_subset(
    dataset: xr.Dataset, var_names: Optional[Collection[str]] = None
) -> xr.Dataset:
    """Select data variable from given *dataset* and create new dataset.

    Args:
        dataset: The dataset from which to select variables.
        var_names: The names of data variables to select.

    Returns:
        A new dataset. It is empty, if *var_names* is empty. It is
        *dataset*, if *var_names* is None.
    """
    if var_names is None:
        return dataset
    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset
    return dataset.drop_vars(dropped_variables)


def select_spatial_subset(
    dataset: xr.Dataset,
    ij_bbox: Optional[Tuple[int, int, int, int]] = None,
    ij_border: int = 0,
    xy_bbox: Optional[Tuple[float, float, float, float]] = None,
    xy_border: float = 0.0,
    grid_mapping: Optional[GridMapping] = None,
) -> Optional[xr.Dataset]:
    """Select a spatial subset of *dataset* for the
    bounding box *ij_bbox* or *xy_bbox*.

    *ij_bbox* or *xy_bbox* must not be given both.

    Args:
        xy_bbox: The bounding box in x,y coordinates.
        xy_border: Border in units of the x,y coordinates.
        dataset: Source dataset.
        ij_bbox: Bounding box (i_min, i_min, j_max, j_max) in pixel
            coordinates.
        ij_border: Extra border added to *ij_bbox* in number of pixels
        grid_mapping: Optional dataset grid mapping.

    Returns:
        Spatial dataset subset
    """

    if ij_bbox is None and xy_bbox is None:
        raise ValueError("One of ij_bbox and xy_bbox must be given")
    if ij_bbox and xy_bbox:
        raise ValueError("Only one of ij_bbox and xy_bbox can be given")

    if grid_mapping is None:
        grid_mapping = GridMapping.from_dataset(dataset)
    x_name, y_name = grid_mapping.xy_var_names
    x = dataset[x_name]
    y = dataset[y_name]

    if x.ndim == 1 and y.ndim == 1:
        # Hotfix für #981 and #985
        if xy_bbox:
            if y.values[0] < y.values[-1]:
                ds = dataset.sel(
                    **{
                        x_name: slice(xy_bbox[0] - xy_border, xy_bbox[2] + xy_border),
                        y_name: slice(xy_bbox[1] - xy_border, xy_bbox[3] + xy_border),
                    }
                )
            else:
                ds = dataset.sel(
                    **{
                        x_name: slice(xy_bbox[0] - xy_border, xy_bbox[2] + xy_border),
                        y_name: slice(xy_bbox[3] + xy_border, xy_bbox[1] - xy_border),
                    }
                )
            return ds
        else:
            return dataset.isel(
                **{
                    x_name: slice(ij_bbox[0] - ij_border, ij_bbox[2] + ij_border),
                    y_name: slice(ij_bbox[1] - ij_border, ij_bbox[3] + ij_border),
                }
            )
    else:
        if xy_bbox:
            ij_bbox = grid_mapping.ij_bbox_from_xy_bbox(
                xy_bbox, ij_border=ij_border, xy_border=xy_border
            )
            if ij_bbox[0] == -1:
                return None
        width, height = grid_mapping.size
        i_min, j_min, i_max, j_max = ij_bbox
        if i_min > 0 or j_min > 0 or i_max < width - 1 or j_max < height - 1:
            x_dim, y_dim = grid_mapping.xy_dim_names
            i_slice = slice(i_min, i_max + 1)
            j_slice = slice(j_min, j_max + 1)
            return dataset.isel({x_dim: i_slice, y_dim: j_slice})
        return dataset


def select_temporal_subset(
    dataset: xr.Dataset, time_range: TimeRange, time_name: str = "time"
) -> xr.Dataset:
    """Select a temporal subset from *dataset* given *time_range*.

    Args:
        dataset: The dataset. Must include time
        time_range: Time range given as two time stamps (start, end)
            that may be (ISO) strings or datetime objects.
        time_name: optional name of the time coordinate variable.
            Defaults to "time".

    Returns:

    """
    assert_given(time_range, "time_range")
    time_name = time_name or "time"
    if time_name not in dataset:
        raise ValueError(
            f"cannot compute temporal subset: variable"
            f' "{time_name}" not found in dataset'
        )
    time_1, time_2 = time_range
    time_1 = pd.to_datetime(time_1) if time_1 is not None else None
    time_2 = pd.to_datetime(time_2) if time_2 is not None else None
    if time_1 is None and time_2 is None:
        return dataset
    if time_2 is not None:
        delta = time_2 - time_2.floor("1D")
        if delta == pd.Timedelta("0 days 00:00:00"):
            time_2 += pd.Timedelta("1D")
    try:
        time_slice = ensure_time_index_compatible(
            dataset, slice(time_1, time_2), time_name
        )
        return dataset.sel({time_name or "time": time_slice})
    except TypeError:
        calendar = dataset.time.encoding.get("calendar")
        time_1 = cftime.datetime(
            time_1.year, time_1.month, time_1.day, calendar=calendar
        )
        time_2 = cftime.datetime(
            time_2.year, time_2.month, time_2.day, calendar=calendar
        )
        time_slice = ensure_time_index_compatible(
            dataset, slice(time_1, time_2), time_name
        )
        return dataset.sel({time_name or "time": time_slice})


_PREDICATE_SIGNATURE = (
    "predicate(" "slice_array: xr.DataArray, " "slice_info: Dict" ") -> bool"
)

Predicate = Callable[[xr.DataArray, Dict[str, Any]], bool]


def select_label_subset(
    dataset: xr.Dataset,
    dim: str,
    predicate: Union[Predicate, Mapping[str, Predicate]],
    use_dask: bool = False,
):
    """Select the labels in *dataset* along a given dimension *dim*
    using a predicate function *predicate* that is called for
    all variable slices for a current label.

    The *predicate* can also be provided as a mapping
    from variable names to dedicated predicate functions.

    The predicate function is called for all *dim* labels in *dataset*
    and for every variable that contains *dim*.

    If *predicate* returns False for any given label,
    that label will be dropped from dimension *dim*.

    Predicate functions are defined as follows:

    ```python
        def predicate(slice_array: xr.DataArray, slice_info: Dict) -> bool:
            ...
    ```

    Here, *slice_array* is a variable's array slice for the given label.
    The argument *slice_info* is a dictionary that contains the
    following keys:

    * var: str - name of the current variable.
    * dim: str - value of *dim*.
    * index: int - value for the current index within dimension *dim*.
    * label: Optional[xr.DataArray] - value for the current label
      within dimension *dim*.

    Note, the value of "label" will be None, if *dataset*
    does not contain a 1D-coordinate variable named *dim*.

    The following example selects only time labels
    from a 3-D (time, y, x) cube where the 2-D (y, x) images
    of variable "CHL" comprises more than 50% valid values:

    ```
    >>> chl_data = np.random.random((5, 10, 20))
    >>> chl_data = np.where(chl_data > 0.5, chl_data, np.nan)
    >>> ds = xr.Dataset({"CHL": (["time", "y", "x"], chl_data)})
    >>>
    >>> def is_valid_slice(slice_array, slice_label):
    >>>     return np.sum(np.isnan(slice_array)) / slice_array.size <= 0.5
    >>>
    >>> ds_subset = select_label_subset(ds, "time",
    >>>                                 predicate={"CHL": is_valid_slice})
    ```

    Args:
        dataset: The dataset.
        dim: The name of the dimension from which to select the labels.
        predicate: The predicate function or a mapping from variable
            names to variable-specific predicate functions.
        use_dask: Whether to use a Dask graph that will compute the
            validity of labels in parallel. For a large number of
            labels, very complex Dask graphs will result (every label is
            a node) whose overhead may compensate the performance gain.

    Returns:
        A new dataset with labels along *dim* selected by the
        *predicate*. If all labels are selected, *dataset* is returned
        without change.
    """
    if callable(predicate):
        predicate_lookup = {
            var_name: predicate
            for var_name, var in dataset.data_vars.items()
            if dim in var.dims
        }
    elif isinstance(predicate, Mapping):
        predicate_lookup = predicate
        for var_name, var_predicate in predicate_lookup.items():
            if not callable(var_predicate):
                raise TypeError(
                    f"predicate for variable {var_name!r}"
                    f" must be callable with"
                    f" signature {_PREDICATE_SIGNATURE}"
                )
    else:
        raise TypeError(
            f"predicate" f" must be callable with" f" signature {_PREDICATE_SIGNATURE}"
        )

    num_labels = dataset.dims[dim]

    valid_mask = [
        _is_label_valid(dataset, predicate_lookup, dim, index)
        for index in range(num_labels)
    ]

    if use_dask:
        valid_mask = da.stack(valid_mask).compute()

    dropped_indexes = [i for i in range(num_labels) if not valid_mask[i]]
    if not dropped_indexes:
        return dataset

    return dataset.drop_isel({dim: dropped_indexes})


def _is_label_valid(
    dataset: xr.Dataset, predicate_lookup: Mapping[str, Predicate], dim: str, index: int
) -> da.Array:
    label = dataset[dim][index] if dim in dataset else None
    results: List[da.Array] = []
    for var_name, var in dataset.data_vars.items():
        if dim in var.dims:
            predicate = predicate_lookup.get(var_name)
            if predicate is not None:
                slice_array = var.isel({dim: index})
                slice_info = dict(var=var_name, dim=dim, index=index, label=label)
                result = predicate(slice_array, slice_info)
                if isinstance(result, xr.DataArray):
                    result = result.data
                if isinstance(result, da.Array):
                    results.append(result)
                else:
                    results.append(da.from_array(result))
    if len(results) == 0:
        return da.from_array(True)
    elif len(results) == 1:
        return results[0]
    else:
        return da.all(da.stack(results))
