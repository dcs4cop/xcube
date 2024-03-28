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

import warnings
from datetime import datetime
from typing import Optional, Sequence, Union, Tuple

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from jdcal import jd2gcal

from xcube.core.gridmapping import GridMapping
from xcube.core.timecoord import get_timestamp_from_string
from xcube.core.timecoord import get_timestamps_from_string
from xcube.core.verify import assert_cube

DatetimeTypes = np.datetime64, cftime.datetime, datetime
Datetime = Union[np.datetime64, cftime.datetime, datetime]


class DatasetIsNotACubeError(BaseException):
    """Raised, if at least a subset of a dataset's variables
    have data cube dimensions ('time' , [...], y_dim_name, x_dim_name),
    where y_dim_name and x_dim_name are determined by a dataset's
    `GridMapping`.
    """

    pass


def cubify_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the geo- and time-coding upon opening the given
    dataset w.r.t. a common (CF-compatible) convention.

    Will throw a value error if the dataset could not not be
    converted to a cube.
    """
    ds = normalize_dataset(ds)
    return assert_cube(ds)


def normalize_dataset(
    ds: xr.Dataset, reverse_decreasing_lat: bool = False
) -> xr.Dataset:
    """Normalize the geo- and time-coding upon opening the given
    dataset w.r.t. a common (CF-compatible) convention.

    That is,
    * variables named "latitude" will be renamed to "lat";
    * variables named "longitude" or "long" will be renamed to "lon";

    Then, for equi-rectangular grids,
    * Remove 2D "lat" and "lon" variables;
    * Two new 1D coordinate variables "lat" and "lon" will be generated
    from original 2D forms.

    Then, if no coordinate variable time is present but the CF attributes
    "time_coverage_start"
    and optionally "time_coverage_end" are given, a scalar time dimension
    and coordinate variable
    will be generated.

    Finally, it will be ensured that a "time" coordinate variable will
    be of type *datetime*.

    Args:
        ds: The dataset to normalize.
        reverse_decreasing_lat: Whether decreasing latitude values shall
            be normalized so they are increasing

    Returns:
        The normalized dataset, or the original dataset, if it is
        already "normal".
    """
    ds = _normalize_zonal_lat_lon(ds)
    ds = normalize_coord_vars(ds)
    ds = _normalize_lat_lon(ds)
    ds = _normalize_lat_lon_2d(ds)
    ds = _normalize_dim_order(ds)
    ds = _normalize_lon_360(ds)
    if reverse_decreasing_lat:
        ds = _reverse_decreasing_lat(ds)
    ds = normalize_missing_time(ds)
    ds = _normalize_jd2datetime(ds)
    return ds


def encode_cube(
    cube: xr.Dataset,
    grid_mapping: Optional[GridMapping] = None,
    non_cube_subset: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """Encode a *cube* with its *grid_mapping*, and additional variables in
    *non_cube_subset* into a new dataset.

    This is the inverse of the operation `decode_cube()`:

    ```python
        cube, gm, non_cube = decode_cube(dataset)
        dataset = encode_cube(cube, gm, non_cube)
    ```

    The returned data cube comprises all variables in *cube*,
    whose dimensions should be ("time" , [...], y_dim_name, x_dim_name),
    and where y_dim_name, x_dim_name are defined by *grid_mapping*, if
    given.
    If *grid_mapping* is not geographic, a new variable "crs" will
    be added that holds CF-compliant attributes which encode the
    cube's spatial CRS. *non_cube_subset*, if given may be used
    to add non-cube variables the to resulting dataset.

    Args:
        cube: data cube dataset, whose dimensions should be ("time" ,
            [...], y_dim_name, x_dim_name)
        grid_mapping: Optional grid mapping for *cube*.
        non_cube_subset: An optional dataset providing non-cube data
            variables.

    Returns:
        A Dataset instance
    """
    if non_cube_subset is not None:
        dataset = cube.assign(**non_cube_subset.data_vars)
    else:
        dataset = cube

    if grid_mapping is None:
        return dataset

    if (
        grid_mapping.crs.is_geographic
        and grid_mapping.is_regular
        and grid_mapping.xy_dim_names == ("lon", "lat")
        and grid_mapping.xy_var_names == ("lon", "lat")
    ):
        # No need to add CRS variable
        return dataset

    return dataset.assign(crs=xr.DataArray(0, attrs=grid_mapping.crs.to_cf()))


def decode_cube(
    dataset: xr.Dataset,
    normalize: bool = False,
    force_copy: bool = False,
    force_non_empty: bool = False,
    force_geographic: bool = False,
) -> Tuple[xr.Dataset, GridMapping, xr.Dataset]:
    """Decode a *dataset* into a cube variable subset, a grid mapping, and
    the non-cube variables of *dataset*.

    This is the inverse of the operation `encode_cube()`:

    ```python
        cube, gm, non_cube = decode_cube(dataset)
        dataset = encode_cube(cube, gm, non_cube)
    ```

    The returned data cube comprises all variables in *dataset*
    whose dimensions are ("time" , [...], y_dim_name, x_dim_name).
    Here y_dim_name and x_dim_name are determined by the
    `GridMapping` derived from *dataset*.

    Args:
        dataset: The dataset.
        normalize: Whether to normalize the *dataset*, before the cube
            subset is determined. If normalisation fails, the cube
            subset is created from *dataset*.
        force_copy: whether to create a copy of this dataset even if
            this dataset is identical to its cube subset.
        force_non_empty: whether the resulting cube must have at least
            one data variable. If True, a `DatasetIsNotACubeError`
            may be raised.
        force_geographic: whether a geographic grid mapping is required.
            If True, a `DatasetIsNotACubeError` may be raised.

    Returns:
        A 3-tuple comprising the data cube subset of *dataset* the
        cube's grid mapping, and the remaining variables.

    Raises:
        DatasetIsNotACubeError: If it is not possible to determine a
            data cube subset from *dataset*.
    """
    if normalize:
        try:
            dataset = normalize_dataset(dataset)
        except ValueError:
            pass
    try:
        grid_mapping = GridMapping.from_dataset(dataset, tolerance=1e-4)
    except ValueError as e:
        raise DatasetIsNotACubeError(f"Failed to detect grid mapping:" f" {e}") from e
    if force_geographic and not grid_mapping.crs.is_geographic:
        # We will need to overcome this soon!
        raise DatasetIsNotACubeError(
            f"Grid mapping must use geographic CRS,"
            f" but was {grid_mapping.crs.name!r}"
        )

    x_dim_name, y_dim_name = grid_mapping.xy_dim_names
    time_name = "time"

    cube_vars = set()
    dropped_vars = set()
    for var_name, var in dataset.data_vars.items():
        if (
            var.ndim >= 3
            and var.dims[0] == time_name
            and var.dims[-2] == y_dim_name
            and var.dims[-1] == x_dim_name
            and np.all(var.shape)
            and var.shape[-2] > 1
            and var.shape[-1] > 1
        ):
            cube_vars.add(var_name)
        else:
            dropped_vars.add(var_name)

    if force_non_empty and len(dropped_vars) == len(dataset.data_vars):
        # Or just return empty dataset?
        raise DatasetIsNotACubeError(
            f"No variables found with dimensions"
            f" ({time_name!r}, [...]"
            f" {y_dim_name!r}, {x_dim_name!r})"
            f" or dimension sizes too small"
        )

    if not force_copy and not dropped_vars:
        # Pure cube!
        return dataset, grid_mapping, xr.Dataset()

    cube = dataset.drop_vars(dropped_vars).assign_attrs(dataset.attrs)

    return (cube, grid_mapping, dataset.drop_vars(cube_vars))


def _normalize_zonal_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """In case that the dataset only contains lat_centers and is a zonal mean dataset,
    the longitude dimension created and filled with the variable value of certain latitude.

    Args:
        ds: some xarray dataset

    Returns:
        a normalized xarray dataset
    """

    if "latitude_centers" not in ds.coords or "lon" in ds.coords:
        return ds

    ds_zonal = ds.copy()
    resolution = ds.latitude_centers[1].values - ds.latitude_centers[0].values
    ds_zonal = ds_zonal.assign_coords(
        lon=[i + (resolution / 2) for i in np.arange(-180.0, 180.0, resolution)]
    )

    for var in ds_zonal.data_vars:
        if "latitude_centers" in ds_zonal[var].dims:
            ds_zonal[var] = xr.concat([ds_zonal[var] for _ in ds_zonal.lon], "lon")
            ds_zonal[var]["lon"] = ds_zonal.lon
            var_dims = ds_zonal[var].attrs.get("dimensions", [])
            lat_center_index = var_dims.index("latitude_centers")
            var_dims.remove("latitude_centers")
            var_dims.append("lat")
            var_dims.append("lon")
            var_chunk_sizes = ds_zonal[var].attrs.get("chunk_sizes", [])
            lat_chunk_size = var_chunk_sizes[lat_center_index]
            del var_chunk_sizes[lat_center_index]
            var_chunk_sizes.append(lat_chunk_size)
            var_chunk_sizes.append(ds_zonal.lon.size)
    ds_zonal = ds_zonal.rename_dims({"latitude_centers": "lat"})
    ds_zonal = ds_zonal.drop_vars("latitude_centers")
    ds_zonal = ds_zonal.assign_coords(lat=ds.latitude_centers.values)
    ds_zonal = ds_zonal.transpose(..., "lat", "lon")

    has_lon_bnds = "lon_bnds" in ds_zonal.coords or "lon_bnds" in ds_zonal
    if not has_lon_bnds:
        lon_values = [
            [i - (resolution / 2), i + (resolution / 2)] for i in ds_zonal.lon.values
        ]
        ds_zonal = ds_zonal.assign_coords(
            lon_bnds=xr.DataArray(lon_values, dims=["lon", "bnds"])
        )
    has_lat_bnds = "lat_bnds" in ds_zonal.coords or "lat_bnds" in ds_zonal
    if not has_lat_bnds:
        lat_values = [
            [i - (resolution / 2), i + (resolution / 2)] for i in ds_zonal.lat.values
        ]
        ds_zonal = ds_zonal.assign_coords(
            lat_bnds=xr.DataArray(lat_values, dims=["lat", "bnds"])
        )

    ds_zonal.lon.attrs["bounds"] = "lon_bnds"
    ds_zonal.lon.attrs["long_name"] = "longitude"
    ds_zonal.lon.attrs["standard_name"] = "longitude"
    ds_zonal.lon.attrs["units"] = "degrees_east"

    ds_zonal.lat.attrs["bounds"] = "lat_bnds"
    ds_zonal.lat.attrs["long_name"] = "latitude"
    ds_zonal.lat.attrs["standard_name"] = "latitude"
    ds_zonal.lat.attrs["units"] = "degrees_north"

    return ds_zonal


def _normalize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables named 'longitude' or 'long' to 'lon', and 'latitude' to 'lon'.

    Args:
        ds: some xarray dataset

    Returns:
        a normalized xarray dataset, or the original one
    """
    lat_name = get_lat_dim_name_impl(ds)
    lon_name = get_lon_dim_name_impl(ds)

    name_dict = dict()
    if lat_name and "lat" not in ds:
        name_dict[lat_name] = "lat"

    if lon_name and "lon" not in ds:
        name_dict[lon_name] = "lon"

    if name_dict:
        ds = ds.rename(name_dict)

    return ds


def _normalize_lat_lon_2d(ds: xr.Dataset) -> xr.Dataset:
    """Detect 2D 'lat', 'lon' variables that span a equi-rectangular grid. Then:
    Drop original 'lat', 'lon' variables
    Rename original dimensions names of 'lat', 'lon' variables, usually ('y', 'x'), to
    ('lat', 'lon').
    Insert new 1D 'lat', 'lon' coordinate variables with dimensions 'lat' and 'lon', respectively.

    Args:
        ds: some xarray dataset

    Returns:
        a normalized xarray dataset, or the original one
    """
    if not ("lat" in ds and "lon" in ds):
        return ds

    lat_var = ds["lat"]
    lon_var = ds["lon"]

    lat_dims = lat_var.dims
    lon_dims = lon_var.dims
    if lat_dims != lon_dims:
        return ds

    spatial_dims = lon_dims
    if len(spatial_dims) != 2:
        return ds

    x_dim_name = spatial_dims[-1]
    y_dim_name = spatial_dims[-2]

    lat_data_1 = lat_var[:, 0]
    lat_data_2 = lat_var[:, -1]
    lon_data_1 = lon_var[0, :]
    lon_data_2 = lon_var[-1, :]

    equal_lat = np.allclose(lat_data_1, lat_data_2, equal_nan=True)
    equal_lon = np.allclose(lon_data_1, lon_data_2, equal_nan=True)

    # Drop lat lon in any case. If note qual_lat and equal_lon subset_spatial_impl will
    # subsequently fail with a ValidationError

    ds = ds.drop_vars(["lon", "lat"])

    if not (equal_lat and equal_lon):
        return ds

    ds = ds.rename(
        {
            x_dim_name: "lon",
            y_dim_name: "lat",
        }
    )

    ds = ds.assign_coords(lon=np.array(lon_data_1), lat=np.array(lat_data_1))

    return ds


def _normalize_lon_360(ds: xr.Dataset) -> xr.Dataset:
    """Fix the longitude of the given dataset `ds` so that it ranges from -180 to +180 degrees.

    Args:
        ds: The dataset whose longitudes may be given in the range 0 to
            360.

    Returns:
        The fixed dataset or the original dataset.
    """

    if "lon" not in ds.coords:
        return ds

    lon_var = ds.coords["lon"]

    if len(lon_var.shape) != 1:
        return ds

    lon_size = lon_var.shape[0]
    if lon_size < 2:
        return ds

    lon_size_05 = lon_size // 2
    lon_values = lon_var.values
    if not np.any(lon_values[lon_size_05:] > 180.0):
        return ds

    delta_lon = lon_values[1] - lon_values[0]

    var_names = [var_name for var_name in ds.data_vars]

    ds = ds.assign_coords(
        lon=xr.DataArray(
            np.linspace(-180.0 + 0.5 * delta_lon, +180.0 - 0.5 * delta_lon, lon_size),
            dims=ds["lon"].dims,
            attrs=dict(
                long_name="longitude", standard_name="longitude", units="degrees east"
            ),
        )
    )

    ds = adjust_spatial_attrs(ds, True)

    new_vars = dict()
    for var_name in var_names:
        var = ds[var_name]
        if "lon" in var.dims:
            new_var = var.roll(lon=lon_size_05, roll_coords=False)
            new_var.encoding.update(var.encoding)
            new_vars[var_name] = new_var

    return ds.assign(**new_vars)


def _reverse_decreasing_lat(ds: xr.Dataset) -> xr.Dataset:
    """In case the latitude decreases, invert it

    Args:
        ds: some xarray dataset

    Returns:
        a normalized xarray dataset
    """
    try:
        if _is_lat_decreasing(ds.lat):
            ds = ds.sel(lat=slice(None, None, -1))
    except AttributeError:
        # The dataset doesn't have 'lat', probably not geospatial
        pass
    except ValueError:
        # The dataset still has an ND 'lat' array
        pass
    return ds


def _normalize_jd2datetime(ds: xr.Dataset) -> xr.Dataset:
    """Convert the time dimension of the given dataset from Julian date to
    datetime.

    Args:
        ds: Dataset on which to run conversion
    """

    try:
        time = ds.time
    except AttributeError:
        return ds

    try:
        units = time.units
    except AttributeError:
        units = None

    try:
        long_name = time.long_name
    except AttributeError:
        long_name = None

    if units:
        units = units.lower().strip()

    if long_name:
        units = long_name.lower().strip()

    units = units or long_name

    if not units or units != "time in julian days":
        return ds

    ds = ds.copy()
    # Decode JD time
    # noinspection PyTypeChecker
    tuples = [jd2gcal(x, 0) for x in ds.time.values]
    # Replace JD time with datetime
    ds["time"] = [datetime(x[0], x[1], x[2]) for x in tuples]
    # Adjust attributes
    ds.time.attrs["long_name"] = "time"
    ds.time.attrs["calendar"] = "standard"

    return ds


def normalize_coord_vars(ds: xr.Dataset) -> xr.Dataset:
    """Turn potential coordinate variables from data variables into coordinate variables.

    Any data variable is considered a coordinate variable

    * whose name is its only dimension name;
    * whose number of dimensions is two and where the first dimension name is also a variable name
        and whose last dimension is named "bnds".

    Args:
        ds: The dataset

    Returns:
        The same dataset or a shallow copy with potential coordinate
        variables turned into coordinate variables.
    """

    if "bnds" not in ds.dims:
        return ds

    coord_var_names = set()
    for data_var_name in ds.data_vars:
        data_var = ds.data_vars[data_var_name]
        if is_coord_var(ds, data_var):
            coord_var_names.add(data_var_name)

    if not coord_var_names:
        return ds

    old_ds = ds
    ds = old_ds.drop_vars(coord_var_names)
    ds = ds.assign_coords(
        **{
            bounds_var_name: old_ds[bounds_var_name]
            for bounds_var_name in coord_var_names
        }
    )

    return ds


def normalize_missing_time(ds: xr.Dataset) -> xr.Dataset:
    """Add a time coordinate variable and their associated bounds coordinate variables
    if either temporal CF attributes `time_coverage_start` and `time_coverage_end`
    are given or time information can be extracted from the file name but the time dimension is
    missing.

    In case the time information is given by a variable called 't' instead of 'time', it will be
    renamed into 'time'.

    The new time coordinate variable will be named `time` with dimension ['time'] and shape [1].
    The time bounds coordinates variable will be named `time_bnds`
    with dimensions ['time', 'bnds'] and shape [1,2].
    Both are of data type `datetime64`.

    Args:
        ds: Dataset to adjust

    Returns:
        Adjusted dataset
    """
    time_coverage_start, time_coverage_end = _get_time_coverage_from_ds(ds)

    if not time_coverage_start and not time_coverage_end:
        # Can't do anything
        return ds

    time = _get_valid_time_coord(ds, "time")
    if time is None:
        time = _get_valid_time_coord(ds, "t")
        if time is not None:
            ds = ds.rename_vars({"t": "time"})
            ds = ds.assign_coords(time=("time", ds.time.data))
            time = ds.time
    if time is not None:
        if not time.dims:
            ds = ds.drop_vars("time")
        elif len(time.dims) == 1:
            time_dim_name = time.dims[0]
            is_time_used_as_dim = any(
                [(time_dim_name in ds[var_name].dims) for var_name in ds.data_vars]
            )
            if is_time_used_as_dim:
                # It seems we already have valid time coordinates
                return ds
            time_bnds_var_name = time.attrs.get("bounds")
            if time_bnds_var_name in ds:
                ds = ds.drop_vars(time_bnds_var_name)
            ds = ds.drop_vars("time")
            ds = ds.drop_vars(
                [
                    var_name
                    for var_name in ds.coords
                    if time_dim_name in ds.coords[var_name].dims
                ]
            )

    try:
        ds = ds.expand_dims("time")
    except BaseException as e:
        warnings.warn(f"failed to add time dimension: {e}")
    if time_coverage_start and time_coverage_end:
        time_value = time_coverage_start + 0.5 * (
            time_coverage_end - time_coverage_start
        )
    else:
        time_value = time_coverage_start or time_coverage_end
    new_coord_vars = dict(time=xr.DataArray([time_value], dims=["time"]))
    if time_coverage_start and time_coverage_end:
        has_time_bnds = "time_bnds" in ds.coords or "time_bnds" in ds
        if not has_time_bnds:
            new_coord_vars.update(
                time_bnds=xr.DataArray(
                    [[time_coverage_start, time_coverage_end]], dims=["time", "bnds"]
                )
            )
    ds = ds.assign_coords(**new_coord_vars)
    ds.coords["time"].attrs["long_name"] = "time"
    ds.coords["time"].attrs["standard_name"] = "time"
    ds.coords["time"].encoding["units"] = "days since 1970-01-01"
    if "time_bnds" in ds.coords:
        ds.coords["time"].attrs["bounds"] = "time_bnds"
        ds.coords["time_bnds"].attrs["long_name"] = "time"
        ds.coords["time_bnds"].attrs["standard_name"] = "time"
        ds.coords["time_bnds"].encoding["units"] = "days since 1970-01-01"

    return ds


def _get_valid_time_coord(ds: xr.Dataset, name: str) -> Optional[xr.DataArray]:
    if name in ds:
        time = ds[name]
        if time.size > 0 and time.ndim == 1 and _is_supported_time_dtype(time.dtype):
            return time
    return None


def _is_supported_time_dtype(dtype: np.dtype) -> bool:
    return any(np.issubdtype(dtype, t) for t in DatetimeTypes)


def _get_time_coverage_from_ds(ds: xr.Dataset) -> (pd.Timestamp, pd.Timestamp):
    time_coverage_start = ds.attrs.get("time_coverage_start")
    if time_coverage_start is not None:
        time_coverage_start = get_timestamp_from_string(time_coverage_start)

    time_coverage_end = ds.attrs.get("time_coverage_end")
    if time_coverage_end is not None:
        time_coverage_end = get_timestamp_from_string(time_coverage_end)

    if time_coverage_start or time_coverage_end:
        return time_coverage_start, time_coverage_end

    filename = ds.encoding.get("source", "").split("/")[-1]
    return get_timestamps_from_string(filename)


# TODO (forman): replace adjust_spatial_attrs function
#   by another on that uses a dataset's grid mapping,
#   see code in xcube/core/gen2/local/mladjuster.py


def adjust_spatial_attrs(ds: xr.Dataset, allow_point: bool = False) -> xr.Dataset:
    """Adjust the global spatial attributes of the dataset by doing some
    introspection of the dataset and adjusting the appropriate attributes
    accordingly.

    In case the determined attributes do not exist in the dataset, these will
    be added.

    For more information on suggested global attributes see
    `Attribute Convention for Data Discovery
    <http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery>`_

    Args:
        ds: Dataset to adjust
        allow_point: Whether to accept single point cells

    Returns:
        Adjusted dataset
    """

    copied = False

    for dim in ("lon", "lat"):
        geo_spatial_attrs = get_geo_spatial_attrs_from_var(
            ds, dim, allow_point=allow_point
        )
        if geo_spatial_attrs:
            # Copy any new attributes into the shallow Dataset copy
            for key in geo_spatial_attrs:
                if geo_spatial_attrs[key] is not None:
                    if not copied:
                        ds = ds.copy()
                        copied = True
                    ds.attrs[key] = geo_spatial_attrs[key]

    lon_min = ds.attrs.get("geospatial_lon_min")
    lat_min = ds.attrs.get("geospatial_lat_min")
    lon_max = ds.attrs.get("geospatial_lon_max")
    lat_max = ds.attrs.get("geospatial_lat_max")
    # TODO (forman): add geospatial_lon/lat_resolution

    if (
        lon_min is not None
        and lat_min is not None
        and lon_max is not None
        and lat_max is not None
    ):
        if not copied:
            ds = ds.copy()

        ds.attrs["geospatial_bounds_crs"] = "CRS84"
        ds.attrs["geospatial_bounds"] = (
            f"POLYGON(("
            f"{lon_min} {lat_min}, "
            f"{lon_min} {lat_max}, "
            f"{lon_max} {lat_max}, "
            f"{lon_max} {lat_min}, "
            f"{lon_min} {lat_min}))"
        )

    return ds


def is_coord_var(ds: xr.Dataset, var: xr.DataArray):
    if len(var.dims) == 1 and var.name == var.dims[0]:
        return True
    return is_bounds_var(ds, var)


def is_bounds_var(ds: xr.Dataset, var: xr.DataArray):
    if len(var.dims) == 2 and var.shape[1] == 2 and var.dims[1] == "bnds":
        coord_name = var.dims[0]
        return coord_name in ds
    return False


def get_geo_spatial_attrs_from_var(
    ds: xr.Dataset, var_name: str, allow_point: bool = False
) -> Optional[dict]:
    """Get spatial boundaries, resolution and units of the given dimension of the given
    dataset. If the 'bounds' are explicitly defined, these will be used for
    boundary calculation, otherwise it will rest purely on information gathered
    from 'dim' itself.

    Args:
        ds: The dataset
        var_name: The variable/dimension name.
        allow_point: True, if it is ok to have no actual spatial extent.

    Returns:
        A dictionary {'attr_name': attr_value}
    """

    if var_name not in ds:
        return None

    var = ds[var_name]
    res_name = "geospatial_{}_resolution".format(var_name)
    min_name = "geospatial_{}_min".format(var_name)
    max_name = "geospatial_{}_max".format(var_name)
    units_name = "geospatial_{}_units".format(var_name)

    if "bounds" in var.attrs:
        # According to CF Conventions the corresponding 'bounds' coordinate variable name
        # should be in the attributes of the coordinate variable
        bnds_name = var.attrs["bounds"]
    else:
        # If 'bounds' attribute is missing, the bounds coordinate variable may be named
        # "<dim>_bnds"
        bnds_name = "%s_bnds" % var_name

    dim_var = None

    if bnds_name in ds:
        bnds_var = ds[bnds_name]
        if (
            len(bnds_var.shape) == 2
            and bnds_var.shape[0] > 0
            and bnds_var.shape[1] == 2
        ):
            dim_var = bnds_var
            dim_res = abs(bnds_var[0, 1] - bnds_var[0, 0])
            if bnds_var.shape[0] > 1:
                dim_min = min(bnds_var[0, 0], bnds_var[-1, 1])
                dim_max = max(bnds_var[0, 0], bnds_var[-1, 1])
            else:
                dim_min = min(bnds_var[0, 0], bnds_var[0, 1])
                dim_max = max(bnds_var[0, 0], bnds_var[0, 1])

    if dim_var is None:
        if len(var.shape) == 1 and var.shape[0] > 0:
            if var.shape[0] > 1:
                dim_var = var
                dim_res = abs(var[1] - var[0])
                dim_min = min(var[0], var[-1]) - 0.5 * dim_res
                dim_max = max(var[0], var[-1]) + 0.5 * dim_res
            elif var.size == 1:
                if res_name in ds.attrs:
                    dim_res = ds.attrs[res_name]
                    if isinstance(dim_res, str):
                        # remove any units from string
                        dim_res = (
                            dim_res.replace("degree", "")
                            .replace("deg", "")
                            .replace("°", "")
                            .strip()
                        )
                    try:
                        dim_res = float(dim_res)
                        dim_var = var
                        # Consider extent in metadata if provided
                        dim_min = var[0] - 0.5 * dim_res
                        dim_max = var[0] + 0.5 * dim_res
                    except ValueError:
                        if allow_point:
                            dim_var = var
                            # Actually a point with no extent
                            dim_res = 0.0
                            dim_min = var[0]
                            dim_max = var[0]
                elif allow_point:
                    dim_var = var
                    # Actually a point with no extent
                    dim_res = 0.0
                    dim_min = var[0]
                    dim_max = var[0]

    if dim_var is None:
        # Cannot determine spatial extent for variable/dimension var_name
        return None

    if "units" in var.attrs:
        dim_units = var.attrs["units"]
    else:
        dim_units = None

    geo_spatial_attrs = dict()
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[res_name] = float(dim_res)
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[min_name] = float(dim_min)
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[max_name] = float(dim_max)
    geo_spatial_attrs[units_name] = dim_units

    return geo_spatial_attrs


def get_lon_dim_name_impl(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """Get the name of the longitude dimension.

    Args:
        ds: An xarray Dataset

    Returns:
        the name or None
    """
    return _get_dim_name(ds, ["lon", "longitude", "long"])


def get_lat_dim_name_impl(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """Get the name of the latitude dimension.

    Args:
        ds: An xarray Dataset

    Returns:
        the name or None
    """
    return _get_dim_name(ds, ["lat", "latitude"])


def _get_dim_name(
    ds: Union[xr.Dataset, xr.DataArray], possible_names: Sequence[str]
) -> Optional[str]:
    for name in possible_names:
        if name in ds.dims:
            return name
    return None


def _is_lat_decreasing(lat: xr.DataArray) -> bool:
    """Determine if the latitude is decreasing"""
    if lat[0] > lat[-1]:
        return True

    return False


def _normalize_dim_order(ds: xr.Dataset) -> xr.Dataset:
    copy_created = False

    for var_name in ds.data_vars:
        var = ds[var_name]
        dim_names = list(var.dims)
        num_dims = len(dim_names)
        if num_dims == 0:
            continue

        must_transpose = False

        if "time" in dim_names:
            time_index = dim_names.index("time")
            if time_index > 0:
                must_transpose = _swap_pos(dim_names, time_index, 0)

        if num_dims >= 2 and "lat" in dim_names and "lon" in dim_names:
            lat_index = dim_names.index("lat")
            if lat_index != num_dims - 2:
                must_transpose = _swap_pos(dim_names, lat_index, -2)
            lon_index = dim_names.index("lon")
            if lon_index != num_dims - 1:
                must_transpose = _swap_pos(dim_names, lon_index, -1)

        if must_transpose:
            if not copy_created:
                ds = ds.copy()
                copy_created = True
            ds[var_name] = var.transpose(*dim_names)
            if var.encoding and "chunksize" in ds[var_name].data:
                ds[var_name].encoding["chunks"] = ds[var_name].data.chunksize

    return ds


def _swap_pos(lst, i1, i2):
    e1, e2 = lst[i1], lst[i2]
    lst[i2], lst[i1] = e1, e2
    return True
