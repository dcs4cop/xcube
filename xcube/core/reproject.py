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

import warnings
from typing import Tuple, List, Union, Dict, Any

import deprecated
import numpy as np
from osgeo import gdal
import xarray as xr

from xcube.constants import CRS_WKT_EPSG_4326, GLOBAL_GEO_EXTENT

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

DEFAULT_RESAMPLING = "Nearest"
DEFAULT_TP_RESAMPLING = "Bilinear"

CoordRange = Tuple[float, float, float, float]


@deprecated.deprecated(
    "not maintained anymore," " instead use methods from xcube.core.resampling",
    version="1.4.2",
)
def reproject_xy_to_wgs84(
    src_dataset: xr.Dataset,
    src_xy_var_names: Tuple[str, str],
    src_xy_tp_var_names: Tuple[str, str] = None,
    src_xy_crs: str = None,
    src_xy_gcp_step: Union[int, Tuple[int, int]] = 10,
    src_xy_tp_gcp_step: Union[int, Tuple[int, int]] = 1,
    dst_size: Tuple[int, int] = None,
    dst_region: CoordRange = None,
    dst_resampling: Union[str, Dict[str, str]] = DEFAULT_RESAMPLING,
    include_xy_vars: bool = False,
    include_non_spatial_vars: bool = False,
) -> xr.Dataset:
    """Reprojection of xarray datasets with 2D geo-coding,
    e.g., with variables lon(y,x), lat(y, x) to
    EPSG:4326 (WGS-84) coordinate reference system.

    If *dst_resampling* is a string, it provides the default
    resampling for all variables.
    If *dst_resampling* is a dictionary, it provides a mapping
    from variable names to the desired
    resampling for that variable.

    The resampling may be one of the following up-sampling
    algorithms:

    * `Nearest`
    * `Bilinear`
    * `Cubic`
    * `CubicSpline`
    * `Lanczos`

    Or one of the down-sampling algorithms:

    * `Average`
    * `Min`
    * `Max`
    * `Median`
    * `Mode`
    * `Q1`
    * `Q3`

    Args:
        src_dataset: The source dataset
        src_xy_var_names: Names of the x,y coordinate variables
        src_xy_tp_var_names: Names of the x,y tie-point coordinate variables
        src_xy_crs:
        src_xy_gcp_step:
        src_xy_tp_gcp_step:
        dst_size:
        dst_region:
        dst_resampling: The spatial resampling algorithm.
            Either a string that provides the default resampling
            algorithm name or a dictionary that maps variable names to per-variable resampling algorithm names.
        include_non_spatial_vars:
        include_xy_vars: Whether to include the variables
            given by *src_xy_var_names*.
            Useful for projection-validation.
    Returns:
        the reprojected dataset
    """
    x_name, y_name = src_xy_var_names
    tp_x_name, tp_y_name = src_xy_tp_var_names or (None, None)

    # Set defaults
    src_xy_crs = src_xy_crs or CRS_WKT_EPSG_4326
    gcp_i_step, gcp_j_step = (
        (src_xy_gcp_step, src_xy_gcp_step)
        if isinstance(src_xy_gcp_step, int)
        else src_xy_gcp_step
    )
    tp_gcp_i_step, tp_gcp_j_step = (
        (src_xy_tp_gcp_step, src_xy_tp_gcp_step)
        if src_xy_tp_gcp_step is None or isinstance(src_xy_tp_gcp_step, int)
        else src_xy_tp_gcp_step
    )

    dst_width, dst_height = dst_size

    _assert(src_dataset is not None)
    _assert(dst_width > 1)
    _assert(dst_height > 1)
    _assert(gcp_i_step > 0)
    _assert(gcp_j_step > 0)

    _assert(x_name in src_dataset)
    _assert(y_name in src_dataset)
    x_var = src_dataset[x_name]
    y_var = src_dataset[y_name]
    if len(x_var.dims) == 1 and len(y_var.dims) == 1:
        y_var, x_var = xr.broadcast(y_var, x_var)
    _assert(len(x_var.dims) == 2)
    _assert(y_var.dims == x_var.dims)
    _assert(x_var.shape[-1] >= 2)
    _assert(x_var.shape[-2] >= 2)
    _assert(y_var.shape == x_var.shape)

    src_width = x_var.shape[-1]
    src_height = x_var.shape[-2]

    dst_region = _ensure_valid_region(dst_region, GLOBAL_GEO_EXTENT, x_var, y_var)
    dst_x1, dst_y1, dst_x2, dst_y2 = dst_region

    dst_res = max((dst_x2 - dst_x1) / dst_width, (dst_y2 - dst_y1) / dst_height)
    _assert(dst_res > 0)

    dst_geo_transform = (dst_x1, dst_res, 0.0, dst_y2, 0.0, -dst_res)

    # Extract GCPs from full-res lon/lat 2D variables
    gcps = _get_gcps(x_var, y_var, gcp_i_step, gcp_j_step)

    if (
        tp_x_name
        and tp_y_name
        and tp_x_name in src_dataset
        and tp_y_name in src_dataset
    ):
        # If there are tie-point variables in the src_dataset
        tp_x_var = src_dataset[tp_x_name]
        tp_y_var = src_dataset[tp_y_name]
        _assert(len(tp_x_var.shape) == 2)
        _assert(tp_x_var.shape == tp_y_var.shape)
        tp_width = tp_x_var.shape[-1]
        tp_height = tp_x_var.shape[-2]
        _assert(tp_gcp_i_step is not None and tp_gcp_i_step > 0)
        _assert(tp_gcp_j_step is not None and tp_gcp_j_step > 0)
        # Extract GCPs also from tie-point lon/lat 2D variables
        tp_gcps = _get_gcps(tp_x_var, tp_y_var, tp_gcp_i_step, tp_gcp_j_step)
    else:
        # No tie-point variables
        tp_x_var = None
        tp_width = None
        tp_height = None
        tp_gcps = None

    mem_driver = gdal.GetDriverByName("MEM")

    dst_x2 = dst_x1 + dst_res * dst_width
    dst_y1 = dst_y2 - dst_res * dst_height

    dst_dataset = _new_dst_dataset(
        dst_width, dst_height, dst_res, dst_x1, dst_y1, dst_x2, dst_y2
    )

    if dst_resampling is None:
        dst_resampling = {}
    if isinstance(dst_resampling, str):
        dst_resampling = {
            var_name: dst_resampling for var_name in src_dataset.variables
        }

    for var_name in src_dataset.variables:
        src_var = src_dataset[var_name]

        if src_var.dims == x_var.dims:
            is_tp_var = False
            if var_name == x_name or var_name == y_name:
                if not include_xy_vars:
                    # Don't store lat and lon 2D vars in destination
                    continue
                dst_var_name = "src_" + var_name
            else:
                dst_var_name = var_name
            # PERF: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            data_type = numpy_to_gdal_dtype(src_var.dtype)
            src_var_dataset = mem_driver.Create(
                f"src_{var_name}", src_width, src_height, band_count, data_type, []
            )
            src_var_dataset.SetGCPs(gcps, src_xy_crs)
        elif tp_x_var is not None and src_var.dims == tp_x_var.dims:
            is_tp_var = True
            if var_name == tp_x_name or var_name == tp_y_name:
                if not include_xy_vars:
                    # Don't store lat and lon 2D vars in destination
                    continue
                dst_var_name = "src_" + var_name
            else:
                dst_var_name = var_name
            # PERF: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            data_type = numpy_to_gdal_dtype(src_var.dtype)
            src_var_dataset = mem_driver.Create(
                f"src_{var_name}", tp_width, tp_height, band_count, data_type, []
            )
            src_var_dataset.SetGCPs(tp_gcps, src_xy_crs)
        elif include_non_spatial_vars:
            # Store any variable as-is, that does not have the lat/lon 2D dims, then continue
            dst_dataset[var_name] = src_var
            continue
        else:
            continue

        # We use GDT_Float64 to introduce NaN as no-data-value
        dst_data_type = gdal.GDT_Float64
        dst_var_dataset = mem_driver.Create(
            f"dst_{var_name}", dst_width, dst_height, band_count, dst_data_type, []
        )
        dst_var_dataset.SetProjection(CRS_WKT_EPSG_4326)
        dst_var_dataset.SetGeoTransform(dst_geo_transform)

        # TODO (forman): PERFORMANCE: stack multiple variables of same src_data_type
        #                to perform the reprojection only once per stack

        # TODO (forman): CODE-DUPLICATION: refactor out common code block in reproject_crs_to_wgs84()

        for band_index in range(1, band_count + 1):
            src_var_dataset.GetRasterBand(band_index).SetNoDataValue(float("nan"))
            src_var_dataset.GetRasterBand(band_index).WriteArray(src_var.values)
            dst_var_dataset.GetRasterBand(band_index).SetNoDataValue(float("nan"))

        resample_alg, resample_alg_name = _get_resample_alg(
            dst_resampling,
            var_name,
            default=DEFAULT_TP_RESAMPLING if is_tp_var else DEFAULT_RESAMPLING,
        )

        warp_mem_limit = 0
        error_threshold = 0
        # See http://www.gdal.org/structGDALWarpOptions.html
        options = ["INIT_DEST=NO_DATA"]
        gdal.ReprojectImage(
            src_var_dataset,
            dst_var_dataset,
            None,
            None,
            resample_alg,
            warp_mem_limit,
            error_threshold,
            None,  # callback,
            None,  # callback_data,
            options,
        )  # options

        dst_values = dst_var_dataset.GetRasterBand(1).ReadAsArray()
        # print(var_name, dst_values.shape, np.nanmin(dst_values), np.nanmax(dst_values))

        dst_dataset[dst_var_name] = _new_dst_variable(
            src_var, dst_values, resample_alg_name
        )

    return dst_dataset


def _new_dst_variable(src_var, dst_values, resample_alg_name):
    dst_var_attrs = dict(**src_var.attrs, spatial_resampling=resample_alg_name)
    dst_attrs = dict(**src_var.attrs, spatial_resampling=resample_alg_name)
    if "grid_mapping" in dst_attrs:
        del dst_attrs["grid_mapping"]

    dst_var = xr.DataArray(dst_values, dims=["lat", "lon"], attrs=dst_var_attrs)
    dst_var.encoding = src_var.encoding
    if (
        np.issubdtype(dst_var.dtype, np.floating)
        and np.issubdtype(src_var.encoding.get("dtype"), np.integer)
        and src_var.encoding.get("_FillValue") is None
    ):
        warnings.warn(
            f"variable {dst_var.name!r}: setting _FillValue=0 to replace any NaNs"
        )
        dst_var.encoding["_FillValue"] = 0
    return dst_var


def _new_dst_dataset(dst_width, dst_height, dst_res, dst_x1, dst_y1, dst_x2, dst_y2):
    return xr.Dataset(
        coords=dict(
            lon=xr.DataArray(
                np.linspace(dst_x1 + dst_res / 2, dst_x2 - dst_res / 2, dst_width),
                dims=[
                    "lon",
                ],
                attrs=dict(**_LON_ATTRS, bounds="lon_bnds"),
            ),
            lat=xr.DataArray(
                np.linspace(dst_y2 - dst_res / 2, dst_y1 + dst_res / 2, dst_height),
                dims=[
                    "lat",
                ],
                attrs=dict(**_LAT_ATTRS, bounds="lat_bnds"),
            ),
            lon_bnds=xr.DataArray(
                list(
                    zip(
                        np.linspace(dst_x1, dst_x2 - dst_res, dst_width),
                        np.linspace(dst_x1 + dst_res, dst_x2, dst_width),
                    )
                ),
                dims=["lon", "bnds"],
                attrs=_LON_ATTRS,
            ),
            lat_bnds=xr.DataArray(
                list(
                    zip(
                        np.linspace(dst_y2, dst_y1 + dst_res, dst_height),
                        np.linspace(dst_y2 - dst_res, dst_y1, dst_height),
                    )
                ),
                dims=["lat", "bnds"],
                attrs=_LAT_ATTRS,
            ),
        )
    )


_LON_ATTRS = dict(
    long_name="longitude", standard_name="longitude", units="degrees_east"
)
_LAT_ATTRS = dict(long_name="latitude", standard_name="latitude", units="degrees_north")

_NUMPY_TO_GDAL_DTYPE_MAPPING = {
    np.dtype(np.int8): gdal.GDT_Int16,
    np.dtype(np.int16): gdal.GDT_Int16,
    np.dtype(np.int32): gdal.GDT_Int32,
    np.dtype(np.uint8): gdal.GDT_Byte,
    np.dtype(np.uint16): gdal.GDT_UInt16,
    np.dtype(np.uint32): gdal.GDT_UInt32,
    np.dtype(np.float16): gdal.GDT_Float32,
    np.dtype(np.float32): gdal.GDT_Float32,
    np.dtype(np.float64): gdal.GDT_Float64,
}


def numpy_to_gdal_dtype(np_dtype):
    if np_dtype in _NUMPY_TO_GDAL_DTYPE_MAPPING:
        return _NUMPY_TO_GDAL_DTYPE_MAPPING[np_dtype]
    warnings.warn(f"unhandled numpy dtype {np_dtype}, using float64 instead")
    return gdal.GDT_Float64


def _get_resample_alg(dst_resampling, var_name, default):
    resample_alg_name = dst_resampling.get(var_name, default)
    if resample_alg_name not in NAME_TO_GDAL_RESAMPLE_ALG:
        raise ValueError(
            f"{resample_alg_name!r} is not a name of a known resampling algorithm"
        )
    resample_alg = NAME_TO_GDAL_RESAMPLE_ALG[resample_alg_name]
    return resample_alg, resample_alg_name


NAME_TO_GDAL_RESAMPLE_ALG: Dict[str, Any] = dict(
    # Up-sampling
    Nearest=gdal.GRA_NearestNeighbour,
    Bilinear=gdal.GRA_Bilinear,
    Cubic=gdal.GRA_Cubic,
    CubicSpline=gdal.GRA_CubicSpline,
    Lanczos=gdal.GRA_Lanczos,
    # Down-sampling
    Average=gdal.GRA_Average,
    Min=gdal.GRA_Min,
    Max=gdal.GRA_Max,
    Median=gdal.GRA_Med,
    Mode=gdal.GRA_Mode,
    Q1=gdal.GRA_Q1,
    Q3=gdal.GRA_Q3,
)


def _assert(cond, text="Assertion failed"):
    if not cond:
        raise ValueError(text)


def _ensure_valid_region(
    region: CoordRange,
    valid_region: CoordRange,
    x_var: xr.DataArray,
    y_var: xr.DataArray,
    extra: float = 0.01,
):
    if region:
        # Extract region
        _assert(len(region) == 4)
        x1, y1, x2, y2 = region
    else:
        # Determine region from full-res lon/lat 2D variables
        x1, x2 = x_var.min(), x_var.max()
        y1, y2 = y_var.min(), y_var.max()
        extra_space = extra * max(x2 - x1, y2 - y1)
        # Add extra space in units of the source coordinates
        x1, x2 = x1 - extra_space, x2 + extra_space
        y1, y2 = y1 - extra_space, y2 + extra_space
    x_min, y_min, x_max, y_max = valid_region
    x1, x2 = max(x_min, x1), min(x_max, x2)
    y1, y2 = max(y_min, y1), min(y_max, y2)
    assert x1 < x2 and y1 < y2
    return x1, y1, x2, y2


def _get_gcps(
    x_var: xr.DataArray, y_var: xr.DataArray, i_step: int, j_step: int
) -> List[gdal.GCP]:
    x_values = x_var.values
    y_values = y_var.values
    i_size = x_var.shape[-1]
    j_size = x_var.shape[-2]
    gcps = []
    gcp_id = 0
    i_count = (i_size + i_step - 1) // i_step
    j_count = (j_size + j_step - 1) // j_step
    for j in np.linspace(0, j_size - 1, j_count, dtype=np.int32):
        for i in np.linspace(0, i_size - 1, i_count, dtype=np.int32):
            x, y = float(x_values[j, i]), float(y_values[j, i])
            gcps.append(
                gdal.GCP(x, y, 0.0, i + 0.5, j + 0.5, "%s,%s" % (i, j), str(gcp_id))
            )
            gcp_id += 1
    return gcps


def get_projection_wkt(
    name: str,
    proj_name: str,
    latitude_of_origin: float = 0.0,
    central_meridian: float = 0.0,
    scale_factor: float = 1.0,
    false_easting: float = 0.0,
    false_northing: float = 0.0,
):
    return (
        f'PROJCS["{name}",'
        f'  GEOGCS["WGS 84",'
        f'    DATUM["WGS_1984",'
        f'      SPHEROID["WGS 84", 6378137, 298.257223563,'
        f'        AUTHORITY["EPSG", 7030]],'
        f"      TOWGS84[0,0,0,0,0,0,0],"
        f'      AUTHORITY["EPSG", 6326]],'
        f'    PRIMEM["Greenwich", 0, AUTHORITY["EPSG", 8901]],'
        f'    UNIT["DMSH",0.0174532925199433,AUTHORITY["EPSG", 9108]],'
        f'    AXIS["Lat", NORTH],'
        f'    AXIS["Long", EAST],'
        f'    AUTHORITY["EPSG", 4326]],'
        f'  PROJECTION["{proj_name}"],'
        f'  PARAMETER["latitude_of_origin", {latitude_of_origin}],'
        f'  PARAMETER["central_meridian", {central_meridian}],'
        f'  PARAMETER["scale_factor", {scale_factor}],'
        f'  PARAMETER["false_easting", {false_easting}],'
        f'  PARAMETER["false_northing", {false_northing}],'
        f'  UNIT["metre", 1, AUTHORITY["EPSG", "9001"]]'
        f"]"
    )
