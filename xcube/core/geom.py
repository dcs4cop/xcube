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

import math
import warnings
from typing import Optional, Union, Dict, Tuple, Sequence, Any, Mapping, List

import affine
import dask.array as da
import geopandas as gpd
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.geometry
import shapely.wkt
import xarray as xr
from deprecated import deprecated

from xcube.core.schema import get_dataset_bounds_var_name
from xcube.core.schema import get_dataset_chunks
from xcube.core.schema import get_dataset_xy_var_names
from xcube.core.update import update_dataset_spatial_attrs
from xcube.util.geojson import GeoJSON
from xcube.util.types import normalize_scalar_or_pair

GeometryLike = Union[
    shapely.geometry.base.BaseGeometry, Dict[str, Any], str, Sequence[Union[float, int]]
]
Bounds = Tuple[float, float, float, float]
SplitBounds = Tuple[Bounds, Optional[Bounds]]

Name = str
Attrs = Mapping[Name, Any]
GeoJSONFeature = Mapping[Name, Any]
GeoJSONFeatures = Sequence[GeoJSONFeature]
GeoDataFrame = "pandas.geodataframe.GeoDataFrame"
VarProps = Mapping[Name, Mapping[Name, Any]]

_INVALID_GEOMETRY_MSG = (
    "Geometry must be either a shapely geometry object, "
    "a GeoJSON-serializable dictionary, a geometry WKT string, "
    "box coordinates (x1, y1, x2, y2), "
    "or point coordinates (x, y)"
)

_INVALID_BOX_COORDS_MSG = "Invalid box coordinates"


def rasterize_features(
    dataset: xr.Dataset,
    features: Union[GeoDataFrame, GeoJSONFeatures],
    feature_props: Sequence[Name],
    var_props: Dict[Name, VarProps] = None,
    tile_size: Union[int, Tuple[int, int]] = None,
    all_touched: bool = False,
    in_place: bool = False,
) -> Optional[xr.Dataset]:
    """Rasterize feature properties given by *feature_props* of
    vector-data *features* as new variables into *dataset*.

    *dataset* must have two spatial 1-D coordinates, either
    `lon` and `lat` in degrees, reprojected coordinates,
    `x` and `y`, or similar.

    *feature_props* is a sequence of names of feature properties
    that must exists in each feature of *features*.

    *features* may be passed as pandas.GeoDataFrame` or as an
    iterable of GeoJSON features.

    Using the optional *var_props*, the properties of newly
    created variables from feature properties can be specified.
    It is a mapping of feature property names to mappings of
    variable properties.
    Here is an example variable properties mapping:

    ```python
    {
        'name': 'land_class',  # (str) - the variable's name,
                               # default is the feature property name;
        'dtype' np.int16,      # (str|np.dtype) - the variable's dtype,
                               # default is np.float64;
        'fill_value': -999,    # (bool|int|float|np.nparray) -
                               # the variable's fill value,
                               # default is np.nan;
        'attrs': {},           # (Mapping[str, Any]) -
                               # the variable's fill value, default is {};
        'converter': int,      # (Callable[[Any], Any]) -
                               # a converter function used to convert
                               # from property feature value to variable
                               # value, default is float.
                               # Deprecated, no longer used.
    }
    ```

    Note that newly created variables will have data type `np.float64`
    because `np.nan` is used to encode missing values. `fill_value` and
    `dtype` are used to encode the variables when persisting the data.

    Currently, the coordinates of the geometries in the given
    *features* must use the same CRS as the given *dataset*.

    Args:
        dataset: The xarray dataset.
            features: A `geopandas.GeoDataFrame` instance
            or a sequence of GeoJSON features.
        feature_props: Sequence of names of numeric feature
            properties to be rasterized.
        var_props: Optional mapping of feature property name
            to a name or a 5-tuple (name, dtype, fill_value,
            attributes, converter) for the new variable.
        tile_size: If given, the unconditional spatial chunk sizes
            in x- and y-direction in pixels.
            May be given as integer scalar or x,y-pair of integers.
        all_touched: If True, all pixels intersected by a
            feature's geometry outlines will be included.
            If False, only pixels whose center is
            within the feature polygon or that are selected by Bresenham’s line
            algorithm will be included in the mask.
            The default is False.
        in_place: Whether to add new variables to *dataset*.
            If False, a copy will be created and returned.
    Returns:
        dataset with rasterized feature_property
    """

    var_props = var_props or {}
    for v in var_props.values():
        if v and "converter" in v:
            warnings.warn(
                f'the "converter" property of var_props'
                f" has been deprecated and will be ignored",
                DeprecationWarning,
            )

    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    x_min, y_min, x_max, y_max = get_dataset_bounds(dataset)
    x_var_name, y_var_name = xy_var_names
    x_var, y_var = dataset[x_var_name], dataset[y_var_name]
    x_dim, y_dim = x_var.dims[0], y_var.dims[0]
    yx_coords = {y_var_name: y_var, x_var_name: x_var}
    yx_dims = y_dim, x_dim

    width = x_var.size
    height = y_var.size
    x_res = (x_max - x_min) / width
    y_res = (y_max - y_min) / height

    yx_chunks = _get_spatial_chunks(dataset, x_var_name, y_var_name, tile_size)

    if isinstance(features, gpd.GeoDataFrame):
        geo_data_frame = features
    else:
        geo_data_frame = gpd.GeoDataFrame.from_features(features)
    for feature_prop_name in feature_props:
        if feature_prop_name not in geo_data_frame:
            raise ValueError(f"feature property " f"{feature_prop_name!r} not found")

    # Filter out empty or invalid geometries, remember valid rows
    geometries = []
    valid_row_indices = []
    for row_index in range(len(geo_data_frame)):
        geometry = geo_data_frame.geometry[row_index]
        if not geometry.is_empty and geometry.is_valid:
            valid_row_indices.append(row_index)
            geometries.append(geometry.__geo_interface__)

    # Collect columns of feature data for valid row indices
    valid_row_indices = np.array(valid_row_indices)
    feature_data = []
    for feature_prop_name in feature_props:
        feature_values = geo_data_frame[feature_prop_name].to_numpy()
        feature_values = feature_values[valid_row_indices]
        var_prop_mapping = var_props.get(feature_prop_name, {})
        dtype = var_prop_mapping.get("dtype", feature_values.dtype)
        if dtype != feature_values.dtype:
            feature_values = feature_values.astype(dtype=dtype)
        feature_data.append(feature_values)

    # Rasterize all features into single blocks of type np.float64
    #
    # Note, we process all rows and all features in every block,
    # so we need to mask only once per block and because masking
    # is an expensive operation.
    #
    # When there are many feature variables, and they are accessed
    # independently of each other, this may quickly become
    # expensive in terms of CPU.
    # However, if we'd create dask arrays for every row, this will become
    # also expensive in terms of memory and CPU, because of the potentially
    # very large graphs.
    # Alternatively, we could create independent dask graphs for
    # every feature with all rows processed in each block.
    #
    num_features = len(feature_props)
    chunks = da.core.normalize_chunks(
        (num_features, *yx_chunks), (num_features, height, width)
    )
    rasterized_features = da.map_blocks(
        _rasterize_features_into_block,
        chunks=chunks,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
        geometries=geometries,
        feature_data=feature_data,
        x_offset=x_min,
        y_offset=y_max,
        x_res=x_res,
        y_res=y_res,
        all_touched=all_touched,
    )

    if y_var[0] < y_var[-1]:
        rasterized_features = rasterized_features[:, ::-1, ::]

    if not in_place:
        dataset = xr.Dataset(coords=dataset.coords, attrs=dataset.attrs)

    # Create feature variables from rasterized features
    for feature_index, feature_prop_name in enumerate(feature_props):
        var_prop_mapping = var_props.get(feature_prop_name, {})
        var_name = var_prop_mapping.get("name", feature_prop_name.replace(" ", "_"))
        var_dtype = np.dtype(var_prop_mapping.get("dtype", np.float64))
        var_fill_value = var_prop_mapping.get("fill_value", np.nan)
        var_attrs = var_prop_mapping.get("attrs", {})

        feature_image = rasterized_features[feature_index]

        feature_var = xr.DataArray(
            feature_image, coords=yx_coords, dims=yx_dims, attrs=var_attrs
        )
        feature_var.encoding.update(_FillValue=var_fill_value, dtype=var_dtype)
        dataset[var_name] = feature_var

    return dataset


def _rasterize_features_into_block(
    block_info: Dict[Union[str, None], Any] = None,
    geometries: List[Dict[str, Any]] = None,
    feature_data: List[np.ndarray] = None,
    x_offset: float = None,
    y_offset: float = None,
    x_res: float = None,
    y_res: float = None,
    all_touched: bool = None,
):
    ret_info = block_info[None]
    dtype = ret_info["dtype"]
    chunk_shape = ret_info["chunk-shape"]
    num_features, height, width = chunk_shape
    image_shape = height, width
    _, (y_start, y_end), (x_start, x_end) = ret_info["array-location"]
    x1 = x_offset + x_res * x_start
    x2 = x_offset + x_res * x_end
    y1 = y_offset - y_res * y_start
    y2 = y_offset - y_res * y_end
    transform = affine.Affine(x_res, 0.0, x1, 0.0, -y_res, y1)
    block_bounds = shapely.geometry.box(x1, min(y1, y2), x2, max(y1, y2))
    block = np.full(chunk_shape, np.nan, dtype=dtype)
    for row_index, geometry in enumerate(geometries):
        shape = shapely.geometry.shape(geometry)
        shape = block_bounds.intersection(shape)
        if shape.is_empty:
            continue
        if not shape.is_valid:
            continue
        mask = rasterio.features.geometry_mask(
            [shape],
            out_shape=image_shape,
            transform=transform,
            all_touched=all_touched,
            invert=True,
        )
        for i in range(num_features):
            background = block[i]
            foreground = np.full(image_shape, feature_data[i][row_index], dtype=dtype)
            block[i, :, :] = np.where(mask, foreground, background)

    return block


def mask_dataset_by_geometry(
    dataset: xr.Dataset,
    geometry: GeometryLike,
    tile_size: Union[int, Tuple[int, int]] = None,
    excluded_vars: Sequence[str] = None,
    no_clip: bool = False,
    all_touched: bool = False,
    save_geometry_mask: Union[str, bool] = False,
    save_geometry_wkt: Union[str, bool] = False,
) -> Optional[xr.Dataset]:
    """Mask a dataset according to the given geometry. The cells of
    variables of the returned dataset will have NaN-values where their
    spatial coordinates are not intersecting
    the given geometry.

    Args:
        dataset: The dataset
        geometry: A geometry-like object, see
            py:function:`convert_geometry`.
        tile_size: If given, the unconditional spatial chunk sizes in x-
            and y-direction in pixels. May be given as integer scalar or
            x,y-pair of integers.
        excluded_vars: Optional sequence of names of data variables that
            should not be masked (but still may be clipped).
        no_clip: If True, the function will not clip the dataset before
            masking, this is, the returned dataset will have the same
            dimension size as the given *dataset*.
        all_touched: If True, all pixels intersected by geometry
            outlines will be included in the mask. If False, only pixels
            whose center is within the polygon or that are selected by
            Bresenham’s line algorithm will be included in the mask.
            The default value is set to `False`.
        save_geometry_mask: If the value is a string, the effective
            geometry mask array is stored as a 2D data variable named by
            *save_geometry_mask*. If the value is True, the name
            "geometry_mask" is used.
        save_geometry_wkt: If the value is a string, the effective
            intersection geometry is stored as a Geometry WKT string in
            the global attribute named by *save_geometry*. If the value
            is True, the name "geometry_wkt" is used.

    Returns:
        The dataset spatial subset, or None if the bounding box of the
        dataset has a no or a zero area intersection with the bounding
        box of the geometry.
    """
    geometry = normalize_geometry(geometry)
    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    dataset_bounds = get_dataset_bounds(dataset, xy_var_names=xy_var_names)
    intersection_geometry = intersect_geometries(dataset_bounds, geometry)
    if intersection_geometry is None:
        return None

    if not no_clip:
        dataset = _clip_dataset_by_geometry(
            dataset, intersection_geometry, xy_var_names
        )

    x_min, y_min, x_max, y_max = get_dataset_bounds(dataset, xy_var_names=xy_var_names)

    x_var_name, y_var_name = xy_var_names
    x_var, y_var = dataset[x_var_name], dataset[y_var_name]

    width = x_var.size
    height = y_var.size
    x_res = (x_max - x_min) / width
    y_res = (y_max - y_min) / height

    yx_chunks = _get_spatial_chunks(dataset, x_var_name, y_var_name, tile_size)

    chunks = da.core.normalize_chunks(yx_chunks, shape=(height, width))

    mask_data = da.map_blocks(
        _mask_block,
        chunks=chunks,
        dtype=bool,
        meta=np.array((), dtype=bool),
        geometry=intersection_geometry,
        x_offset=x_min,
        y_offset=y_max,
        x_res=x_res,
        y_res=y_res,
        all_touched=all_touched,
    )

    if y_var[0] < y_var[-1]:
        mask_data = mask_data[::-1, ::]

    mask = xr.DataArray(
        mask_data,
        coords={y_var_name: y_var, x_var_name: x_var},
        dims=(y_var.dims[0], x_var.dims[0]),
    )

    dataset_vars = {}
    for var_name, var in dataset.data_vars.items():
        if not excluded_vars or var_name not in excluded_vars:
            dataset_vars[var_name] = var.where(mask)
        else:
            dataset_vars[var_name] = var

    masked_dataset = xr.Dataset(
        dataset_vars, coords=dataset.coords, attrs=dataset.attrs
    )

    _save_geometry_mask(masked_dataset, mask, save_geometry_mask)
    _save_geometry_wkt(masked_dataset, intersection_geometry, save_geometry_wkt)

    return masked_dataset


def _mask_block(
    block_info: Dict[Union[str, None], Any] = None,
    geometry: Dict[str, Any] = None,
    x_offset: float = None,
    y_offset: float = None,
    x_res: float = None,
    y_res: float = None,
    all_touched: bool = None,
):
    ret_info = block_info[None]
    height, width = ret_info["chunk-shape"]
    (y_start, _), (x_start, _) = ret_info["array-location"]
    x1 = x_offset + x_res * x_start
    y1 = y_offset - y_res * y_start
    transform = affine.Affine(x_res, 0.0, x1, 0.0, -y_res, y1)
    return rasterio.features.geometry_mask(
        [shapely.geometry.shape(geometry)],
        out_shape=(height, width),
        transform=transform,
        all_touched=all_touched,
        invert=True,
    )


def _get_spatial_chunks(
    dataset: xr.Dataset,
    x_var_name: str,
    y_var_name: str,
    tile_size: Union[None, int, Tuple[int, int]],
):
    width = dataset[x_var_name].size
    height = dataset[y_var_name].size
    if tile_size:
        tile_size = normalize_scalar_or_pair(tile_size, item_type=int, name="tile_size")
        yx_chunks = (min(height, tile_size[1]), min(width, tile_size[0]))
    else:
        dataset_chunks = get_dataset_chunks(dataset)
        yx_chunks = (dataset_chunks.get(y_var_name), dataset_chunks.get(x_var_name))
        if not all(yx_chunks):
            yx_chunks = (min(height, 1024), min(width, 1024))
    return yx_chunks


def clip_dataset_by_geometry(
    dataset: xr.Dataset,
    geometry: GeometryLike,
    save_geometry_wkt: Union[str, bool] = False,
) -> Optional[xr.Dataset]:
    """Spatially clip a dataset according to the bounding box of a
    given geometry.

    Args:
        dataset: The dataset
        geometry: A geometry-like object, see
            py:function:`convert_geometry`.
        save_geometry_wkt: If the value is a string, the effective
            intersection geometry is stored as a Geometry WKT string in
            the global attribute named by *save_geometry*. If the value
            is True, the name "geometry_wkt" is used.

    Returns:
        The dataset spatial subset, or None if the bounding box of the
        dataset has a no or a zero area intersection with the bounding
        box of the geometry.
    """
    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    intersection_geometry = intersect_geometries(
        get_dataset_bounds(dataset, xy_var_names=xy_var_names), geometry
    )
    if intersection_geometry is None:
        return None
    return _clip_dataset_by_geometry(
        dataset,
        intersection_geometry,
        xy_var_names,
        save_geometry_wkt=save_geometry_wkt,
    )


def _clip_dataset_by_geometry(
    dataset: xr.Dataset,
    intersection_geometry: shapely.geometry.base.BaseGeometry,
    xy_var_names: Tuple[str, str],
    save_geometry_wkt: bool = False,
) -> Optional[xr.Dataset]:
    # TODO (forman): the following code is wrong,
    #   if the dataset bounds cross the anti-meridian!

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = get_dataset_bounds(
        dataset, xy_var_names=xy_var_names
    )

    x_var_name, y_var_name = xy_var_names
    x_var = dataset[x_var_name]
    y_var = dataset[y_var_name]

    width = x_var.size
    height = y_var.size
    res_x = (ds_x_max - ds_x_min) / width
    res_y = (ds_y_max - ds_y_min) / height

    g_x_min, g_y_min, g_x_max, g_y_max = intersection_geometry.bounds
    x1 = _clamp(int(math.floor((g_x_min - ds_x_min) / res_x)), 0, width - 1)
    x2 = _clamp(int(math.ceil((g_x_max - ds_x_min) / res_x)), 0, width - 1)
    y1 = _clamp(int(math.floor((g_y_min - ds_y_min) / res_y)), 0, height - 1)
    y2 = _clamp(int(math.ceil((g_y_max - ds_y_min) / res_y)), 0, height - 1)
    if y_var[0] > y_var[-1]:  # inverse ?
        _y1, _y2 = y1, y2
        y1 = height - _y2 - 1
        y2 = height - _y1 - 1

    dataset_subset = dataset.isel(
        **{x_var_name: slice(x1, x2), y_var_name: slice(y1, y2)}
    )

    update_dataset_spatial_attrs(dataset_subset, update_existing=True, in_place=True)

    _save_geometry_wkt(dataset_subset, intersection_geometry, save_geometry_wkt)

    return dataset_subset


def _save_geometry_mask(dataset, mask, save_mask):
    if save_mask:
        var_name = save_mask if isinstance(save_mask, str) else "geometry_mask"
        dataset[var_name] = mask


def _save_geometry_wkt(dataset, intersection_geometry, save_geometry):
    if save_geometry:
        attr_name = save_geometry if isinstance(save_geometry, str) else "geometry_wkt"
        dataset.attrs.update({attr_name: intersection_geometry.wkt})


def intersect_geometries(
    geometry1: GeometryLike, geometry2: GeometryLike
) -> Optional[shapely.geometry.base.BaseGeometry]:
    geometry1 = normalize_geometry(geometry1)
    if geometry1 is None:
        return None
    geometry2 = normalize_geometry(geometry2)
    if geometry2 is None:
        return geometry1
    intersection_geometry = geometry1.intersection(geometry2)
    if not intersection_geometry.is_valid or intersection_geometry.is_empty:
        return None
    return intersection_geometry


def normalize_geometry(
    geometry: Optional[GeometryLike],
) -> Optional[shapely.geometry.base.BaseGeometry]:
    """Convert a geometry-like object into a shapely geometry
    object (`shapely.geometry.BaseGeometry`).

    A geometry-like object may be any shapely geometry object,
    * a dictionary that can be serialized to valid GeoJSON,
    * a WKT string,
    * a box given by a string of the form "<x1>,<y1>,<x2>,<y2>"
      or by a sequence of four numbers x1, y1, x2, y2,
    * a point by a string of the form "<x>,<y>"
      or by a sequence of two numbers x, y.

    Handling of geometries crossing the anti-meridian:

    * If box coordinates are given, it is allowed to pass
      x1, x2 where x1 > x2, which is interpreted as a box crossing
      the anti-meridian. In this case the function splits the box
      along the anti-meridian and returns a multi-polygon.
    * In all other cases, 2D geometries are assumed to _not cross
      the anti-meridian at all_.

    Args:
        geometry: A geometry-like object

    Returns:
        Shapely geometry object or None.
    """

    if isinstance(geometry, shapely.geometry.base.BaseGeometry):
        return geometry

    if isinstance(geometry, dict):
        if GeoJSON.is_geometry(geometry):
            return shapely.geometry.shape(geometry)
        elif GeoJSON.is_feature(geometry):
            geometry = GeoJSON.get_feature_geometry(geometry)
            if geometry is not None:
                return shapely.geometry.shape(geometry)
        elif GeoJSON.is_feature_collection(geometry):
            features = GeoJSON.get_feature_collection_features(geometry)
            if features is not None:
                geometries = [
                    f2
                    for f2 in [GeoJSON.get_feature_geometry(f1) for f1 in features]
                    if f2 is not None
                ]
                if geometries:
                    geometry = dict(type="GeometryCollection", geometries=geometries)
                    return shapely.geometry.shape(geometry)
        raise ValueError(_INVALID_GEOMETRY_MSG)

    if isinstance(geometry, str):
        return shapely.wkt.loads(geometry)

    if geometry is None:
        return None

    invalid_box_coords = False
    # noinspection PyBroadException
    try:
        x1, y1, x2, y2 = geometry
        is_point = x1 == x2 and y1 == y2
        if is_point:
            return shapely.geometry.Point(x1, y1)
        invalid_box_coords = x1 == x2 or y1 >= y2
        if not invalid_box_coords:
            return get_box_split_bounds_geometry(x1, y1, x2, y2)
    except Exception:
        # noinspection PyBroadException
        try:
            x, y = geometry
            return shapely.geometry.Point(x, y)
        except Exception:
            pass

    if invalid_box_coords:
        raise ValueError(_INVALID_BOX_COORDS_MSG)
    raise ValueError(_INVALID_GEOMETRY_MSG)


@deprecated(
    version="0.11.2",
    reason="convert_geometry() has been" " renamed to normalize_geometry()",
)
def convert_geometry(
    geometry: Optional[GeometryLike],
) -> Optional[shapely.geometry.base.BaseGeometry]:
    return normalize_geometry(geometry)


convert_geometry.__doc__ = normalize_geometry.__doc__


@deprecated(
    version="0.11.2", reason='Uses wrong definition of "inverted". No longer used.'
)
def is_dataset_y_axis_inverted(
    dataset: Union[xr.Dataset, xr.DataArray], xy_var_names: Tuple[str, str] = None
) -> bool:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    y_var = dataset[xy_var_names[1]]
    return float(y_var[0]) < float(y_var[-1])


def is_lon_lat_dataset(
    dataset: Union[xr.Dataset, xr.DataArray], xy_var_names: Tuple[str, str] = None
) -> bool:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    x_var_name, y_var_name = xy_var_names
    if x_var_name == "lon" and y_var_name == "lat":
        return True
    x_var = dataset[x_var_name]
    y_var = dataset[y_var_name]
    return (
        x_var.attrs.get("long_name") == "longitude"
        and y_var.attrs.get("long_name") == "latitude"
    )


def get_dataset_geometry(
    dataset: Union[xr.Dataset, xr.DataArray], xy_var_names: Tuple[str, str] = None
) -> shapely.geometry.base.BaseGeometry:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    geo_bounds = get_dataset_bounds(dataset, xy_var_names=xy_var_names)
    if is_lon_lat_dataset(dataset, xy_var_names=xy_var_names):
        return get_box_split_bounds_geometry(*geo_bounds)
    else:
        return shapely.geometry.box(*geo_bounds)


def get_dataset_bounds(
    dataset: Union[xr.Dataset, xr.DataArray],
    xy_var_names: Optional[Tuple[str, str]] = None,
) -> Bounds:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    x_name, y_name = xy_var_names
    x_var, y_var = dataset.coords[x_name], dataset.coords[y_name]
    is_lon = xy_var_names[0] == "lon"

    # Note, x_min > x_max then we intersect with the anti-meridian
    x_bnds_name = get_dataset_bounds_var_name(dataset, x_name)
    if x_bnds_name is not None:
        x_bnds_var = dataset[x_bnds_name]
        x1 = x_bnds_var[0, 0]
        x2 = x_bnds_var[0, 1]
        x3 = x_bnds_var[-1, 0]
        x4 = x_bnds_var[-1, 1]
        x_min = min(x1, x2)
        x_max = max(x3, x4)
    else:
        x_min = x_var[0]
        x_max = x_var[-1]
        delta = (x_max - x_min + (0 if (x_max >= x_min or not is_lon) else 360)) / (
            x_var.size - 1
        )
        x_min -= 0.5 * delta
        x_max += 0.5 * delta

    # Note, x-axis may be inverted
    y_bnds_name = get_dataset_bounds_var_name(dataset, y_name)
    if y_bnds_name is not None:
        y_bnds_var = dataset[y_bnds_name]
        y1 = y_bnds_var[0, 0]
        y2 = y_bnds_var[0, 1]
        y3 = y_bnds_var[-1, 0]
        y4 = y_bnds_var[-1, 1]
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
    else:
        y1 = y_var[0]
        y2 = y_var[-1]
        delta = abs(y2 - y1) / (y_var.size - 1)
        y_min = min(y1, y2) - 0.5 * delta
        y_max = max(y1, y2) + 0.5 * delta

    return float(x_min), float(y_min), float(x_max), float(y_max)


def get_box_split_bounds(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> SplitBounds:
    if lon_max >= lon_min:
        return ((lon_min, lat_min, lon_max, lat_max), None)
    else:
        return ((lon_min, lat_min, 180.0, lat_max), (-180.0, lat_min, lon_max, lat_max))


def get_box_split_bounds_geometry(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> shapely.geometry.base.BaseGeometry:
    box_1, box_2 = get_box_split_bounds(lon_min, lat_min, lon_max, lat_max)
    if box_2 is not None:
        return shapely.geometry.MultiPolygon(
            polygons=[shapely.geometry.box(*box_1), shapely.geometry.box(*box_2)]
        )
    else:
        return shapely.geometry.box(*box_1)


def _clamp(x, x1, x2):
    if x < x1:
        return x1
    if x > x2:
        return x2
    return x
