# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import io
import logging
import math
from typing import Optional, Tuple, Dict, Any, Hashable, Union, Sequence, List

import PIL
import matplotlib.colors
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from xcube.constants import LOG
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.cmaps import ColormapProvider
from xcube.util.cmaps import DEFAULT_CMAP_NAME
from xcube.util.perf import measure_time_cm
from xcube.util.projcache import ProjCache
from xcube.util.timeindex import ensure_time_label_compatible
from xcube.util.types import Pair
from xcube.util.types import ScalarOrPair
from xcube.util.types import normalize_scalar_or_pair
from .mldataset import MultiLevelDataset
from .tilingscheme import DEFAULT_CRS_NAME
from .tilingscheme import DEFAULT_TILE_SIZE
from .tilingscheme import TilingScheme

DEFAULT_VALUE_RANGE = (0.0, 1.0)
DEFAULT_CMAP_NORM = "lin"
DEFAULT_FORMAT = "png"
DEFAULT_TILE_ENLARGEMENT = 1

ValueRange = Tuple[float, float]


def compute_tiles(
    ml_dataset: MultiLevelDataset,
    variable_names: Union[str, Sequence[str]],
    tile_bbox: Tuple[float, float, float, float],
    tile_crs: Union[str, pyproj.CRS] = DEFAULT_CRS_NAME,
    tile_size: ScalarOrPair[int] = DEFAULT_TILE_SIZE,
    level: int = 0,
    non_spatial_labels: Optional[Dict[str, Any]] = None,
    as_dataset: bool = False,
    tile_enlargement: int = DEFAULT_TILE_ENLARGEMENT,
    trace_perf: bool = False,
) -> Optional[Union[List[np.ndarray], xr.Dataset]]:
    """Compute tiles for given *variable_names* in
    given multi-resolution dataset *mr_dataset*.

    The algorithm is as follows:

    1. compute the z-index of the dataset pyramid from requested
       map CRS and tile_z.
    2. select a spatial 2D slice from the dataset pyramid.
    3. create a 2D array of the tile's map coordinates at each pixel
    4. transform 2D array of tile map coordinates into coordinates of
       the dataset CRS.
    5. use the 2D array's outline to retrieve the bbox in dataset
       coordinates.
    6. spatially subset the 2D slice using that bbox.
    7. compute a new bbox from the actual spatial subset.
    8. use the new bbox to transform the 2D array of the tile's dataset
       coordinates into indexes into the spatial subset.
    9. use the tile indices as 2D index into the spatial subset.
       The result is a reprojected array with the shape of the tile.
    10. turn that array into an RGBA image.
    11. encode RGBA image into PNG bytes.

    Args:
        ml_dataset: Multi-level dataset
        variable_names: Single variable name
            or a sequence of three names.
        tile_bbox: Tile bounding box
        tile_crs: Spatial tile coordinate reference system.
            Must be a geographical CRS, such as "EPSG:4326", or
            web mercator, i.e. "EPSG:3857". Defaults to "CRS84".
        tile_size: The tile size in pixels.
            Can be a scalar or an integer width/height pair.
            Defaults to 256.
        level: Dataset level to be used.
            Defaults to 0, the base level.
        non_spatial_labels: Labels for the non-spatial dimensions
            in the given variables.
        as_dataset: If set, an ``xr.Dataset`` is returned
            instead of a list of numpy arrays.
        tile_enlargement: Enlargement in pixels applied to
            the computed source tile read from the data.
            Can be used to increase the accuracy of the borders of target
            tiles at high zoom levels. Defaults to 1.
        trace_perf: If set, detailed performance
            metrics are logged using the level of the "xcube" logger.
    Returns:
        A list of numpy.ndarray instances according to variables
        given by *variable_names*. Returns None, if the resulting
        spatial subset would be too small.
    Raises: TileNotFoundException
    """
    if isinstance(variable_names, str):
        variable_names = (variable_names,)

    tile_size = normalize_scalar_or_pair(tile_size)
    tile_width, tile_height = tile_size

    logger = LOG if trace_perf else None

    measure_time = measure_time_cm(disabled=not trace_perf, logger=LOG)

    tile_x_min, tile_y_min, tile_x_max, tile_y_max = tile_bbox

    dataset = ml_dataset.get_dataset(level)

    with measure_time("Preparing 2D subset"):
        variables = [
            _get_variable(
                ml_dataset.ds_id, dataset, variable_name, non_spatial_labels, logger
            )
            for variable_name in variable_names
        ]

    variable_0 = variables[0]

    with measure_time("Transforming tile map to dataset coordinates"):
        ds_x_name, ds_y_name = ml_dataset.grid_mapping.xy_dim_names

        ds_y_coords = variable_0[ds_y_name]
        ds_y_points_up = bool(ds_y_coords[0] < ds_y_coords[-1])

        tile_res_x = (tile_x_max - tile_x_min) / (tile_width - 1)
        tile_res_y = (tile_y_max - tile_y_min) / (tile_height - 1)

        tile_x_1d = np.linspace(
            tile_x_min + 0.5 * tile_res_x, tile_x_max - 0.5 * tile_res_x, tile_width
        )
        tile_y_1d = np.linspace(
            tile_y_min + 0.5 * tile_res_y, tile_y_max - 0.5 * tile_res_y, tile_height
        )

        tile_x_2d = np.tile(tile_x_1d, (tile_height, 1))
        tile_y_2d = np.tile(tile_y_1d, (tile_width, 1)).transpose()

        assert tile_x_2d.shape == (tile_height, tile_width)
        assert tile_y_2d.shape == tile_x_2d.shape

        t_map_to_ds = ProjCache.INSTANCE.get_transformer(
            tile_crs, ml_dataset.grid_mapping.crs
        )

        tile_ds_x_2d, tile_ds_y_2d = t_map_to_ds.transform(tile_x_2d, tile_y_2d)

    with measure_time("Getting spatial subset"):
        # Get min/max of the 1D arrays surrounding the 2D array
        # North
        ds_x_n = tile_ds_x_2d[0, :]
        ds_y_n = tile_ds_y_2d[0, :]
        # South
        ds_x_s = tile_ds_x_2d[tile_height - 1, :]
        ds_y_s = tile_ds_y_2d[tile_height - 1, :]
        # West
        ds_x_w = tile_ds_x_2d[:, 0]
        ds_y_w = tile_ds_y_2d[:, 0]
        # East
        ds_x_e = tile_ds_x_2d[:, tile_width - 1]
        ds_y_e = tile_ds_y_2d[:, tile_width - 1]
        # Min
        ds_x_min = np.nanmin(
            [np.nanmin(ds_x_n), np.nanmin(ds_x_s), np.nanmin(ds_x_w), np.nanmin(ds_x_e)]
        )
        ds_y_min = np.nanmin(
            [np.nanmin(ds_y_n), np.nanmin(ds_y_s), np.nanmin(ds_y_w), np.nanmin(ds_y_e)]
        )
        # Max
        ds_x_max = np.nanmax(
            [np.nanmax(ds_x_n), np.nanmax(ds_x_s), np.nanmax(ds_x_w), np.nanmax(ds_x_e)]
        )
        ds_y_max = np.nanmax(
            [np.nanmax(ds_y_n), np.nanmax(ds_y_s), np.nanmax(ds_y_w), np.nanmax(ds_y_e)]
        )
        if (
            np.isnan(ds_x_min)
            or np.isnan(ds_y_min)
            or np.isnan(ds_y_max)
            or np.isnan(ds_y_max)
        ):
            raise TileNotFoundException(
                "Tile bounds NaN after map projection", logger=logger
            )

        num_extra_pixels = tile_enlargement
        res_x = (ds_x_max - ds_x_min) / tile_width
        res_y = (ds_y_max - ds_y_min) / tile_height
        extra_dx = num_extra_pixels * res_x
        extra_dy = num_extra_pixels * res_y
        ds_x_slice = slice(ds_x_min - extra_dx, ds_x_max + extra_dx)
        if ds_y_points_up:
            ds_y_slice = slice(ds_y_min - extra_dy, ds_y_max + extra_dy)
        else:
            ds_y_slice = slice(ds_y_max + extra_dy, ds_y_min - extra_dy)

        var_subsets = [
            variable.sel({ds_x_name: ds_x_slice, ds_y_name: ds_y_slice})
            for variable in variables
        ]
        for var_subset in var_subsets:
            # A zero or a one in the tile's shape will produce a
            # non-existing or too small tile. It will also prevent
            # determining the current resolution.
            if 0 in var_subset.shape or 1 in var_subset.shape:
                return None

    with measure_time("Transforming dataset coordinates into indices"):
        var_subset_0 = var_subsets[0]
        ds_x_coords = var_subset_0[ds_x_name]
        ds_y_coords = var_subset_0[ds_y_name]

        ds_x1 = float(ds_x_coords[0])
        ds_x2 = float(ds_x_coords[-1])
        ds_y1 = float(ds_y_coords[0])
        ds_y2 = float(ds_y_coords[-1])

        ds_size_x = ds_x_coords.size
        ds_size_y = ds_y_coords.size

        ds_dx = (ds_x2 - ds_x1) / (ds_size_x - 1)
        ds_dy = (ds_y2 - ds_y1) / (ds_size_y - 1)

        ds_x_indices = (tile_ds_x_2d - ds_x1) / ds_dx
        ds_y_indices = (tile_ds_y_2d - ds_y1) / ds_dy

        ds_x_indices = ds_x_indices.astype(dtype=np.int64)
        ds_y_indices = ds_y_indices.astype(dtype=np.int64)

    with measure_time("Masking dataset indices"):
        ds_mask = (
            (ds_x_indices >= 0)
            & (ds_x_indices < ds_size_x)
            & (ds_y_indices >= 0)
            & (ds_y_indices < ds_size_y)
        )

        ds_x_indices = np.where(ds_mask, ds_x_indices, 0)
        ds_y_indices = np.where(ds_mask, ds_y_indices, 0)

    var_tiles = []
    for var_subset in var_subsets:
        with measure_time("Loading 2D data for spatial subset"):
            # Note, we need to load the values here into a numpy array,
            # because 2D indexing by [ds_y_indices, ds_x_indices]
            # does not (yet) work with dask arrays.
            var_tile = var_subset.values
            # Remove any axes above the 2nd. This is safe,
            # they will be of size one, if any.
            var_tile = var_tile.reshape(var_tile.shape[-2:])

        with measure_time("Looking up dataset indices"):
            # This does the actual projection trick.
            # Lookup indices ds_y_indices, ds_x_indices to create
            # the actual tile.
            var_tile = var_tile[ds_y_indices, ds_x_indices]
            var_tile = np.where(ds_mask, var_tile, np.nan)

        var_tiles.append(var_tile)

    if as_dataset:
        return _new_tile_dataset(
            [(var, dataset[var.name].dims) for var in variables],
            var_tiles,
            (ds_x_name, ds_y_name),
            (tile_x_1d, tile_y_1d),
            tile_crs,
        )

    return var_tiles


def _new_tile_dataset(
    original_vars: List[Tuple[xr.DataArray, Tuple[Hashable, ...]]],
    tiles: List[np.ndarray],
    xy_names: Tuple[str, str],
    xy_coords: Tuple[np.ndarray, np.ndarray],
    crs: Union[str, pyproj.CRS],
):
    data_vars = {}
    non_spatial_coords = {}
    for i, (original_var, original_dims) in enumerate(original_vars):
        var_name = original_var.name
        non_spatial_dims = []
        for dim in original_dims:
            if dim not in xy_names:
                non_spatial_dims.append(dim)
                if dim not in non_spatial_coords and dim in original_var.coords:
                    non_spatial_coords[dim] = original_var.coords[dim]
        data_2d = tiles[i]
        data_nd = data_2d[(*(len(non_spatial_dims) * [np.newaxis]), ...)]
        data_vars[var_name] = xr.DataArray(
            data=data_nd,
            dims=(*non_spatial_dims, "y", "x"),
            name=var_name,
            attrs=dict(**original_var.attrs, grid_mapping="crs"),
        )
    return xr.Dataset(
        data_vars=dict(
            **data_vars, crs=xr.DataArray((), attrs=pyproj.CRS(crs).to_cf())
        ),
        coords=dict(
            **{
                k: xr.DataArray([v.values], dims=k, attrs=v.attrs)
                for k, v in non_spatial_coords.items()
            },
            y=xr.DataArray(
                xy_coords[1],
                dims="y",
                attrs=dict(
                    long_name="y coordinate of projection",
                    standard_name="projection_y_coordinate",
                ),
            ),
            x=xr.DataArray(
                xy_coords[0],
                dims="x",
                attrs=dict(
                    long_name="x coordinate of projection",
                    standard_name="projection_x_coordinate",
                ),
            ),
        ),
    )


def compute_rgba_tile(
    ml_dataset: MultiLevelDataset,
    variable_names: Union[str, Sequence[str]],
    tile_x: int,
    tile_y: int,
    tile_z: int,
    cmap_provider: ColormapProvider,
    crs_name: str = DEFAULT_CRS_NAME,
    tile_size: ScalarOrPair[int] = DEFAULT_TILE_SIZE,
    cmap_name: Optional[str] = DEFAULT_CMAP_NAME,
    cmap_norm: Optional[str] = DEFAULT_CMAP_NORM,
    value_ranges: Optional[Union[ValueRange, Sequence[ValueRange]]] = None,
    non_spatial_labels: Optional[Dict[str, Any]] = None,
    format: str = DEFAULT_FORMAT,
    tile_enlargement: int = DEFAULT_TILE_ENLARGEMENT,
    trace_perf: bool = False,
) -> Union[bytes, np.ndarray]:
    """Compute an RGBA image tile from *variable_names* in
    given multi-resolution dataset *mr_dataset*.

    If length of *variable_names* is three, we create a direct component
    RGBA image where the three variables correspond to the three R, G, B
    channels and A is computed from NaN values in the three variables.
    *value_ranges* must have exactly three elements too.

    If length of *variable_names* is one, we create a color mapped
    RGBA image using *cmap_name* where R, G, B is computed from a color bar
    and A is computed from NaN values in the variable.
    *value_ranges* must have exactly one element.

    The algorithm is as follows:

    1. compute the z-index of the dataset pyramid from requested
       map CRS and tile_z.
    2. select a spatial 2D slice from the dataset pyramid.
    3. create a 2D array of the tile's map coordinates at each pixel
    4. transform 2D array of tile map coordinates into coordinates of
       the dataset CRS.
    5. use the 2D array's outline to retrieve the bbox in dataset
       coordinates.
    6. spatially subset the 2D slice using that bbox.
    7. compute a new bbox from the actual spatial subset.
    8. use the new bbox to transform the 2D array of the tile's dataset
       coordinates into indexes into the spatial subset.
    9. use the tile indices as 2D index into the spatial subset.
       The result is a reprojected array with the shape of the tile.
    10. turn that array into an RGBA image.
    11. encode RGBA image into PNG bytes.

    Args:
        ml_dataset: Multi-level dataset
        variable_names: Single variable name or a sequence of three
            names.
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        tile_z: Tile Z coordinate
        cmap_provider: Provider for colormaps.
        crs_name: Spatial tile coordinate reference system. Must be a
            geographical CRS, such as "EPSG:4326", or web mercator, i.e.
            "EPSG:3857". Defaults to "CRS84".
        tile_size: The tile size in pixels. Can be a scalar or an
            integer width/height pair. Defaults to 256.
        cmap_name: Color map name.
            Only used if a single variable name is given.
            Defaults to "viridis".
        cmap_norm: Color map normalisation. One of "lin" (linear), "log"
            (logarithmic), "cat" (categorical).
            Only used if a single variable name is given.
            Defaults to "lin".
        value_ranges: A single value range, or value ranges for each
            variable name.
        non_spatial_labels: Labels for the non-spatial dimensions in the
            given variables.
        tile_enlargement: Enlargement in pixels applied to the computed
            source tile read from the data. Can be used to increase the
            accuracy of the borders of target tiles at high zoom levels.
            Defaults to 1.
        format: Either 'png', 'image/png' or 'numpy'.
        trace_perf: If set, detailed performance metrics are logged
            using the level of the "xcube" logger.

    Returns:
        PNG bytes or unit8 numpy array, depending on *format*

    Raises:
        TileNotFoundException
        TileRequestException
    """
    if isinstance(variable_names, str):
        variable_names = (variable_names,)
    num_components = len(variable_names)
    assert_true(
        num_components in (1, 3),
        message="number of names in" " variable_names must be 1 or 3",
    )
    tile_size = normalize_scalar_or_pair(tile_size)
    tile_width, tile_height = tile_size
    if not value_ranges:
        value_ranges = num_components * (DEFAULT_VALUE_RANGE,)
    else:
        assert_instance(value_ranges, (list, tuple), name="value_ranges")
    if isinstance(value_ranges[0], (int, float)):
        value_ranges = num_components * (value_ranges,)
    assert_true(
        num_components == len(value_ranges),
        message="value_ranges must have" " same length as variable_names",
    )
    format = _normalize_format(format)
    assert_in(format, ("png", "numpy"), name="format")

    measure_time = measure_time_cm(disabled=not trace_perf, logger=LOG)

    tiling_scheme = TilingScheme.for_crs(crs_name).derive(tile_size=tile_size)

    tile_bbox = tiling_scheme.get_tile_extent(tile_x, tile_y, tile_z)
    if tile_bbox is None:
        # Whether to raise or not
        # could be made a configuration option.
        #
        # raise TileRequestException.new(
        #     'Tile indices out of tile grid bounds',
        #     logger=logger
        # )
        return TransparentRgbaTilePool.INSTANCE.get(tile_size, format)

    ds_level = tiling_scheme.get_resolutions_level(
        tile_z, ml_dataset.avg_resolutions, ml_dataset.grid_mapping.spatial_unit_name
    )

    var_tiles = compute_tiles(
        ml_dataset,
        variable_names,
        tile_bbox,
        tiling_scheme.crs,
        tile_size=tile_size,
        level=ds_level,
        non_spatial_labels=non_spatial_labels,
        tile_enlargement=tile_enlargement,
        trace_perf=trace_perf,
    )

    if var_tiles is None:
        return TransparentRgbaTilePool.INSTANCE.get(tile_size, format)

    cmap_norm = cmap_norm or DEFAULT_CMAP_NORM

    if len(var_tiles) == 1:
        with measure_time("Decoding color mapping"):
            # Note, we measure here because cmap_name may be an
            # JSON-encoded user-defined color map.
            cmap, colormap = cmap_provider.get_cmap(cmap_name or DEFAULT_CMAP_NAME)

        with measure_time("Encoding tile as RGBA image"):
            var_tile, value_range = var_tiles[0], value_ranges[0]
            if cmap_norm == "cat" and colormap.bounds:
                norm = matplotlib.colors.BoundaryNorm(
                    colormap.bounds, ncolors=cmap.N, clip=False
                )
            else:
                norm = get_continuous_norm(value_range, cmap_norm)
            norm_var_tile = norm(var_tile[::-1, :])
            var_tile_rgba = cmap(norm_var_tile)
            var_tile_rgba = (255 * var_tile_rgba).astype(np.uint8)
    else:
        with measure_time("Encoding 3 tiles as RGBA image"):
            norm_var_tiles = []
            for var_tile, value_range in zip(var_tiles, value_ranges):
                norm = get_continuous_norm(value_range, cmap_norm)
                norm_var_tiles.append(norm(var_tile[::-1, :]))

            r, g, b = norm_var_tiles
            var_tile_rgba = np.zeros((tile_height, tile_width, 4), dtype=np.uint8)
            var_tile_rgba[..., 0] = 255 * r
            var_tile_rgba[..., 1] = 255 * g
            var_tile_rgba[..., 2] = 255 * b
            var_tile_rgba[..., 3] = np.where(np.isfinite(r + g + b), 255, 0)

    if format == "png":
        with measure_time("Encoding RGBA image as PNG bytes"):
            return _encode_rgba_as_png(var_tile_rgba)
    else:  # format == 'numpy'
        return var_tile_rgba


def get_continuous_norm(
    value_range: Tuple[float, float], cmap_norm: Optional[str]
) -> matplotlib.colors.Normalize:
    value_min, value_max = value_range
    if value_max < value_min:
        value_min, value_max = value_max, value_min
    if math.isclose(value_min, value_max):
        value_max = value_min + 1
    if cmap_norm == "log":
        return matplotlib.colors.LogNorm(value_min, value_max, clip=True)
    else:
        return matplotlib.colors.Normalize(value_min, value_max, clip=True)


def get_var_cmap_params(
    var: xr.DataArray,
    cmap_name: Optional[str],
    cmap_norm: Optional[str],
    cmap_range: Tuple[Optional[float], Optional[float]],
    valid_range: Optional[Tuple[float, float]],
) -> Tuple[str, str, Tuple[float, float]]:
    if cmap_name is None:
        cmap_name = var.attrs.get("color_bar_name")
        if cmap_name is None:
            cmap_name = DEFAULT_CMAP_NAME
    if cmap_norm is None:
        cmap_norm = var.attrs.get("color_norm")
        if cmap_norm is None:
            cmap_norm = DEFAULT_CMAP_NORM
    cmap_vmin, cmap_vmax = cmap_range
    if cmap_vmin is None:
        cmap_vmin = var.attrs.get("color_value_min")
        if cmap_vmin is None and valid_range is not None:
            cmap_vmin = valid_range[0]
        if cmap_vmin is None:
            cmap_vmin = DEFAULT_VALUE_RANGE[0]
    if cmap_vmax is None:
        cmap_vmax = var.attrs.get("color_value_max")
        if cmap_vmax is None and valid_range is not None:
            cmap_vmax = valid_range[1]
        if cmap_vmax is None:
            cmap_vmax = DEFAULT_VALUE_RANGE[1]
    return cmap_name, cmap_norm, (cmap_vmin, cmap_vmax)


def get_var_valid_range(var: xr.DataArray) -> Optional[Tuple[float, float]]:
    valid_min = None
    valid_max = None
    valid_range = var.attrs.get("valid_range")
    if valid_range:
        try:
            valid_min, valid_max = map(float, valid_range)
        except (TypeError, ValueError):
            pass
    if valid_min is None:
        valid_min = var.attrs.get("valid_min")
    if valid_max is None:
        valid_max = var.attrs.get("valid_max")
    if valid_min is None and valid_max is None:
        valid_range = None
    elif valid_min is not None and valid_max is not None:
        valid_range = valid_min, valid_max
    elif valid_min is None:
        valid_range = -np.inf, valid_max
    else:
        valid_range = valid_min, +np.inf
    return valid_range


def _get_variable(ds_name, dataset, variable_name, non_spatial_labels, logger):
    if variable_name not in dataset:
        raise TileNotFoundException(
            f"Variable {variable_name!r}" f" not found in dataset {ds_name!r}",
            logger=logger,
        )
    variable = dataset[variable_name]
    non_spatial_labels = _get_non_spatial_labels(
        dataset, variable, non_spatial_labels, logger
    )
    if non_spatial_labels:
        non_spatial_labels_safe = ensure_time_label_compatible(
            variable, non_spatial_labels
        )
        variable = variable.sel(**non_spatial_labels_safe, method="nearest")
    return variable


class TileException(Exception):
    def __init__(self, message: str, logger: Optional[logging.Logger] = None):
        super().__init__(message)
        if logger is not None:
            logger.warning(message)


class TileNotFoundException(TileException):
    pass


class TileRequestException(TileException):
    pass


class TransparentRgbaTilePool:
    """A cache for fully-transparent RGBA tiles of a given size and format."""

    INSTANCE: "TransparentRgbaTilePool"

    def __init__(self):
        self._transparent_tiles: Dict[str, Union[bytes, np.ndarray]] = dict()

    def get(self, tile_size: Pair[int], format: str) -> Union[bytes, np.ndarray]:
        tile_w, tile_h = tile_size
        key = f"{format}-{tile_w}-{tile_h}"
        if key not in self._transparent_tiles:
            data = np.zeros((tile_h, tile_w, 4), dtype=np.uint8)
            if format == "png":
                data = _encode_rgba_as_png(data)
            self._transparent_tiles[key] = data
        return self._transparent_tiles[key]


TransparentRgbaTilePool.INSTANCE = TransparentRgbaTilePool()


def _get_non_spatial_labels(
    dataset: xr.Dataset,
    variable: xr.DataArray,
    labels: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[Hashable, Any]:
    labels = labels if labels is not None else {}

    new_labels = {}
    # assuming last two dims are spatial: [..., y, x]
    assert variable.ndim >= 2
    non_spatial_dims = variable.dims[0:-2]
    if not non_spatial_dims:
        #  Ignore any extra labels passed.
        return new_labels

    for dim in non_spatial_dims:
        try:
            coord_var = dataset.coords[dim]
            if coord_var.size == 0:
                continue
        except KeyError:
            continue

        dim_name = str(dim)

        label = labels.get(dim_name)
        if label is None:
            if logger:
                logger.debug(
                    (
                        f"missing label for dimension {dim!r},"
                        f" using first label instead"
                    )
                )
            label = coord_var[0].values

        elif isinstance(label, str):
            if "/" in label:
                # In case of WMTS tile requests the tame range labels
                # from WMTS dimensions may be passed.
                label = label.split("/", maxsplit=1)[0]

            if label.lower() == "first":
                label = coord_var[0].values
            elif label.lower() in ("last", "current"):
                label = coord_var[-1].values
            else:
                try:
                    if np.issubdtype(coord_var.dtype, np.datetime64):
                        timestamp = pd.Timestamp(label)
                        if timestamp.tz is not None:
                            # Convert the timestamp to timezone-naive, since
                            # NumPy doesn't like parsing timezone-aware
                            # representations into datetime64 types.
                            label = timestamp.tz_convert(None).isoformat()
                    label = np.array(label).astype(coord_var.dtype)
                except (TypeError, ValueError) as e:
                    raise TileRequestException(
                        f"Illegal label {label!r} for dimension {dim!r}"
                    ) from e

        new_labels[dim] = label

    return new_labels


def _normalize_format(format: str) -> str:
    if format in ("png", "PNG", "image/png"):
        return "png"
    return format


def _encode_rgba_as_png(rgba_array: np.ndarray) -> bytes:
    # noinspection PyUnresolvedReferences
    image = PIL.Image.fromarray(rgba_array)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return bytes(stream.getvalue())
