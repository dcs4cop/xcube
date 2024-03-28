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

import abc
import copy
import math
import threading
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.dask import get_block_iterators
from xcube.util.dask import get_chunk_sizes
from xcube.constants import CRS_CRS84
from .helpers import AffineTransformMatrix
from .helpers import Number
from .helpers import _assert_valid_xy_coords
from .helpers import _assert_valid_xy_names
from .helpers import _from_affine
from .helpers import _normalize_int_pair
from .helpers import _normalize_number_pair
from .helpers import _to_affine
from .helpers import scale_xy_res_and_size

# WGS84, axis order: lat, lon
CRS_WGS84 = pyproj.crs.CRS(4326)


# Default tolerance for all operations that
# accept a key-word argument "tolerance":
DEFAULT_TOLERANCE = 1.0e-5


class GridMapping(abc.ABC):
    """An abstract base class for grid mappings that define an image grid and
    a transformation from image pixel coordinates to spatial Earth coordinates
    defined in a well-known coordinate reference system (CRS).

    This class cannot be instantiated directly. Use one of its factory methods
    to create instances:

    * `regular()`
    * `from_dataset()`
    * `from_coords()`

    Some instance methods can be used to derive new instances:

    * `derive()`
    * `scale()`
    * `transform()`
    * `to_regular()`

    This class is thread-safe.

    """

    def __init__(
        self,
        /,
        size: Union[int, Tuple[int, int]],
        tile_size: Optional[Union[int, Tuple[int, int]]],
        xy_bbox: Tuple[Number, Number, Number, Number],
        xy_res: Union[Number, Tuple[Number, Number]],
        crs: pyproj.crs.CRS,
        xy_var_names: Tuple[str, str],
        xy_dim_names: Tuple[str, str],
        is_regular: Optional[bool],
        is_lon_360: Optional[bool],
        is_j_axis_up: Optional[bool],
        x_coords: Optional[xr.DataArray] = None,
        y_coords: Optional[xr.DataArray] = None,
        xy_coords: Optional[xr.DataArray] = None,
    ):
        width, height = _normalize_int_pair(size, name="size")
        assert_true(width > 1 and height > 1, "invalid size")

        tile_width, tile_height = _normalize_int_pair(
            tile_size, default=(width, height)
        )
        assert_true(tile_width > 1 and tile_height > 1, "invalid tile_size")

        assert_given(xy_bbox, name="xy_bbox")
        assert_given(xy_res, name="xy_res")
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")
        _assert_valid_xy_names(xy_dim_names, name="xy_dim_names")
        assert_instance(crs, pyproj.crs.CRS, name="crs")

        if x_coords is not None:
            assert_instance(x_coords, xr.DataArray, name="x_coords")
            assert_true(
                x_coords.ndim in (1, 2),
                message=f"x_coords.ndim must be 1 or 2," f" was {x_coords.ndim}",
            )
        if y_coords is not None:
            assert_instance(y_coords, xr.DataArray, name="y_coords")
            assert_true(
                y_coords.ndim in (1, 2),
                message=f"y_coords.ndim must be 1 or 2," f" was {y_coords.ndim}",
            )
        if xy_coords is not None:
            assert_instance(xy_coords, xr.DataArray, name="xy_coords")
            assert_true(
                xy_coords.shape == (2, height, width),
                message=f"xy_coords.shape must be"
                f" {(2, height, width)},"
                f" was {xy_coords.shape}",
            )

        x_min, y_min, x_max, y_max = xy_bbox
        x_res, y_res = _normalize_number_pair(xy_res, name="xy_res")
        assert_true(x_res > 0 and y_res > 0, "invalid xy_res")

        self._lock = threading.RLock()

        self._size = width, height
        self._tile_size = tile_width, tile_height
        self._xy_bbox = x_min, y_min, x_max, y_max
        self._xy_res = x_res, y_res
        self._crs = crs
        self._xy_var_names = xy_var_names
        self._xy_dim_names = xy_dim_names
        self._is_regular = is_regular
        self._is_lon_360 = is_lon_360
        self._is_j_axis_up = is_j_axis_up
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._xy_coords = xy_coords

    def derive(
        self,
        /,
        xy_var_names: Tuple[str, str] = None,
        xy_dim_names: Tuple[str, str] = None,
        tile_size: Union[int, Tuple[int, int]] = None,
        is_j_axis_up: bool = None,
    ):
        """Derive a new grid mapping from this one with some properties changed.

        Args:
            xy_var_names: The new x-, and y-variable names.
            xy_dim_names: The new x-, and y-dimension names.
            tile_size: The new tile size
            is_j_axis_up: Whether j-axis points up.

        Returns:
            A new, derived grid mapping.
        """
        other = copy.copy(self)
        if xy_var_names is not None:
            _assert_valid_xy_names(xy_var_names, name="xy_var_names")
            other._xy_var_names = xy_var_names
        if xy_dim_names is not None:
            _assert_valid_xy_names(xy_dim_names, name="xy_dim_names")
            other._xy_dim_names = xy_dim_names
        if tile_size is not None:
            tile_width, tile_height = _normalize_int_pair(tile_size, name="tile_size")
            assert_true(tile_width > 1 and tile_height > 1, "invalid tile_size")
            tile_size = tile_width, tile_height
            if other.tile_size != tile_size:
                other._tile_size = tile_width, tile_height
                with self._lock:
                    if other._xy_coords is not None:
                        other._xy_coords = other._xy_coords.chunk(
                            other.xy_coords_chunks
                        )
        if is_j_axis_up is not None and is_j_axis_up != other._is_j_axis_up:
            other._is_j_axis_up = is_j_axis_up
            if other._y_coords is not None:
                other._y_coords = other._y_coords[::-1]
            if other._xy_coords is not None:
                other._xy_coords = other._xy_coords[:, ::-1, :]
                other._xy_coords = other._xy_coords.chunk(other.xy_coords_chunks)

        return other

    def scale(
        self,
        xy_scale: Union[Number, Tuple[Number, Number]],
        tile_size: Union[int, Tuple[int, int]] = None,
    ) -> "GridMapping":
        """Derive a scaled version of this regular grid mapping.

        Scaling factors lower than one correspond to up-scaling
        (pixels sizes decrease, image size increases).

        Scaling factors larger than one correspond to down-scaling.
        (pixels sizes increase, image size decreases).

        Args:
            xy_scale: The x-, and y-scaling factors. May be a single
                number or tuple.
            tile_size: The new tile size

        Returns:
            A new, scaled grid mapping.
        """
        self._assert_regular()
        x_scale, y_scale = _normalize_number_pair(xy_scale)
        new_xy_res, new_size = scale_xy_res_and_size(
            self.xy_res, self.size, (x_scale, y_scale)
        )
        if tile_size is not None:
            tile_width, tile_height = _normalize_int_pair(tile_size, name="tile_size")
        else:
            tile_width, tile_height = self.tile_size
        tile_width = min(new_size[0], tile_width)
        tile_height = min(new_size[1], tile_height)
        return self.regular(
            new_size,
            (self.x_min, self.y_min),
            new_xy_res,
            self.crs,
            tile_size=(tile_width, tile_height),
            is_j_axis_up=self.is_j_axis_up,
        ).derive(xy_dim_names=self.xy_dim_names, xy_var_names=self.xy_var_names)

    @property
    def size(self) -> Tuple[int, int]:
        """Image size (width, height) in pixels."""
        return self._size

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.size[1]

    @property
    def tile_size(self) -> Tuple[int, int]:
        """Image tile size (width, height) in pixels."""
        return self._tile_size

    @property
    def is_tiled(self) -> bool:
        """Whether the image is tiled."""
        return self.size != self.tile_size

    @property
    def tile_width(self) -> int:
        """Image tile width in pixels."""
        return self.tile_size[0]

    @property
    def tile_height(self) -> int:
        """Image tile height in pixels."""
        return self.tile_size[1]

    @property
    def x_coords(self):
        """The 1D or 2D x-coordinate array of
        shape (width,) or (height, width).
        """
        return self._get_computed_attribute("_x_coords", self._new_x_coords)

    @abc.abstractmethod
    def _new_x_coords(self) -> xr.DataArray:
        """Create new 1D or 2D x-coordinate array of
        shape (width,) or (height, width).
        """

    @property
    def y_coords(self):
        """The 1D or 2D y-coordinate array of
        shape (width,) or (height, width).
        """
        return self._get_computed_attribute("_y_coords", self._new_y_coords)

    @abc.abstractmethod
    def _new_y_coords(self) -> xr.DataArray:
        """Create new 1D or 2D y-coordinate array of
        shape (width,) or (height, width).
        """

    @property
    def xy_coords(self) -> xr.DataArray:
        """The x,y coordinates as data array of shape (2, height, width).
        Coordinates are given in units of the CRS.
        """
        xy_coords = self._get_computed_attribute("_xy_coords", self._new_xy_coords)
        _assert_valid_xy_coords(xy_coords)
        return xy_coords

    @property
    def xy_coords_chunks(self) -> Tuple[int, int, int]:
        """Get the chunks for the *xy_coords* array."""
        return 2, self.tile_height, self.tile_width

    @abc.abstractmethod
    def _new_xy_coords(self) -> xr.DataArray:
        """Create new coordinate array of shape (2, height, width)."""

    def _get_computed_attribute(self, name: str, computer: Callable[[], Any]) -> Any:
        """Get the value for a computed attribute.
        Utility to be used by this and derived classes.
        """
        value = getattr(self, name)
        if value is not None:
            return value
        with self._lock:
            # Double null check
            value = getattr(self, name)
            if value is not None:
                return value
            value = computer()
            setattr(self, name, value)
            return value

    @property
    def xy_var_names(self) -> Tuple[str, str]:
        """The variable names of the x,y coordinates as
        tuple (x_var_name, y_var_name).
        """
        return self._xy_var_names

    @property
    def xy_dim_names(self) -> Tuple[str, str]:
        """The dimension names of the x,y coordinates as
        tuple (x_dim_name, y_dim_name).
        """
        return self._xy_dim_names

    @property
    def xy_bbox(self) -> Tuple[float, float, float, float]:
        """The image's bounding box in CRS coordinates."""
        return self._xy_bbox

    @property
    def x_min(self) -> Number:
        """Minimum x-coordinate in CRS units."""
        return self._xy_bbox[0]

    @property
    def y_min(self) -> Number:
        """Minimum y-coordinate in CRS units."""
        return self._xy_bbox[1]

    @property
    def x_max(self) -> Number:
        """Maximum x-coordinate in CRS units."""
        return self._xy_bbox[2]

    @property
    def y_max(self) -> Number:
        """Maximum y-coordinate in CRS units."""
        return self._xy_bbox[3]

    @property
    def xy_res(self) -> Tuple[Number, Number]:
        """Pixel size in x and y direction."""
        return self._xy_res

    @property
    def x_res(self) -> Number:
        """Pixel size in CRS units per pixel in x-direction."""
        return self._xy_res[0]

    @property
    def y_res(self) -> Number:
        """Pixel size in CRS units per pixel in y-direction."""
        return self._xy_res[1]

    @property
    def crs(self) -> pyproj.crs.CRS:
        """The coordinate reference system."""
        return self._crs

    @property
    def spatial_unit_name(self) -> str:
        return self._crs.axis_info[0].unit_name

    @property
    def is_lon_360(self) -> Optional[bool]:
        """Check whether *x_max* is greater than 180 degrees.
        Effectively tests whether the range *x_min*, *x_max* crosses
        the anti-meridian at 180 degrees.
        Works only for geographical coordinate reference systems.
        """
        return self._is_lon_360

    @property
    def is_regular(self) -> Optional[bool]:
        """Do the x,y coordinates for a regular grid?
        A regular grid has a constant delta in both
        x- and y-directions of the x- and y-coordinates.

        Returns:
            `None`, if this property cannot be determined,
            `True` or `False` otherwise.
        """
        return self._is_regular

    @property
    def is_j_axis_up(self) -> Optional[bool]:
        """Does the positive image j-axis point up?
        By default, the positive image j-axis points down.

        Returns:
            `None`, if this property cannot be determined,
            `True` or `False` otherwise.
        """
        return self._is_j_axis_up

    @property
    def ij_to_xy_transform(self) -> AffineTransformMatrix:
        """The affine transformation matrix from image to CRS coordinates.
        Defined only for grid mappings with rectified x,y coordinates.
        """
        self._assert_regular()
        if self.is_j_axis_up:
            return (
                (self.x_res, 0.0, self.x_min),
                (0.0, self.y_res, self.y_min),
            )
        else:
            return (
                (self.x_res, 0.0, self.x_min),
                (0.0, -self.y_res, self.y_max),
            )

    @property
    def xy_to_ij_transform(self) -> AffineTransformMatrix:
        """The affine transformation matrix from CRS to image coordinates.
        Defined only for grid mappings with rectified x,y coordinates.
        """
        self._assert_regular()
        return _from_affine(~_to_affine(self.ij_to_xy_transform))

    def ij_transform_to(self, other: "GridMapping") -> AffineTransformMatrix:
        """Get the affine transformation matrix that transforms
        image coordinates of *other* into image coordinates
        of this image geometry.

        Defined only for grid mappings with rectified x,y coordinates.

        Args:
            other: The other image geometry

        Returns:
            Affine transformation matrix
        """
        self._assert_regular()
        self.assert_regular(other, name="other")
        a = _to_affine(self.ij_to_xy_transform)
        b = _to_affine(other.xy_to_ij_transform)
        return _from_affine(b * a)

    def ij_transform_from(self, other: "GridMapping") -> AffineTransformMatrix:
        """Get the affine transformation matrix that transforms
        image coordinates of this image geometry to image
        coordinates of *other*.

        Defined only for grid mappings with rectified x,y coordinates.

        Args:
            other: The other image geometry

        Returns:
            Affine transformation matrix
        """
        self._assert_regular()
        self.assert_regular(other, name="other")
        a = _to_affine(self.ij_transform_to(other))
        return _from_affine(~a)

    @property
    def ij_bbox(self) -> Tuple[int, int, int, int]:
        """The image's bounding box in pixel coordinates."""
        return 0, 0, self.width, self.height

    @property
    def ij_bboxes(self) -> np.ndarray:
        """The image tiles' bounding boxes in image pixel coordinates."""
        chunk_sizes = get_chunk_sizes(
            (self.height, self.width), (self.tile_height, self.tile_width)
        )
        _, _, block_slices = get_block_iterators(chunk_sizes)
        block_slices = tuple(block_slices)
        n = len(block_slices)
        ij_bboxes = np.ndarray((n, 4), dtype=np.int64)
        for i in range(n):
            y_slice, x_slice = block_slices[i]
            ij_bboxes[i, 0] = x_slice.start
            ij_bboxes[i, 1] = y_slice.start
            ij_bboxes[i, 2] = x_slice.stop
            ij_bboxes[i, 3] = y_slice.stop
        return ij_bboxes

    @property
    def xy_bboxes(self) -> np.ndarray:
        """The image tiles' bounding boxes in CRS coordinates."""
        if self.is_j_axis_up:
            xy_offset = np.array([self.x_min, self.y_min, self.x_min, self.y_min])
            xy_scale = np.array([self.x_res, self.y_res, self.x_res, self.y_res])
            xy_bboxes = xy_offset + xy_scale * self.ij_bboxes
        else:
            xy_offset = np.array([self.x_min, self.y_max, self.x_min, self.y_max])
            xy_scale = np.array([self.x_res, -self.y_res, self.x_res, -self.y_res])
            xy_bboxes = xy_offset + xy_scale * self.ij_bboxes
            xy_bboxes[:, [1, 3]] = xy_bboxes[:, [3, 1]]
        return xy_bboxes

    def ij_bbox_from_xy_bbox(
        self,
        xy_bbox: Tuple[float, float, float, float],
        xy_border: float = 0.0,
        ij_border: int = 0,
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box in i,j pixel coordinates given a
        bounding box *xy_bbox* in x,y coordinates.

        Args:
            xy_bbox: Box (x_min, y_min, x_max, y_max) given in the same
                CS as x and y.
            xy_border: If non-zero, grows the bounding box *xy_bbox*
                before using it for comparisons. Defaults to 0.
            ij_border: If non-zero, grows the returned i,j bounding box
                and clips it to size. Defaults to 0.

        Returns:
            Bounding box in (i_min, j_min, i_max, j_max) in pixel
            coordinates. Returns `(-1, -1, -1, -1)` if *xy_bbox* isn't
            intersecting any of the x,y coordinates.
        """
        xy_bboxes = np.array([xy_bbox], dtype=np.float64)
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        self.ij_bboxes_from_xy_bboxes(
            xy_bboxes, xy_border=xy_border, ij_border=ij_border, ij_bboxes=ij_bboxes
        )
        # noinspection PyTypeChecker
        return tuple(map(int, ij_bboxes[0]))

    def ij_bboxes_from_xy_bboxes(
        self,
        xy_bboxes: np.ndarray,
        xy_border: float = 0.0,
        ij_border: int = 0,
        ij_bboxes: np.ndarray = None,
    ) -> np.ndarray:
        """Compute bounding boxes in pixel coordinates given bounding boxes
        *xy_bboxes* [[x_min, y_min, x_max, y_max], ...] in x,y coordinates.

        The returned array in i,j pixel coordinates
        has the same shape as *xy_bboxes*. The value ranges in the
        returned array [[i_min, j_min, i_max, j_max], ..]] are:

        * i_min from 0 to width-1, i_max from 1 to width;
        * j_min from 0 to height-1, j_max from 1 to height;

        so the i,j pixel coordinates can be used as array index slices.

        Args:
            xy_bboxes: Numpy array of x,y bounding boxes [[x_min, y_min,
                x_max, y_max], ...] given in the same CS as x and y.
            xy_border: If non-zero, grows the bounding box *xy_bbox*
                before using it for comparisons. Defaults to 0.
            ij_border: If non-zero, grows the returned i,j bounding box
                and clips it to size. Defaults to 0.
            ij_bboxes: Numpy array of pixel i,j bounding boxes [[x_min,
                y_min, x_max, y_max], ...]. If given, must have same
                shape as *xy_bboxes*.

        Returns:
            Bounding boxes in [[i_min, j_min, i_max, j_max], ..]] in
            pixel coordinates.
        """
        from .bboxes import compute_ij_bboxes

        if ij_bboxes is None:
            ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        else:
            ij_bboxes[:, :] = -1
        xy_coords = self.xy_coords.values
        compute_ij_bboxes(
            xy_coords[0], xy_coords[1], xy_bboxes, xy_border, ij_border, ij_bboxes
        )
        return ij_bboxes

    def to_dataset_attrs(self) -> Mapping[str, Any]:
        """Get spatial dataset attributes as recommended by
        https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#Recommended

        Returns:
            dictionary with dataset coordinate attributes.
        """

        x1, y1, x2, y2 = self.xy_bbox

        if self.crs.is_geographic:
            lon_min, lat_min, lon_max, lat_max = self.xy_bbox
            lon_res, lat_res = self.xy_res
        else:
            x_res, y_res = self.xy_res
            # center position
            xm1 = (x1 + x2) / 2
            ym1 = (y1 + y2) / 2
            # center position + delta
            xm2 = xm1 + x_res
            ym2 = ym1 + y_res
            transformer = pyproj.Transformer.from_crs(
                crs_from=self.crs, crs_to=CRS_CRS84
            )
            xx, yy = transformer.transform((x1, x2, xm1, xm2), (y1, y2, ym1, ym2))
            lon_min, lon_max, lon_m1, lon_m2 = xx
            lat_min, lat_max, lat_m1, lat_m2 = yy
            # Estimate resolution (note, this may be VERY wrong)
            lon_res = abs(lon_m2 - lon_m1)
            lat_res = abs(lat_m2 - lat_m1)

        geospatial_bounds_crs = "CRS84"
        geospatial_bounds = (
            f"POLYGON(("
            f"{lon_min} {lat_min}, "
            f"{lon_min} {lat_max}, "
            f"{lon_max} {lat_max}, "
            f"{lon_max} {lat_min}, "
            f"{lon_min} {lat_min}"
            f"))"
        )

        return dict(
            geospatial_lon_units="degrees_east",
            geospatial_lon_min=lon_min,
            geospatial_lon_max=lon_max,
            geospatial_lon_resolution=lon_res,
            geospatial_lat_units="degrees_north",
            geospatial_lat_min=lat_min,
            geospatial_lat_max=lat_max,
            geospatial_lat_resolution=lat_res,
            geospatial_bounds_crs=geospatial_bounds_crs,
            geospatial_bounds=geospatial_bounds,
        )

    def to_coords(
        self,
        xy_var_names: Tuple[str, str] = None,
        xy_dim_names: Tuple[str, str] = None,
        exclude_bounds: bool = False,
        reuse_coords: bool = False,
    ) -> Mapping[str, xr.DataArray]:
        """Get CF-compliant axis coordinate variables and cell boundary
        coordinate variables.

        Defined only for grid mappings with regular x,y coordinates.

        Args:
            xy_var_names: Optional coordinate variable names
                (x_var_name, y_var_name).
            xy_dim_names: Optional coordinate dimensions names
                (x_dim_name, y_dim_name).
            exclude_bounds: If True, do not create bounds coordinates.
                Defaults to False.
            reuse_coords: Whether to either reuse target coordinate
                arrays from target_gm or to compute new ones.

        Returns:
            dictionary with coordinate variables
        """
        self._assert_regular()
        from .coords import grid_mapping_to_coords

        return grid_mapping_to_coords(
            self,
            xy_var_names=xy_var_names,
            xy_dim_names=xy_dim_names,
            exclude_bounds=exclude_bounds,
            reuse_coords=reuse_coords,
        )

    def transform(
        self,
        crs: Union[str, pyproj.crs.CRS],
        *,
        tile_size: Union[int, Tuple[int, int]] = None,
        xy_var_names: Tuple[str, str] = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> "GridMapping":
        """Transform this grid mapping so it uses the given
        spatial coordinate reference system into another *crs*.

        Args:
            crs: The new spatial coordinate reference system.
            tile_size: Optional new tile size.
            xy_var_names: Optional new coordinate names.
            tolerance: Absolute tolerance used when comparing
                coordinates with each other. Must be in the units of the
                *crs* and must be greater zero.

        Returns:
            A new grid mapping that uses *crs*.
        """
        from .transform import transform_grid_mapping

        return transform_grid_mapping(
            self,
            crs,
            tile_size=tile_size,
            xy_var_names=xy_var_names,
            tolerance=tolerance,
        )

    @classmethod
    def regular(
        cls,
        size: Union[int, Tuple[int, int]],
        xy_min: Tuple[float, float],
        xy_res: Union[float, Tuple[float, float]],
        crs: Union[str, pyproj.crs.CRS],
        *,
        tile_size: Union[int, Tuple[int, int]] = None,
        is_j_axis_up: bool = False,
    ) -> "GridMapping":
        """Create a new regular grid mapping.

        Args:
            size: Size in pixels.
            xy_min: Minimum x- and y-coordinates.
            xy_res: Resolution in x- and y-directions.
            crs: Spatial coordinate reference system.
            tile_size: Optional tile size.
            is_j_axis_up: Whether positive j-axis points up. Defaults to
                false.

        Returns:
            A new regular grid mapping.
        """
        from .regular import new_regular_grid_mapping

        return new_regular_grid_mapping(
            size=size,
            xy_min=xy_min,
            xy_res=xy_res,
            crs=crs,
            tile_size=tile_size,
            is_j_axis_up=is_j_axis_up,
        )

    def to_regular(
        self, tile_size: Union[int, Tuple[int, int]] = None, is_j_axis_up: bool = False
    ) -> "GridMapping":
        """Transform this grid mapping into one that is regular.

        Args:
            tile_size: Optional tile size.
            is_j_axis_up: Whether positive j-axis points up. Defaults to
                false.

        Returns:
            A new regular grid mapping or this grid mapping, if it is
            already regular.
        """
        from .regular import to_regular_grid_mapping

        return to_regular_grid_mapping(
            self, tile_size=tile_size, is_j_axis_up=is_j_axis_up
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: xr.Dataset,
        *,
        crs: Union[str, pyproj.crs.CRS] = None,
        tile_size: Union[int, Tuple[int, int]] = None,
        prefer_is_regular: bool = True,
        prefer_crs: Union[str, pyproj.crs.CRS] = None,
        emit_warnings: bool = False,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> "GridMapping":
        """Create a grid mapping for the given *dataset*.

        Args:
            dataset: The dataset.
            crs: Optional spatial coordinate reference system.
            tile_size: Optional tile size
            prefer_is_regular: Whether to prefer a regular grid mapping
                if multiple found. Default is True.
            prefer_crs: The preferred CRS of a grid mapping if multiple
                found.
            emit_warnings: Whether to emit warning for non-CF compliant
                datasets.
            tolerance: Absolute tolerance used when comparing
                coordinates with each other. Must be in the units of the
                *crs* and must be greater zero.

        Returns:
            a new grid mapping instance.
        """
        from .dataset import new_grid_mapping_from_dataset

        return new_grid_mapping_from_dataset(
            dataset=dataset,
            crs=crs,
            tile_size=tile_size,
            prefer_is_regular=prefer_is_regular,
            prefer_crs=prefer_crs,
            emit_warnings=emit_warnings,
            tolerance=tolerance,
        )

    @classmethod
    def from_coords(
        cls,
        x_coords: xr.DataArray,
        y_coords: xr.DataArray,
        crs: Union[str, pyproj.crs.CRS],
        *,
        tile_size: Union[int, Tuple[int, int]] = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> "GridMapping":
        """Create a grid mapping from given x- and y-coordinates
        *x_coords*, *y_coords* and spatial coordinate reference
        system *crs*.

        Args:
            x_coords: The x-coordinates.
            y_coords: The y-coordinates.
            crs: The spatial coordinate reference system.
            tile_size: Optional tile size.
            tolerance: Absolute tolerance used when comparing
                coordinates with each other. Must be in the units of the
                *crs* and must be greater zero.

        Returns:
            A new grid mapping.
        """
        from .coords import new_grid_mapping_from_coords

        return new_grid_mapping_from_coords(
            x_coords=x_coords,
            y_coords=y_coords,
            crs=crs,
            tile_size=tile_size,
            tolerance=tolerance,
        )

    def is_close(
        self, other: "GridMapping", tolerance: float = DEFAULT_TOLERANCE
    ) -> bool:
        """Tests whether this grid mapping is close to *other*.

        Args:
            other: The other grid mapping.
            tolerance: Absolute tolerance used when comparing
                coordinates with each other. Must be in the units of the
                *crs* and must be greater zero.

        Returns:
            True, if so, False otherwise.
        """
        if self is other:
            return True
        if (
            self.is_j_axis_up == other.is_j_axis_up
            and self.is_lon_360 == other.is_lon_360
            and self.is_regular == other.is_regular
            and self.size == other.size
            and self.tile_size == other.tile_size
            and self.crs == other.crs
        ):
            sxr, syr = self.xy_res
            oxr, oyr = other.xy_res
            if math.isclose(sxr, oxr, abs_tol=tolerance) and math.isclose(
                syr, oyr, abs_tol=tolerance
            ):
                sx1, sy1, sx2, sy2 = self.xy_bbox
                ox1, oy1, ox2, oy2 = other.xy_bbox
                return (
                    math.isclose(sx1, ox1, abs_tol=tolerance)
                    and math.isclose(sy1, oy1, abs_tol=tolerance)
                    and math.isclose(sx2, ox2, abs_tol=tolerance)
                    and math.isclose(sy2, oy2, abs_tol=tolerance)
                )
        return False

    @classmethod
    def assert_regular(cls, value: Any, name: str = None):
        assert_instance(value, GridMapping, name=name)
        if not value.is_regular:
            raise ValueError(f'{name or "value"} must' f" be a regular grid mapping")

    def _assert_regular(self):
        if not self.is_regular:
            raise NotImplementedError(
                "Operation not implemented" " for non-regular grid mappings"
            )

    def _repr_markdown_(self) -> str:
        """Generate an IPython Notebook Markdown representation."""
        is_regular = self.is_regular if self.is_regular is not None else "_unknown_"
        is_j_axis_up = (
            self.is_j_axis_up if self.is_j_axis_up is not None else "_unknown_"
        )
        is_lon_360 = self.is_lon_360 if self.is_lon_360 is not None else "_unknown_"
        xy_res = repr(self.xy_res) + ("" if self.is_regular else "  _estimated_")
        return "\n".join(
            [
                f"class: **{self.__class__.__name__}**",
                f"* is_regular: {is_regular}",
                f"* is_j_axis_up: {is_j_axis_up}",
                f"* is_lon_360: {is_lon_360}",
                f"* crs: {self.crs}",
                f"* xy_res: {xy_res}",
                f"* xy_bbox: {self.xy_bbox}",
                f"* ij_bbox: {self.ij_bbox}",
                f"* xy_dim_names: {self.xy_dim_names}",
                f"* xy_var_names: {self.xy_var_names}",
                f"* size: {self.size}",
                f"* tile_size: {self.tile_size}",
            ]
        )
