# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Optional

import matplotlib.cm
import matplotlib.colors
import numpy as np
import pyproj
import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.tile import compute_rgba_tile
from xcube.core.tile import compute_tiles
from xcube.core.tile import get_var_valid_range
from xcube.core.tilingscheme import GEOGRAPHIC_CRS_NAME
from xcube.core.tilingscheme import WEB_MERCATOR_CRS_NAME
from xcube.util.cmaps import Colormap
from xcube.util.cmaps import ColormapProvider

nan = np.nan


class ColormapProviderMock(ColormapProvider):
    def get_cmap(self, cm_name: str, num_colors: Optional[int] = None):
        cmap = matplotlib.colormaps[cm_name]
        if num_colors is not None:
            cmap = cmap.resampled(lutsize=num_colors)
        return cmap, Colormap(cm_name, cmap=cmap)


CMAP_PROVIDER = ColormapProviderMock()


class TileTest(unittest.TestCase):
    @staticmethod
    def _get_ml_dataset(crs_name: str) -> MultiLevelDataset:
        crs = pyproj.CRS.from_string(crs_name)
        geo_crs = pyproj.CRS.from_string(GEOGRAPHIC_CRS_NAME)
        transformer = pyproj.Transformer.from_crs(geo_crs, crs, always_xy=True)
        (x1, x2), (y1, y2) = transformer.transform((-180, 180), (-85.051129, 85.051129))

        data = np.array(
            [
                [
                    [5, 1, 1, 1, 1, 2, 2, 2, 2, 6],
                    [1, 5, 1, 1, 1, 2, 2, 2, 6, 2],
                    [1, 1, 5, 1, 1, 2, 2, 6, 2, 2],
                    [1, 1, 1, 5, 1, 2, 6, 2, 2, 2],
                    [1, 1, 1, 1, 5, 6, 2, 2, 2, 2],
                    [3, 3, 3, 3, 6, 5, 4, 4, 4, 4],
                    [3, 3, 3, 6, 3, 4, 5, 4, 4, 4],
                    [3, 3, 6, 3, 3, 4, 4, 5, 4, 4],
                    [3, 6, 3, 3, 3, 4, 4, 4, 5, 4],
                    [6, 3, 3, 3, 3, 4, 4, 4, 4, 5],
                ]
            ],
            dtype=np.float32,
        )
        a_data = np.where(data != 6, data + 0, np.nan)
        b_data = np.where(data != 5, data + 1, np.nan)
        c_data = np.where(data != 4, data + 2, np.nan)
        h, w = a_data.shape[-2:]
        res = (x2 - x1) / w
        x1 += 0.5 * res
        x2 -= 0.5 * res
        y1 += 0.5 * res
        y2 -= 0.5 * res

        var_a = xr.DataArray(a_data, dims=["time", "y", "x"])
        var_b = xr.DataArray(b_data, dims=["time", "y", "x"])
        var_c = xr.DataArray(c_data, dims=["time", "y", "x"])
        spatial_ref = xr.DataArray(0, attrs=pyproj.CRS.from_string(crs_name).to_cf())

        ds = xr.Dataset(
            data_vars=dict(
                var_a=var_a, var_b=var_b, var_c=var_c, spatial_ref=spatial_ref
            ),
            coords=dict(
                time=xr.DataArray(np.array([0]), dims="time"),
                y=xr.DataArray(np.linspace(y1, y2, h), dims="y"),
                x=xr.DataArray(np.linspace(x1, x2, w), dims="x"),
            ),
        )
        # ds = ds.chunk(dict(x=4, y=3))
        return BaseMultiLevelDataset(ds)


class ComputeTilesTest(TileTest, unittest.TestCase):
    crs_name = WEB_MERCATOR_CRS_NAME
    ml_ds = TileTest._get_ml_dataset(crs_name)
    tile_w = 12
    tile_h = 8
    args = [ml_ds, ("var_a", "var_b", "var_c"), (-180, -90, 180, 90)]
    kwargs = dict(
        tile_crs=GEOGRAPHIC_CRS_NAME,
        tile_size=(tile_w, tile_h),
        level=0,
        non_spatial_labels={"time": 0},
        tile_enlargement=0,
    )

    def test_compute_tiles_as_ndarrays(self):
        tiles = compute_tiles(*self.args, **self.kwargs)
        self.assertIsInstance(tiles, list)
        self.assert_tiles_ok(self.tile_w, self.tile_h, tiles)

    def test_compute_tiles_as_dataset(self):
        dataset = compute_tiles(*self.args, **self.kwargs, as_dataset=True)
        self.assertIsInstance(dataset, xr.Dataset)

        # Test spatial reference
        self.assertIn("crs", dataset)
        self.assertIn("crs_wkt", dataset["crs"].attrs)
        self.assertIn(
            "latitude_longitude", dataset["crs"].attrs.get("grid_mapping_name")
        )

        # Test data variables
        tiles: list[np.ndarray] = []
        for var_name in ("var_a", "var_b", "var_c"):
            self.assertIn(var_name, dataset.data_vars)
            var = dataset[var_name]
            self.assertEqual(("time", "y", "x"), var.dims)
            self.assertEqual((1, self.tile_h, self.tile_w), var.shape)
            tiles.append(var.data[0, :, :])
        self.assert_tiles_ok(self.tile_w, self.tile_h, tiles)

        # Test coordinates
        dim_sizes = ("time", 1), ("y", self.tile_h), ("x", self.tile_w)
        self.assertEqual(dict(dim_sizes), dataset.sizes)
        for var_name, expected_size in dim_sizes:
            self.assertIn(var_name, dataset.coords)
            var = dataset[var_name]
            self.assertEqual(expected_size, len(var))

    def assert_tiles_ok(
        self, expected_tile_w: int, expected_tile_h: int, actual_tiles: list[np.ndarray]
    ):
        self.assertEqual(3, len(actual_tiles))
        for i in range(3):
            self.assertIsInstance(actual_tiles[i], np.ndarray)
            self.assertEqual(np.float32, actual_tiles[i].dtype)
            self.assertEqual((expected_tile_h, expected_tile_w), actual_tiles[i].shape)
        # print(f'{tiles[0]!r}')
        np.testing.assert_equal(
            actual_tiles[0],
            np.array(
                [
                    [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 2.0, nan, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, nan, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, nan, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, nan, nan, 5.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, nan, 3.0, 3.0, 4.0, 5.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, nan, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )
        # print(f'{tiles[1]!r}')
        np.testing.assert_equal(
            actual_tiles[1],
            np.array(
                [
                    [2.0, 2.0, 2.0, nan, 2.0, 2.0, 2.0, 3.0, 3.0, 7.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, nan, 2.0, 2.0, 2.0, 3.0, 3.0, 7.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, nan, 2.0, 2.0, 3.0, 7.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, 7.0, 3.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, 7.0, 3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0, nan, 5.0, 5.0, 5.0, 5.0],
                    [4.0, 4.0, 4.0, 4.0, 7.0, 4.0, 4.0, 5.0, nan, 5.0, 5.0, 5.0],
                    [4.0, 4.0, 4.0, 7.0, 4.0, 4.0, 4.0, 5.0, 5.0, nan, 5.0, 5.0],
                ],
                dtype=np.float32,
            ),
        )
        # print(f'{tiles[2]!r}')
        np.testing.assert_equal(
            actual_tiles[2],
            np.array(
                [
                    [3.0, 3.0, 3.0, 7.0, 3.0, 3.0, 3.0, 4.0, 4.0, 8.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 7.0, 3.0, 3.0, 3.0, 4.0, 4.0, 8.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 7.0, 3.0, 3.0, 4.0, 8.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 8.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 8.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 7.0, nan, nan, nan, nan],
                    [5.0, 5.0, 5.0, 5.0, 8.0, 5.0, 5.0, nan, 7.0, nan, nan, nan],
                    [5.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0, nan, nan, 7.0, nan, nan],
                ],
                dtype=np.float32,
            ),
        )

    def test_compute_tiles_with_assignment_expr(self):
        tiles = compute_tiles(
            self.args[0],
            "var_d = (var_a + var_b + var_c) / 3",
            self.args[2],
            **self.kwargs,
        )
        self.assertIsInstance(tiles, list)
        self.assertEqual(1, len(tiles))
        tile = tiles[0]
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual((self.tile_h, self.tile_w), tile.shape)
        np.testing.assert_equal(
            tile,
            np.array(
                [
                    [2.0, 2.0, 2.0, nan, 2.0, 2.0, 2.0, 3.0, 3.0, nan, 3.0, 3.0],
                    [2.0, 2.0, 2.0, nan, 2.0, 2.0, 2.0, 3.0, 3.0, nan, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, nan, 2.0, 2.0, 3.0, nan, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, nan, 3.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, nan, 3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                    [4.0, 4.0, 4.0, 4.0, nan, 4.0, 4.0, nan, nan, nan, nan, nan],
                    [4.0, 4.0, 4.0, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan],
                ],
                dtype=np.float32,
            ),
        )


class ComputeRgbaTileTest(TileTest, unittest.TestCase):
    def test_compute_rgba_tile_with_color_mapping(self):
        crs_name = WEB_MERCATOR_CRS_NAME
        ml_ds = self._get_ml_dataset(crs_name)
        args = [ml_ds, "var_a", 0, 0, 0, CMAP_PROVIDER]
        kwargs = dict(
            crs_name=crs_name,
            tile_size=10,
            cmap_name="gray",
            value_ranges=(0, 10),
            non_spatial_labels={"time": 0},
            tile_enlargement=0,
        )
        tile = compute_rgba_tile(*args, **kwargs, format="numpy")
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(np.uint8, tile.dtype)
        self.assertEqual((10, 10, 4), tile.shape)
        r = tile[:, :, 0]
        g = tile[:, :, 1]
        b = tile[:, :, 2]
        a = tile[:, :, 3]
        np.testing.assert_equal(
            r,
            np.array(
                [
                    [0, 0, 76, 76, 76, 76, 102, 102, 102, 128],
                    [76, 76, 0, 76, 76, 76, 102, 102, 128, 102],
                    [76, 76, 76, 0, 76, 76, 102, 128, 102, 102],
                    [76, 76, 76, 76, 0, 0, 128, 102, 102, 102],
                    [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                    [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                    [25, 25, 25, 128, 25, 25, 51, 0, 51, 51],
                    [25, 25, 128, 25, 25, 25, 51, 51, 0, 51],
                    [128, 128, 25, 25, 25, 25, 51, 51, 51, 0],
                    [128, 128, 25, 25, 25, 25, 51, 51, 51, 0],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(r, g)
        np.testing.assert_equal(r, b)
        np.testing.assert_equal(
            a,
            np.array(
                [
                    [0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 0, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 255, 0, 255, 255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 0, 0, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 0, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255, 0, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                ],
                dtype=np.uint8,
            ),
        )

        tile = compute_rgba_tile(*args, **kwargs, format="png")
        self.assertIsInstance(tile, bytes)

    def test_compute_rgba_tile_with_components(self):
        crs_name = WEB_MERCATOR_CRS_NAME
        ml_ds = self._get_ml_dataset(crs_name)
        args = [ml_ds, ("var_a", "var_b", "var_c"), 0, 0, 0, CMAP_PROVIDER]
        kwargs = dict(
            crs_name=crs_name,
            tile_size=10,
            value_ranges=(0, 10),
            non_spatial_labels={"time": 0},
            tile_enlargement=0,
        )
        tile = compute_rgba_tile(*args, **kwargs, format="numpy")
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(np.uint8, tile.dtype)
        self.assertEqual((10, 10, 4), tile.shape)
        r = tile[:, :, 0]
        g = tile[:, :, 1]
        b = tile[:, :, 2]
        a = tile[:, :, 3]
        np.testing.assert_equal(
            r,
            np.array(
                [
                    [0, 0, 76, 76, 76, 76, 102, 102, 102, 127],
                    [76, 76, 0, 76, 76, 76, 102, 102, 127, 102],
                    [76, 76, 76, 0, 76, 76, 102, 127, 102, 102],
                    [76, 76, 76, 76, 0, 0, 127, 102, 102, 102],
                    [25, 25, 25, 25, 127, 127, 0, 51, 51, 51],
                    [25, 25, 25, 25, 127, 127, 0, 51, 51, 51],
                    [25, 25, 25, 127, 25, 25, 51, 0, 51, 51],
                    [25, 25, 127, 25, 25, 25, 51, 51, 0, 51],
                    [127, 127, 25, 25, 25, 25, 51, 51, 51, 0],
                    [127, 127, 25, 25, 25, 25, 51, 51, 51, 0],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(
            g,
            np.array(
                [
                    [178, 178, 102, 102, 102, 102, 127, 127, 127, 0],
                    [102, 102, 178, 102, 102, 102, 127, 127, 0, 127],
                    [102, 102, 102, 178, 102, 102, 127, 0, 127, 127],
                    [102, 102, 102, 102, 178, 178, 0, 127, 127, 127],
                    [51, 51, 51, 51, 0, 0, 178, 76, 76, 76],
                    [51, 51, 51, 51, 0, 0, 178, 76, 76, 76],
                    [51, 51, 51, 0, 51, 51, 76, 178, 76, 76],
                    [51, 51, 0, 51, 51, 51, 76, 76, 178, 76],
                    [0, 0, 51, 51, 51, 51, 76, 76, 76, 178],
                    [0, 0, 51, 51, 51, 51, 76, 76, 76, 178],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(
            b,
            np.array(
                [
                    [204, 204, 127, 127, 127, 127, 0, 0, 0, 178],
                    [127, 127, 204, 127, 127, 127, 0, 0, 178, 0],
                    [127, 127, 127, 204, 127, 127, 0, 178, 0, 0],
                    [127, 127, 127, 127, 204, 204, 178, 0, 0, 0],
                    [76, 76, 76, 76, 178, 178, 204, 102, 102, 102],
                    [76, 76, 76, 76, 178, 178, 204, 102, 102, 102],
                    [76, 76, 76, 178, 76, 76, 102, 204, 102, 102],
                    [76, 76, 178, 76, 76, 76, 102, 102, 204, 102],
                    [178, 178, 76, 76, 76, 76, 102, 102, 102, 204],
                    [178, 178, 76, 76, 76, 76, 102, 102, 102, 204],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(
            a,
            np.array(
                [
                    [0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                    [255, 255, 0, 255, 255, 255, 0, 0, 0, 0],
                    [255, 255, 255, 0, 255, 255, 0, 0, 0, 0],
                    [255, 255, 255, 255, 0, 0, 0, 0, 0, 0],
                    [255, 255, 255, 255, 0, 0, 0, 255, 255, 255],
                    [255, 255, 255, 255, 0, 0, 0, 255, 255, 255],
                    [255, 255, 255, 0, 255, 255, 255, 0, 255, 255],
                    [255, 255, 0, 255, 255, 255, 255, 255, 0, 255],
                    [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                    [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                ],
                dtype=np.uint8,
            ),
        )

        tile = compute_rgba_tile(*args, **kwargs, format="png")
        self.assertIsInstance(tile, bytes)


class GetVarValidRangeTest(unittest.TestCase):
    def test_from_valid_range(self):
        a = xr.DataArray(0, attrs=dict(valid_range=[-1, 1]))
        self.assertEqual((-1, 1), get_var_valid_range(a))

    def test_from_valid_min_max(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1, valid_max=1))
        self.assertEqual((-1, 1), get_var_valid_range(a))

    def test_from_valid_min(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1))
        self.assertEqual((-1, np.inf), get_var_valid_range(a))

    def test_from_valid_max(self):
        a = xr.DataArray(0, attrs=dict(valid_max=1))
        self.assertEqual((-np.inf, 1), get_var_valid_range(a))

    def test_from_nothing(self):
        a = xr.DataArray(0)
        self.assertEqual(None, get_var_valid_range(a))
