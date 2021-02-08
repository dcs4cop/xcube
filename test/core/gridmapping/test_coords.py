import unittest

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)


# noinspection PyMethodMayBeStatic
class Coords1DGridMappingTest(unittest.TestCase):

    def test_1d_j_axis_down(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(np.linspace(1.5, 8.5, 8), dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((8, 10), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((1, -5, 9, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(False, gm.is_lon_360)

    def test_1d_j_axis_up(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(np.linspace(1.5, 8.5, 8), dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(-4.5, 4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((8, 10), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((1, -5, 9, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual(False, gm.is_lon_360)

    def test_1d_lon_360(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(np.linspace(177.5, 184.5, 8), dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((8, 10), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((177, -5, 185, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(True, gm.is_lon_360)

    def test_1d_anti_meridian(self):
        lon = np.linspace(177.5, 184.5, 8)
        lon = np.where(lon > 180, lon - 360, lon)
        gm = GridMapping.from_coords(x_coords=xr.DataArray(lon, dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((8, 10), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((177, -5, 185, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(True, gm.is_lon_360)

    def test_1d_tiles_given(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(np.linspace(177.5, 184.5, 8), dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS,
                                     tile_size=(5, 3))
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((5, 3), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((177, -5, 185, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(True, gm.is_lon_360)

    def test_1d_tiles_from_coords_chunks(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(da.linspace(177.5, 184.5, 8, chunks=4), dims='lon'),
                                     y_coords=xr.DataArray(da.linspace(4.5, -4.5, 10, chunks=5), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((4, 5), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((177, -5, 185, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(True, gm.is_lon_360)

    def test_1d_x_irregular(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.49, 6.5, 7.5, 8.5], dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        self.assertEqual((8, 10), gm.size)
        self.assertEqual((8, 10), gm.tile_size)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual((1, -5, 9, 5), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual(False, gm.is_lon_360)

    def test_1d_xy_coords(self):
        gm = GridMapping.from_coords(x_coords=xr.DataArray(np.linspace(1.5, 8.5, 8), dims='lon'),
                                     y_coords=xr.DataArray(np.linspace(4.5, -4.5, 10), dims='lat'),
                                     crs=GEO_CRS)
        xy_coords = gm.xy_coords
        self.assertIsInstance(xy_coords, xr.DataArray)
        self.assertIs(xy_coords, gm.xy_coords)
        self.assertEqual(('coord', 'lat', 'lon'), xy_coords.dims)
        self.assertEqual((2, 10, 8), xy_coords.shape)
        self.assertEqual(('lon', 'lat'), gm.xy_var_names)
        self.assertEqual(('lon', 'lat'), gm.xy_dim_names)


class Coords2DGridMappingTest(unittest.TestCase):

    def test_2d(self):
        gm = GridMapping.from_coords(
            x_coords=xr.DataArray([
                [10.0, 10.1, 10.2, 10.3],
                [10.1, 10.2, 10.3, 10.4],
                [10.2, 10.3, 10.4, 10.5],
            ], dims=('lat', 'lon')),
            y_coords=xr.DataArray([
                [52.0, 52.2, 52.4, 52.6],
                [52.2, 52.4, 52.6, 52.8],
                [52.4, 52.6, 52.8, 53.0],
            ], dims=('lat', 'lon')),
            crs=GEO_CRS)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertEqual((0.2, 0.2), gm.xy_res)
        self.assertEqual((9.9, 51.9, 10.6, 53.1), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual(False, gm.is_lon_360)

    def test_2d_tile_size_from_chunks(self):
        gm = GridMapping.from_coords(
            x_coords=xr.DataArray(da.array([
                [10.0, 10.1, 10.2, 10.3],
                [10.1, 10.2, 10.3, 10.4],
                [10.2, 10.3, 10.4, 10.5],
            ]).rechunk((2, 3)), dims=('lat', 'lon')),
            y_coords=xr.DataArray(da.array([
                [52.0, 52.2, 52.4, 52.6],
                [52.2, 52.4, 52.6, 52.8],
                [52.4, 52.6, 52.8, 53.0],
            ]).rechunk((2, 3)), dims=('lat', 'lon')),
            crs=GEO_CRS)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((3, 2), gm.tile_size)

    def test_2d_regular(self):
        gm = GridMapping.from_coords(
            x_coords=xr.DataArray([
                [10.2, 10.3, 10.4, 10.5],
                [10.2, 10.3, 10.4, 10.5],
                [10.2, 10.3, 10.4, 10.5],
            ], dims=('lat', 'lon')),
            y_coords=xr.DataArray([
                [52.4, 52.4, 52.4, 52.4],
                [52.6, 52.6, 52.6, 52.6],
                [52.8, 52.8, 52.8, 52.8],
            ], dims=('lat', 'lon')),
            crs=GEO_CRS)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertAlmostEqual(0.1, gm.x_res)
        self.assertAlmostEqual(0.2, gm.y_res)
        self.assertAlmostEqual(10.15, gm.x_min)
        self.assertAlmostEqual(52.3, gm.y_min)
        self.assertAlmostEqual(10.55, gm.x_max)
        self.assertAlmostEqual(52.9, gm.y_max)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual(False, gm.is_lon_360)

    def test_2d_anti_meridian(self):
        gm = GridMapping.from_coords(
            x_coords=xr.DataArray([
                [+177.5, +178.5, +179.5, -179.5],
                [+178.5, +179.5, -179.5, -178.5],
                [+179.5, -179.5, -178.5, -177.5],
            ], dims=('lat', 'lon')),
            y_coords=xr.DataArray([
                [52.4, 52.4, 52.4, 52.4],
                [52.6, 52.6, 52.6, 52.6],
                [52.8, 52.8, 52.8, 52.8],
            ], dims=('lat', 'lon')),
            crs=GEO_CRS)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertAlmostEqual(0.53, gm.x_res)
        self.assertAlmostEqual(0.53, gm.y_res)
        self.assertEqual((177.235, 52.135, 182.765, 53.065), gm.xy_bbox)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual(True, gm.is_lon_360)

    def test_to_regular(self):
        lon = xr.DataArray([[1.0, 6.0],
                            [0.0, 2.0]], dims=('y', 'x'))
        lat = xr.DataArray([[56.0, 53.0],
                            [52.0, 50.0]], dims=('y', 'x'))

        gm_irr = GridMapping.from_coords(lon, lat, GEO_CRS)
        gm_reg_actual = gm_irr.to_regular()
        gm_reg_expected = GridMapping.regular(size=(4, 4),
                                              xy_min=(-1.575, 48.425),
                                              xy_res=3.15,
                                              crs=GEO_CRS)
        self.assertEqual(gm_reg_expected.size, gm_reg_actual.size)
        self.assertEqual(gm_reg_expected.tile_size, gm_reg_actual.tile_size)
        self.assertEqual(gm_reg_expected.xy_res, gm_reg_actual.xy_res)
        self.assertEqual(gm_reg_expected.xy_bbox, gm_reg_actual.xy_bbox)
        self.assertEqual(gm_reg_expected.crs, gm_reg_actual.crs)

    def test_2d_xy_coords(self):
        gm = GridMapping.from_coords(
            x_coords=xr.DataArray([
                [10.0, 10.1, 10.2, 10.3],
                [10.1, 10.2, 10.3, 10.4],
                [10.2, 10.3, 10.4, 10.5],
            ], dims=('lat', 'lon')),
            y_coords=xr.DataArray([
                [52.0, 52.2, 52.4, 52.6],
                [52.2, 52.4, 52.6, 52.8],
                [52.4, 52.6, 52.8, 53.0],
            ], dims=('lat', 'lon')),
            crs=GEO_CRS)
        xy_coords = gm.xy_coords
        self.assertIsInstance(xy_coords, xr.DataArray)
        self.assertIs(xy_coords, gm.xy_coords)
        self.assertEqual(('coord', 'lat', 'lon'), xy_coords.dims)
        self.assertEqual((2, 3, 4), xy_coords.shape)
        self.assertEqual(('lon', 'lat'), gm.xy_var_names)
        self.assertEqual(('lon', 'lat'), gm.xy_dim_names)
