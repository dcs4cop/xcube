import os.path
import unittest

import numpy as np
import pyproj
import xarray as xr

import xcube.core.new
from test.sampledata import create_s2plus_dataset
from xcube.core.gridmapping import GridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class DatasetGridMappingTest(unittest.TestCase):
    def test_from_regular_cube(self):
        dataset = xcube.core.new.new_cube(variables=dict(rad=0.5))
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((360, 180), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual((2, 180, 360), gm.xy_coords.shape)
        self.assertEqual(('coord', 'lat', 'lon'), gm.xy_coords.dims)

    def test_from_non_regular_cube(self):
        lon = np.array([[8, 9.3, 10.6, 11.9],
                        [8, 9.2, 10.4, 11.6],
                        [8, 9.1, 10.2, 11.3]], dtype=np.float32)
        lat = np.array([[56, 56.1, 56.2, 56.3],
                        [55, 55.2, 55.4, 55.6],
                        [54, 54.3, 54.6, 54.9]], dtype=np.float32)
        rad = np.random.random(3 * 4).reshape((3, 4))
        dims = ('y', 'x')
        dataset = xr.Dataset(dict(lon=xr.DataArray(lon, dims=dims),
                                  lat=xr.DataArray(lat, dims=dims),
                                  rad=xr.DataArray(rad, dims=dims)))
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(None, gm.is_j_axis_up)
        self.assertEqual((2, 3, 4), gm.xy_coords.shape)
        self.assertEqual(('coord', 'y', 'x'), gm.xy_coords.dims)
        self.assertEqual((0.93, 0.93), gm.xy_res)

    def test_from_real_olci(self):
        olci_l2_path = os.path.join(os.path.dirname(__file__),
                                    '..', '..', '..',
                                    'examples', 'notebooks', 'S3-OLCI-L2A.zarr.zip')

        dataset = xr.open_zarr(olci_l2_path)
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((1189, 1890), gm.size)
        self.assertEqual((512, 512), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((0.00098, 0.00098), gm.xy_res)
        # self.assertAlmostEqual(12.693771178309552, gm.x_min)
        # self.assertAlmostEqual(20.005413821690446, gm.x_max)
        # self.assertAlmostEqual(55.19965017830955, gm.y_min)
        # self.assertAlmostEqual(60.63871982169044, gm.y_max)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(None, gm.is_j_axis_up)
        self.assertEqual((2, 1890, 1189), gm.xy_coords.shape)
        self.assertEqual(('coord', 'y', 'x'), gm.xy_coords.dims)

        gm = gm.to_regular()
        self.assertEqual((7462, 5551), gm.size)

    def test_from_sentinel_2(self):
        dataset = create_s2plus_dataset()

        gm = GridMapping.from_dataset(dataset)
        # Should pick the projected one which is regular
        self.assertEqual("Projected CRS", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=True)
        # Should pick the projected one which is regular
        self.assertEqual("Projected CRS", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=False)
        # Should pick the geographic one which is irregular
        self.assertEqual('Geographic 2D CRS', gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS)
        # Should pick the geographic one which is irregular
        self.assertEqual('Geographic 2D CRS', gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS, prefer_is_regular=True)
        # Should pick the geographic one which is irregular
        self.assertEqual('Geographic 2D CRS', gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

    def test_xy_names(self):
        dataset = create_s2plus_dataset()
        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(dataset, xy_var_names=('lons', 'lats'))
        self.assertEqual('coordinate variables "lons" or "lats" not found in dataset', f'{cm.exception}')

        dataset = create_s2plus_dataset()
        with self.assertRaises(NotImplementedError) as cm:
            GridMapping.from_dataset(dataset, xy_var_names=('lon', 'lat'))
        self.assertEqual('xy_var_names not yet supported', f'{cm.exception}')

    def test_no_grid_mapping_found(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(xr.Dataset())
        self.assertEqual('cannot find any grid mapping in dataset', f'{cm.exception}')
