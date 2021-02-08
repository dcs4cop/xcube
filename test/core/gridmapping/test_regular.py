import unittest

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.regular import RegularGridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class RegularGridMappingTest(unittest.TestCase):

    def test_default_props(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(10, gm.x_min)
        self.assertEqual(53, gm.y_min)
        self.assertEqual((0.01, 0.01), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)

    def test_invalid_y(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.regular((1000, 1000), (10, -90.5), 0.01, CRS_WGS84)
        self.assertEqual('invalid y_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            GridMapping.regular((1000, 1000), (10, 53), 0.1, CRS_WGS84)
        self.assertEqual('invalid size, y_min combination', f'{cm.exception}')

    def test_xy_bbox(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((10, 53, 20, 63), gm.xy_bbox)
        self.assertEqual(False, gm.is_lon_360)

    def test_xy_bbox_anti_meridian(self):
        gm = GridMapping.regular((2000, 1000), (174.0, -30.0), 0.005, CRS_WGS84)
        self.assertEqual((174.0, -30.0, 184.0, -25.0), gm.xy_bbox)
        self.assertEqual(True, gm.is_lon_360)

    def test_derive(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)
        derived_gm = gm.derive(tile_size=500, is_j_axis_up=True)
        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, RegularGridMapping)
        self.assertEqual((1000, 1000), derived_gm.size)
        self.assertEqual((500, 500), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)

    def test_xy_coords(self):
        gm = GridMapping.regular((8, 4), (10, 53), 0.1, CRS_WGS84).derive(tile_size=(4, 2))
        xy_coords = gm.xy_coords
        self.assertIsInstance(xy_coords, xr.DataArray)
        self.assertIs(gm.xy_coords, xy_coords)
        self.assertEqual(('coord', 'lat', 'lon'), xy_coords.dims)
        self.assertEqual((2, 4, 8), xy_coords.shape)
        self.assertEqual(((2,), (2, 2), (4, 4)), xy_coords.chunks)
        np.testing.assert_almost_equal(
            np.array([
                [10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75],
                [10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75],
                [10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75],
                [10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75]
            ]),
            xy_coords.values[0]
        )
        np.testing.assert_almost_equal(
            np.array([
                [53.35, 53.35, 53.35, 53.35, 53.35, 53.35, 53.35, 53.35],
                [53.25, 53.25, 53.25, 53.25, 53.25, 53.25, 53.25, 53.25],
                [53.15, 53.15, 53.15, 53.15, 53.15, 53.15, 53.15, 53.15],
                [53.05, 53.05, 53.05, 53.05, 53.05, 53.05, 53.05, 53.05]
            ]),
            xy_coords.values[1]
        )

    def test_xy_names(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, GEO_CRS).derive(tile_size=500)
        self.assertEqual(('lon', 'lat'), gm.xy_var_names)
        self.assertEqual(('lon', 'lat'), gm.xy_dim_names)
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, NOT_A_GEO_CRS).derive(tile_size=500)
        self.assertEqual(('x', 'y'), gm.xy_var_names)
        self.assertEqual(('x', 'y'), gm.xy_dim_names)

    def test_ij_bboxes(self):
        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(gm.ij_bboxes,
                                       np.array([[0, 0, 2000, 1000]],
                                                dtype=np.int64))

        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS).derive(tile_size=500)
        np.testing.assert_almost_equal(gm.ij_bboxes,
                                       np.array([
                                           [0, 0, 500, 500],
                                           [500, 0, 1000, 500],
                                           [1000, 0, 1500, 500],
                                           [1500, 0, 2000, 500],
                                           [0, 500, 500, 1000],
                                           [500, 500, 1000, 1000],
                                           [1000, 500, 1500, 1000],
                                           [1500, 500, 2000, 1000]
                                       ], dtype=np.int64))

    def test_xy_bboxes(self):
        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(gm.xy_bboxes,
                                       np.array([[10., 20., 210., 120.]],
                                                dtype=np.float64))

        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS).derive(tile_size=500)
        np.testing.assert_almost_equal(gm.xy_bboxes,
                                       np.array([
                                           [10., 70, 60, 120.],
                                           [60., 70, 110, 120.],
                                           [110., 70, 160, 120.],
                                           [160., 70, 210, 120.],
                                           [10., 20, 60, 70.],
                                           [60., 20, 110, 70.],
                                           [110., 20, 160, 70.],
                                           [160., 20, 210, 70.]
                                       ], dtype=np.float64))

    def test_xy_bboxes_is_j_axis_up(self):
        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS).derive(is_j_axis_up=True)
        np.testing.assert_almost_equal(gm.xy_bboxes,
                                       np.array([[10., 20., 210., 120.]],
                                                dtype=np.float64))

        gm = GridMapping.regular(size=(2000, 1000),
                                 xy_min=(10.0, 20.0),
                                 xy_res=0.1,
                                 crs=NOT_A_GEO_CRS, ).derive(tile_size=500,
                                                             is_j_axis_up=True)
        np.testing.assert_almost_equal(gm.xy_bboxes,
                                       np.array([
                                           [10., 20., 60., 70.],
                                           [60., 20., 110., 70.],
                                           [110., 20., 160., 70.],
                                           [160., 20., 210., 70.],
                                           [10., 70., 60., 120.],
                                           [60., 70., 110., 120.],
                                           [110., 70., 160., 120.],
                                           [160., 70., 210., 120.]
                                       ], dtype=np.float64))

    def test_to_coords(self):
        gm = GridMapping.regular(size=(10, 6),
                                 xy_min=(-2600.0, 1200.0),
                                 xy_res=10.0,
                                 crs=NOT_A_GEO_CRS)

        cv = gm.to_coords(xy_var_names=('x', 'y'))
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1255., 1205.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1260., 1250.),
                                    (1210., 1200.),
                                ))

    def test_coord_vars_j_axis_up(self):
        gm = GridMapping.regular(size=(10, 6),
                                 xy_min=(-2600.0, 1200.0),
                                 xy_res=10.0,
                                 crs=NOT_A_GEO_CRS).derive(is_j_axis_up=True)

        cv = gm.to_coords(xy_var_names=('x', 'y'))
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1205., 1255.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1200., 1210.),
                                    (1250., 1260.),
                                ))

    def test_coord_vars_antimeridian(self):
        gm = GridMapping.regular(size=(10, 10),
                                 xy_min=(172.0, 53.0),
                                 xy_res=2.0,
                                 crs=GEO_CRS)

        cv = gm.to_coords(xy_var_names=('lon', 'lat'))
        self._assert_coord_vars(cv,
                                (10, 10),
                                ('lon', 'lat'),
                                (173.0, -169.0),
                                (72.0, 54.0),
                                ('lon_bnds', 'lat_bnds'),
                                (
                                    (172., 174.),
                                    (-170., -168.),
                                ),
                                (
                                    (73., 71.),
                                    (55., 53.),
                                ))

    def _assert_coord_vars(self,
                           cv,
                           size,
                           xy_names,
                           x_values,
                           y_values,
                           xy_bnds_names,
                           x_bnds_values,
                           y_bnds_values):
        self.assertIsNotNone(cv)
        self.assertIn(xy_names[0], cv)
        self.assertIn(xy_names[1], cv)
        self.assertIn(xy_bnds_names[0], cv)
        self.assertIn(xy_bnds_names[1], cv)

        x = cv[xy_names[0]]
        self.assertEqual((size[0],), x.shape)
        np.testing.assert_almost_equal(x.values[0], np.array(x_values[0]))
        np.testing.assert_almost_equal(x.values[-1], np.array(x_values[-1]))

        y = cv[xy_names[1]]
        self.assertEqual((size[1],), y.shape)
        np.testing.assert_almost_equal(y.values[0], np.array(y_values[0]))
        np.testing.assert_almost_equal(y.values[-1], np.array(y_values[-1]))

        x_bnds = cv[xy_bnds_names[0]]
        self.assertEqual((size[0], 2), x_bnds.shape)
        np.testing.assert_almost_equal(x_bnds.values[0], np.array(x_bnds_values[0]))
        np.testing.assert_almost_equal(x_bnds.values[-1], np.array(x_bnds_values[-1]))

        y_bnds = cv[xy_bnds_names[1]]
        self.assertEqual((size[1], 2), y_bnds.shape)
        np.testing.assert_almost_equal(y_bnds.values[0], y_bnds_values[0])
        np.testing.assert_almost_equal(y_bnds.values[-1], y_bnds_values[-1])
