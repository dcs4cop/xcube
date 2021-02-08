import unittest

import numpy as np
import pyproj
import xarray as xr

from test.sampledata import SourceDatasetMixin
from xcube.core.gridmapping import GridMapping
# noinspection PyProtectedMember
from xcube.core.gridmapping.helpers import _to_affine

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


class TestGridMapping(GridMapping):
    def _new_xy_coords(self) -> xr.DataArray:
        return GridMapping.regular(size=self.size,
                                   tile_size=self.tile_size,
                                   is_j_axis_up=self.is_j_axis_up,
                                   xy_res=self.xy_res,
                                   xy_min=(self.xy_bbox[0], self.xy_bbox[1]),
                                   crs=self.crs).xy_coords


# noinspection PyMethodMayBeStatic
class GridMappingTest(SourceDatasetMixin, unittest.TestCase):
    _kwargs = dict(
        size=(720, 360),
        tile_size=(360, 180),
        xy_bbox=(-180.0, -90.0, 180.0, 90.0),
        xy_res=(360 / 720, 360 / 720),
        crs=GEO_CRS,
        xy_var_names=('x', 'y'),
        xy_dim_names=('x', 'y'),
        is_regular=True,
        is_lon_360=False,
        is_j_axis_up=False,
    )

    def kwargs(self, **kwargs):
        orig_kwargs = dict(self._kwargs)
        orig_kwargs.update(**kwargs)
        if 'xy_min' in orig_kwargs:
            # Replace xy_min by xy_bbox.
            # Using xy_min instead of xy_bbox makes it easier for us
            width, height = orig_kwargs['size']
            try:
                x_res, y_res = orig_kwargs['xy_res']
            except TypeError:
                x_res, y_res = 2 * (orig_kwargs['xy_res'],)
            x_min, y_min = orig_kwargs.pop('xy_min')
            x_max, y_max = x_min + x_res * width, y_min + y_res * height
            orig_kwargs['xy_bbox'] = x_min, y_min, x_max, y_max
        return orig_kwargs

    def test_valid(self):
        gm = TestGridMapping(**self.kwargs())
        self.assertEqual((720, 360), gm.size)
        self.assertEqual(720, gm.width)
        self.assertEqual(360, gm.height)
        self.assertEqual(True, gm.is_tiled)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(360, gm.tile_width)
        self.assertEqual(180, gm.tile_height)
        self.assertEqual((0, 0, 720, 360), gm.ij_bbox)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0), gm.xy_bbox)
        self.assertEqual(-180.0, gm.x_min)
        self.assertEqual(-90.0, gm.y_min)
        self.assertEqual(180.0, gm.x_max)
        self.assertEqual(90.0, gm.y_max)
        self.assertEqual((0.5, 0.5), gm.xy_res)
        self.assertEqual(0.5, gm.x_res)
        self.assertEqual(0.5, gm.y_res)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)

        self.assertIsInstance(gm.xy_coords, xr.DataArray)
        np.testing.assert_equal(np.array([[0, 0, 360, 180],
                                          [360, 0, 720, 180],
                                          [0, 180, 360, 360],
                                          [360, 180, 720, 360]]), gm.ij_bboxes)
        np.testing.assert_equal(np.array([[-180., 0., 0., 90.],
                                          [0., 0., 180., 90.],
                                          [-180., -90., 0., 0.],
                                          [0., -90., 180., 0.]]), gm.xy_bboxes)

    def test_invalids(self):
        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=(360, 1)))
        self.assertEqual('invalid size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=(360,)))
        self.assertEqual('not enough values to unpack (expected 2, got 1)', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=None))
        self.assertEqual('size must be an int or a sequence of two ints', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(tile_size=0))
        self.assertEqual('invalid tile_size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(xy_res=-0.1))
        self.assertEqual('invalid xy_res', f'{cm.exception}')

    def test_scalars(self):
        gm = TestGridMapping(**self.kwargs(size=360, tile_size=180, xy_res=0.1))
        self.assertEqual((360, 360), gm.size)
        self.assertEqual((180, 180), gm.tile_size)
        self.assertEqual((0.1, 0.1), gm.xy_res)

    def test_not_tiled(self):
        gm = TestGridMapping(**self.kwargs(tile_size=None))
        self.assertEqual((720, 360), gm.tile_size)
        self.assertEqual(False, gm.is_tiled)

    def test_ij_to_xy_transform(self):
        image_geom = TestGridMapping(**self.kwargs(size=(1200, 1200),
                                                   xy_min=(0, 0),
                                                   xy_res=1,
                                                   crs=NOT_A_GEO_CRS))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((0, 0), i2crs, (0, 1200))
        self.assertMatrixPoint((1024, 0), i2crs, (1024, 1200))
        self.assertMatrixPoint((0, 1024), i2crs, (0, 1200 - 1024))
        self.assertMatrixPoint((1024, 1024), i2crs, (1024, 1200 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1200)), i2crs)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_min=(-180, -90),
                                                   xy_res=0.25))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, 90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, -90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, -0.25, 90.0)), i2crs)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_min=(-180, -90),
                                                   xy_res=0.25,
                                                   is_j_axis_up=True))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, -90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, 90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, 0.25, -90.0)), i2crs)

    def test_xy_to_ij_transform(self):
        image_geom = TestGridMapping(**self.kwargs(size=(1200, 1200),
                                                   xy_min=(0, 0),
                                                   xy_res=1,
                                                   crs=NOT_A_GEO_CRS))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (0, 1200))
        self.assertMatrixPoint((1024, 0), crs2i, (1024, 1200))
        self.assertMatrixPoint((0, 1024), crs2i, (0, 1200 - 1024))
        self.assertMatrixPoint((1024, 1024), crs2i, (1024, 1200 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1200)), crs2i)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_res=0.25))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 720), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 0), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, -4.0, 360.0)), crs2i)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_res=0.25,
                                                   is_j_axis_up=True))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 720), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, 4.0, 360.0)), crs2i)

    def test_ij_transform_to_and_from(self):
        gm1 = TestGridMapping(**self.kwargs(size=(1440, 720),
                                            xy_res=0.25,
                                            is_j_axis_up=True))
        gm2 = TestGridMapping(**self.kwargs(size=(1000, 1000),
                                            xy_min=(10, 50),
                                            xy_res=0.025,
                                            is_j_axis_up=True))
        self.assertEqual(((10.0, 0.0, -7600.0),
                          (0.0, 10.0, -5600.0)),
                         gm1.ij_transform_to(gm2))
        self.assertEqual(((10.0, 0.0, -7600.0),
                          (0.0, 10.0, -5600.0)),
                         gm2.ij_transform_from(gm1))
        self.assertEqual(((0.1, 0.0, 760.0),
                          (0.0, 0.1, 560.0)),
                         gm2.ij_transform_to(gm1))
        self.assertEqual(((0.1, 0.0, 760.0),
                          (0.0, 0.1, 560.0)),
                         gm1.ij_transform_from(gm2))

    def assertMatrixPoint(self, expected_point, matrix, point):
        affine = _to_affine(matrix)
        actual_point = affine * point
        self.assertAlmostEqual(expected_point[0], actual_point[0])
        self.assertAlmostEqual(expected_point[1], actual_point[1])
        return actual_point

    def test_derive(self):
        gm = TestGridMapping(**self.kwargs())
        self.assertEqual((720, 360), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)

        # force creating of xy_coords array and save value
        xy_coords = gm.xy_coords

        derived_gm = gm.derive(tile_size=270,
                               is_j_axis_up=True,
                               xy_var_names=('u', 'v'),
                               xy_dim_names=('i', 'j'))

        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, TestGridMapping)
        self.assertEqual((720, 360), derived_gm.size)
        self.assertEqual((270, 270), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)
        self.assertEqual(('u', 'v'), derived_gm.xy_var_names)
        self.assertEqual(('i', 'j'), derived_gm.xy_dim_names)

        derived_xy_coords = derived_gm.xy_coords
        self.assertIsNot(xy_coords, derived_xy_coords)
        self.assertEqual(((2,), (270, 90), (270, 270, 180)), derived_xy_coords.chunks)

    def test_ij_bbox_from_xy_bbox(self):
        gm = TestGridMapping(**self.kwargs())

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 180, 90))
        self.assertEqual((0, 0, 720, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 0, 0))
        self.assertEqual((0, 180, 360, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((0, 0, 180, 90))
        self.assertEqual((360, 0, 720, 180), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 0, 0), ij_border=1)
        self.assertEqual((0, 179, 361, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((0, 0, 180, 90), ij_border=1)
        self.assertEqual((359, 0, 720, 181), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-190, -100, -170, -80), ij_border=1)
        self.assertEqual((0, 339, 21, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-190, -100, -180, -90), ij_border=1)
        self.assertEqual((-1, -1, -1, -1), ij_bbox)

    def test_ij_bboxes_from_xy_bboxes(self):
        gm = TestGridMapping(**self.kwargs())

        ij_bboxes = gm.ij_bboxes_from_xy_bboxes(
            xy_bboxes=np.array([
                [-180, -90, 180, 90],
                [-180, -90, 0, 0],
                [0, 0, 180, 90],
                [-180, -90, 0, 0],
                [0, 0, 180, 90],
                [-190, -100, -170, -80],
                [-190, -100, -180, -90]
            ], dtype=np.float32))

        np.testing.assert_equal(
            ij_bboxes,
            np.array([[0, 0, 720, 360],
                      [0, 180, 360, 360],
                      [360, 0, 720, 180],
                      [0, 180, 360, 360],
                      [360, 0, 720, 180],
                      [0, 340, 20, 360],
                      [-1, -1, -1, -1]], dtype=np.int64))
