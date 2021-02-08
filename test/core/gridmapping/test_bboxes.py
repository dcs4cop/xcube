import unittest

import dask.array as da
import numpy as np
import xarray as xr

from xcube.core.gridmapping.bboxes import compute_ij_bboxes
from xcube.core.gridmapping.bboxes import compute_xy_bbox


class ComputeIJBBoxesTest(unittest.TestCase):

    def setUp(self) -> None:
        lon = xr.DataArray(np.linspace(10., 20., 11), dims='x')
        lat = xr.DataArray(np.linspace(50., 60., 11), dims='y')
        lat, lon = xr.broadcast(lat, lon)
        self.lon_values = lon.values
        self.lat_values = lat.values

    def test_all_included(self):
        xy_bboxes = np.array([[10., 50., 20., 60.]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 11, 11]], dtype=np.int64))

    def test_tiles(self):
        a0 = 0.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 6, 6],
                                                 [5, 0, 11, 6],
                                                 [0, 5, 6, 11],
                                                 [5, 5, 11, 11]], dtype=np.int64))

    def test_none_found(self):
        a0 = 11.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1]], dtype=np.int64))

    def test_with_border(self):
        xy_bboxes = np.array([[12.4, 51.6, 12.6, 51.7]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1]], dtype=np.int64))
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.5, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 2, 4, 3]], dtype=np.int64))
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 1.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 1, 4, 3]], dtype=np.int64))
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 2.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[1, 0, 5, 4]], dtype=np.int64))
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 2.0, 2, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 7, 6]], dtype=np.int64))


class ComputeXYBBoxTest(unittest.TestCase):
    data = [
        [
            [10, 11, 12, 13, 14],
            [11, 12, 13, 14, 15],
            [12, 13, 14, 15, 16],
            [13, 14, 15, 16, 17],
        ],
        [
            [50, 51, 52, 53, 54],
            [51, 52, 53, 54, 55],
            [52, 53, 54, 55, 56],
            [53, 54, 55, 56, 57],
        ],
    ]

    def test_numpy_array(self):
        xy_coords = np.array(self.data, dtype=np.float64)
        xy_bbox = compute_xy_bbox(xy_coords)
        self.assertEqual((10, 50, 17, 57), xy_bbox)

    def test_dask_array(self):
        xy_coords = da.array(self.data, dtype=np.float64).rechunk((2, 3, 3))
        xy_bbox = compute_xy_bbox(xy_coords)
        self.assertEqual((10, 50, 17, 57), xy_bbox)

    def test_many_nans(self):
        w = 2000
        h = 1000
        x = np.full(h * w, np.nan)
        y = np.full(h * w, np.nan)

        x[np.random.randint(0, w)] = 73.
        y[np.random.randint(0, h)] = 34.

        xy_coords = da.array([x.reshape((h, w)), y.reshape((h, w))], dtype=np.float64) \
            .rechunk((2, 512, 512))
        xy_bbox = compute_xy_bbox(xy_coords)

        self.assertEqual((73., 34., 73., 34.), xy_bbox)
