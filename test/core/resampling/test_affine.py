import unittest

import numpy as np
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import affine_transform_dataset

nan = np.nan

res = 0.1

source_ds = xr.Dataset(
    data_vars=dict(
        refl=xr.DataArray(
            np.array([
                [0, 1, 0, 2, 0, 3, 0, 4],
                [2, 0, 3, 0, 4, 0, 1, 0],
                [0, 4, 0, 1, 0, 2, 0, 3],
                [1, 0, 2, 0, 3, 0, 4, 0],
                [0, 3, 0, 4, 0, 1, 0, 2],
                [4, 0, 1, 0, 2, 0, 3, 0],
            ], dtype=np.float64),
            dims=('lat', 'lon'))
    ),
    coords=dict(
        lon=xr.DataArray(50.0 + res * np.arange(0, 8) + 0.5 * res, dims='lon'),
        lat=xr.DataArray(10.6 - res * np.arange(0, 6) - 0.5 * res, dims='lat')
    )
)

source_gm = GridMapping.from_dataset(source_ds)


class AffineTransformDatasetTest(unittest.TestCase):
    def test_subset(self):
        target_gm = GridMapping.regular((3, 3), (50.0, 10.0), res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 1],
            ]))

        target_gm = GridMapping.regular((3, 3), (50.1, 10.1), res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [4, 0, 1],
                [0, 2, 0],
                [3, 0, 4],
            ]))

        target_gm = GridMapping.regular((3, 3), (50.05, 10.05), res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [1.25, 1.5, 0.75],
                [1., 1.25, 1.5],
                [1.75, 1., 1.25]
            ]))

    def test_downscale_x2(self):
        target_gm = GridMapping.regular((8, 6), (50, 10), 2 * res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [0.75, 1.25, 1.75, 1.25, nan, nan, nan, nan],
                [1.25, 0.75, 1.25, 1.75, nan, nan, nan, nan],
                [1.75, 1.25, 0.75, 1.25, nan, nan, nan, nan]
            ]))

    def test_downscale_x2_and_shift(self):
        target_gm = GridMapping.regular((8, 6), (49.8, 9.8), 2 * res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, 0.75, 1.25, 1.75, 1.25, nan, nan, nan],
                [nan, 1.25, 0.75, 1.25, 1.75, nan, nan, nan],
                [nan, 1.75, 1.25, 0.75, 1.25, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan]
            ]))

    def test_upscale_x2(self):
        target_gm = GridMapping.regular((8, 6), (50, 10), res / 2, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 0.0, 1.5],
                [0.5, 1.0, 1.5, 1.25, 1.0, 1.5, 2.0, 1.75],
                [0.0, 1.5, 3.0, 1.5, 0.0, 2.0, 4.0, 2.0],
                [2.0, 1.75, 1.5, 1.0, 0.5, 1.25, 2.0, 1.5],
                [4.0, 2.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0],
                [nan, nan, nan, nan, nan, nan, nan, nan]
            ]))

    def test_upscale_x2_and_shift(self):
        target_gm = GridMapping.regular((8, 6), (49.9, 9.95), res / 2, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [nan, nan, 0.5, 1.0, 1.5, 1.25, 1.0, 1.5],
                [nan, nan, 0.0, 1.5, 3.0, 1.5, 0.0, 2.0],
                [nan, nan, 2.0, 1.75, 1.5, 1.0, 0.5, 1.25],
                [nan, nan, 4.0, 2.0, 0.0, 0.5, 1.0, 0.5],
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan]
            ]))

    def test_shift(self):
        target_gm = GridMapping.regular((8, 6), (50.2, 10.1), res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [nan, nan, nan, nan, nan, nan, nan, nan],
                [0.0, 2.0, 0.0, 3.0, 0.0, 4.0, nan, nan],
                [3.0, 0.0, 4.0, 0.0, 1.0, 0.0, nan, nan],
                [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, nan, nan],
                [2.0, 0.0, 3.0, 0.0, 4.0, 0.0, nan, nan],
                [0.0, 4.0, 0.0, 1.0, 0.0, 2.0, nan, nan]
            ]))

        target_gm = GridMapping.regular((8, 6), (49.8, 9.9), res, source_gm.crs)
        target_ds = affine_transform_dataset(source_ds, source_gm, target_gm)
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(set(target_ds.variables), set(source_ds.variables))
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([
                [nan, nan, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
                [nan, nan, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0],
                [nan, nan, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                [nan, nan, 0.0, 3.0, 0.0, 4.0, 0.0, 1.0],
                [nan, nan, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                [nan, nan, nan, nan, nan, nan, nan, nan]
            ]))
