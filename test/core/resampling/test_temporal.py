import unittest

import numpy as np
import pandas as pd

from test.sampledata import new_test_dataset
from xcube.core.chunk import chunk_dataset
from xcube.core.resampling import resample_in_time
from xcube.core.schema import CubeSchema


class ResampleInTimeTest(unittest.TestCase):

    def setUp(self) -> None:
        num_times = 30

        time = []
        periods = ['1D', '1D', '3D', '4D', '2D']
        t = pd.to_datetime('2017-07-01T10:30:15Z', utc=True)
        for i in range(num_times):
            time.append(t.isoformat())
            t += pd.to_timedelta(periods[i % len(periods)])

        temperature, precipitation = zip(*[(272 + 0.1 * i, 120 - 0.2 * i) for i in range(num_times)])

        input_cube = new_test_dataset(time, temperature=temperature, precipitation=precipitation)
        input_cube = chunk_dataset(input_cube, chunk_sizes=dict(time=1, lat=90, lon=180))
        self.input_cube = input_cube

    def test_resample_in_time_min_max(self):
        resampled_cube = resample_in_time(self.input_cube, '2W', ['min', 'max'])
        self.assertIsNot(resampled_cube, self.input_cube)
        self.assertIn('time', resampled_cube)
        self.assertIn('temperature_min', resampled_cube)
        self.assertIn('temperature_max', resampled_cube)
        self.assertIn('precipitation_min', resampled_cube)
        self.assertIn('precipitation_max', resampled_cube)
        self.assertEqual(('time',), resampled_cube.time.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.temperature_min.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.temperature_max.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.precipitation_min.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.precipitation_max.dims)
        self.assertEqual((6,), resampled_cube.time.shape)
        self.assertEqual((6, 180, 360), resampled_cube.temperature_min.shape)
        self.assertEqual((6, 180, 360), resampled_cube.temperature_max.shape)
        self.assertEqual((6, 180, 360), resampled_cube.precipitation_min.shape)
        self.assertEqual((6, 180, 360), resampled_cube.precipitation_max.shape)
        np.testing.assert_equal(resampled_cube.time.values,
                                np.array(
                                    ['2017-06-25T00:00:00', '2017-07-09T00:00:00',
                                     '2017-07-23T00:00:00', '2017-08-06T00:00:00',
                                     '2017-08-20T00:00:00', '2017-09-03T00:00:00'], dtype=np.datetime64))
        np.testing.assert_allclose(resampled_cube.temperature_min.values[..., 0, 0],
                                   np.array([272.0, 272.4, 273.0, 273.8, 274.4, 274.9]))
        np.testing.assert_allclose(resampled_cube.temperature_max.values[..., 0, 0],
                                   np.array([272.3, 272.9, 273.7, 274.3, 274.8, 274.9]))
        np.testing.assert_allclose(resampled_cube.precipitation_min.values[..., 0, 0],
                                   np.array([119.4, 118.2, 116.6, 115.4, 114.4, 114.2]))
        np.testing.assert_allclose(resampled_cube.precipitation_max.values[..., 0, 0],
                                   np.array([120.0, 119.2, 118.0, 116.4, 115.2, 114.2]))

        schema = CubeSchema.new(resampled_cube)
        self.assertEqual(3, schema.ndim)
        self.assertEqual(('time', 'lat', 'lon'), schema.dims)
        self.assertEqual((6, 180, 360), schema.shape)
        self.assertEqual((1, 90, 180), schema.chunks)

    def test_resample_in_time_p90_dask(self):
        # "percentile_<p>" can currently only be used with numpy rather than chunked dask arrays:

        # TypeError raised on Windows, ValueError on Linux
        with self.assertRaises(Exception):
            # noinspection PyUnusedLocal
            resampled_cube = resample_in_time(self.input_cube, '2W', 'percentile_90')

    # TODO (forman): the call to resample_in_time() takes forever,
    #                this is not xcube, but may be an issue in dask 0.14 or dask 2.8.
    # def test_resample_in_time_p90_numpy(self):
    #     # "percentile_<p>" can currently only be used with numpy, so compute() first:
    #     input_cube = self.input_cube.compute()
    #     resampled_cube = resample_in_time(input_cube, '2W', 'percentile_90')
    #     self.assertIsNot(resampled_cube, self.input_cube)
    #     self.assertIn('time', resampled_cube)
    #     self.assertIn('temperature_p90', resampled_cube)
    #     self.assertIn('precipitation_p90', resampled_cube)
    #     self.assertEqual(('time',), resampled_cube.time.dims)
    #     self.assertEqual(('time', 'lat', 'lon'), resampled_cube.temperature_p90.dims)
    #     self.assertEqual(('time', 'lat', 'lon'), resampled_cube.precipitation_p90.dims)
    #     self.assertEqual((6,), resampled_cube.time.shape)
    #     self.assertEqual((6, 180, 360), resampled_cube.temperature_p90.shape)
    #     self.assertEqual((6, 180, 360), resampled_cube.precipitation_p90.shape)
    #     np.testing.assert_equal(resampled_cube.time.values,
    #                             np.array(
    #                                 ['2017-06-25T00:00:00Z', '2017-07-09T00:00:00Z',
    #                                  '2017-07-23T00:00:00Z', '2017-08-06T00:00:00Z',
    #                                  '2017-08-20T00:00:00Z', '2017-09-03T00:00:00Z'], dtype=np.datetime64))
    #     np.testing.assert_allclose(resampled_cube.temperature_p90.values[..., 0, 0],
    #                                np.array([272.3, 272.9, 273.7, 274.3, 274.8, 274.9]))
    #     np.testing.assert_allclose(resampled_cube.precipitation_p90.values[..., 0, 0],
    #                                np.array([120.0, 119.2, 118.0, 116.4, 115.2, 114.2]))
    #
    #     schema = CubeSchema.new(resampled_cube)
    #     self.assertEqual(3, schema.ndim)
    #     self.assertEqual(('time', 'lat', 'lon'), schema.dims)
    #     self.assertEqual((6, 180, 360), schema.shape)
    #     self.assertEqual((1, 90, 180), schema.chunks)

    def test_resample_in_time_with_time_chunk_size(self):
        resampled_cube = resample_in_time(self.input_cube, '2D', ['min', 'max'], time_chunk_size=5)
        schema = CubeSchema.new(resampled_cube)
        self.assertEqual(3, schema.ndim)
        self.assertEqual(('time', 'lat', 'lon'), schema.dims)
        self.assertEqual((33, 180, 360), schema.shape)
        self.assertEqual((5, 90, 180), schema.chunks)

    def test_resample_f_all(self):
        resampled_cube = resample_in_time(self.input_cube, 'all', ['min', 'max'])
        self.assertIsNot(resampled_cube, self.input_cube)
        self.assertIn('time', resampled_cube)
        self.assertIn('temperature_min', resampled_cube)
        self.assertIn('temperature_max', resampled_cube)
        self.assertIn('precipitation_min', resampled_cube)
        self.assertIn('precipitation_max', resampled_cube)
        self.assertEqual(('time',), resampled_cube.time.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.temperature_min.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.temperature_max.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.precipitation_min.dims)
        self.assertEqual(('time', 'lat', 'lon'), resampled_cube.precipitation_max.dims)
        self.assertEqual((1,), resampled_cube.time.shape)
        self.assertEqual((1, 180, 360), resampled_cube.temperature_min.shape)
        self.assertEqual((1, 180, 360), resampled_cube.temperature_max.shape)
        self.assertEqual((1, 180, 360), resampled_cube.precipitation_min.shape)
        self.assertEqual((1, 180, 360), resampled_cube.precipitation_max.shape)
        np.testing.assert_allclose(resampled_cube.temperature_min.values[..., 0, 0],
                                   np.array([272.0]))
        np.testing.assert_allclose(resampled_cube.temperature_max.values[..., 0, 0],
                                   np.array([274.9]))
        np.testing.assert_allclose(resampled_cube.precipitation_min.values[..., 0, 0],
                                   np.array([114.2]))
        np.testing.assert_allclose(resampled_cube.precipitation_max.values[..., 0, 0],
                                   np.array([120.0]))

        schema = CubeSchema.new(resampled_cube)
        self.assertEqual(3, schema.ndim)
        self.assertEqual(('time', 'lat', 'lon'), schema.dims)
        self.assertEqual((1, 180, 360), schema.shape)
