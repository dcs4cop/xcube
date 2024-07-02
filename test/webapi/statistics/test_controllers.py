# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.webapi.statistics.controllers import compute_statistics
from .test_context import get_statistics_ctx


class StatisticsControllerTest(unittest.TestCase):
    def test_compute_statistics_for_point(self):
        lon = 1.768
        lat = 51.465
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()
        dataset = ctx.datasets_ctx.get_dataset("demo")
        expected_value = float(
            dataset["conc_tsm"]
            .sel(lon=lon, lat=lat, time=time, method="nearest")
            .values
        )

        result = compute_statistics(
            ctx,
            "demo",
            "conc_tsm",
            {"type": "Point", "coordinates": [lon, lat]},
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(
            {
                "count": 1,
                "minimum": expected_value,
                "maximum": expected_value,
                "mean": expected_value,
                "deviation": 0.0,
            },
            result,
        )
        self.assertAlmostEqual(44.5496, result.get("mean"), places=4)
        self.assertNotIn("histogram", result)

    def test_compute_statistics_for_oor_point(self):
        lon = -100  # --> out-of-range!
        lat = 51.465
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()

        result = compute_statistics(
            ctx,
            "demo",
            "conc_tsm",
            {"type": "Point", "coordinates": [lon, lat]},
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual({"count": 0}, result)

    def test_compute_statistics_for_polygon(self):
        lon = 1.768
        lat = 51.465
        delta = 0.05
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()

        result = compute_statistics(
            ctx,
            "demo",
            "conc_tsm",
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon, lat],
                        [lon + delta, lat],
                        [lon + delta, lat + delta],
                        [lon, lat + delta],
                        [lon, lat],
                    ]
                ],
            },
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(380, result.get("count"))
        self.assertAlmostEqual(11.6694, result.get("minimum"), places=4)
        self.assertAlmostEqual(105.8394, result.get("maximum"), places=4)
        self.assertAlmostEqual(22.9632, result.get("mean"), places=4)
        self.assertAlmostEqual(10.0869, result.get("deviation"), places=4)
        histogram = result.get("histogram")
        self.assertIsInstance(histogram, dict)
        self.assertIsInstance(histogram.get("values"), list)
        self.assertIsInstance(histogram.get("edges"), list)
        self.assertEqual(100, len(histogram.get("values")))
        self.assertEqual(101, len(histogram.get("edges")))

    def test_compute_statistics_for_polygon_and_var_assignment(self):
        lon = 1.768
        lat = 51.465
        delta = 0.05
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()

        result = compute_statistics(
            ctx,
            "demo",
            "tsm05 = conc_tsm / 2",
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon, lat],
                        [lon + delta, lat],
                        [lon + delta, lat + delta],
                        [lon, lat + delta],
                        [lon, lat],
                    ]
                ],
            },
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(380, result.get("count"))
        self.assertAlmostEqual(11.6694 / 2, result.get("minimum"), places=4)
        self.assertAlmostEqual(105.8394 / 2, result.get("maximum"), places=4)
        self.assertAlmostEqual(22.9632 / 2, result.get("mean"), places=4)
        self.assertAlmostEqual(10.0869 / 2, result.get("deviation"), places=4)
        histogram = result.get("histogram")
        self.assertIsInstance(histogram, dict)
        self.assertIsInstance(histogram.get("values"), list)
        self.assertIsInstance(histogram.get("edges"), list)
        self.assertEqual(100, len(histogram.get("values")))
        self.assertEqual(101, len(histogram.get("edges")))

    def test_compute_statistics_for_oor_polygon(self):
        lon = -100  # --> out-of-range!
        lat = 51.465
        delta = 0.05
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()

        result = compute_statistics(
            ctx,
            "demo",
            "conc_tsm",
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon, lat],
                        [lon + delta, lat],
                        [lon + delta, lat + delta],
                        [lon, lat + delta],
                        [lon, lat],
                    ]
                ],
            },
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual({"count": 0}, result)
