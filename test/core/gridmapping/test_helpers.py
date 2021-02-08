import unittest
from fractions import Fraction

from xcube.core.gridmapping.helpers import round_to_fraction


class RoundToFractionTest(unittest.TestCase):
    dump = False

    def test_invalid(self):
        with self.assertRaises(ValueError):
            round_to_fraction(0.29, digits=0)
        with self.assertRaises(ValueError):
            round_to_fraction(0.29, resolution=0)
        with self.assertRaises(ValueError):
            round_to_fraction(0.29, resolution=0.12)

    def test_1_025(self):
        def f(value):
            return float(round_to_fraction(value, 1, 0.25))

        self.assertAlmostEqual(-1, f(-1))
        self.assertAlmostEqual(0.0, f(0))
        self.assertAlmostEqual(1.0, f(1))
        self.assertAlmostEqual(1.25, f(1.2))
        self.assertAlmostEqual(1.25, f(1.3))
        self.assertAlmostEqual(1.5, f(1.4))
        self.assertAlmostEqual(1.5, f(1.45))
        self.assertAlmostEqual(1.5, f(1.51))
        self.assertAlmostEqual(1.75, f(1.7))
        self.assertAlmostEqual(2.0, f(1.9))
        self.assertAlmostEqual(2.0, f(1.96))
        self.assertAlmostEqual(2.0, f(1.98))
        self.assertAlmostEqual(2.0, f(2))

    def test_2_025(self):
        def f(value):
            return float(round_to_fraction(value, 2, 0.25))

        self.assertAlmostEqual(-1, f(-1))
        self.assertAlmostEqual(0.0, f(0))
        self.assertAlmostEqual(1.0, f(1))
        self.assertAlmostEqual(1.2, f(1.2))
        self.assertAlmostEqual(1.225, f(1.23))
        self.assertAlmostEqual(1.3, f(1.3))
        self.assertAlmostEqual(1.4, f(1.4))
        self.assertAlmostEqual(1.45, f(1.45))
        self.assertAlmostEqual(1.5, f(1.51))
        self.assertAlmostEqual(1.7, f(1.7))
        self.assertAlmostEqual(1.8, f(1.79))
        self.assertAlmostEqual(1.9, f(1.9))
        self.assertAlmostEqual(1.95, f(1.96))
        self.assertAlmostEqual(1.975, f(1.98))
        self.assertAlmostEqual(2.0, f(2))

    def test_default(self):
        values = [
            [-1.0, -1.0, Fraction(-1, 1)],
            [0.0, 0.0, Fraction(0, 1)],
            [5.247476065426347e-09, 5.2e-09, Fraction(13, 2500000000)],
            [3.427467229408875e-06, 3.4e-06, Fraction(17, 5000000)],
            [4.501758583626108e-06, 4.5e-06, Fraction(9, 2000000)],
            [1.1351705264714663e-05, 1.1e-05, Fraction(11, 1000000)],
            [0.00048171747406886744, 0.00048, Fraction(3, 6250)],
            [0.0018032657496927416, 0.0018, Fraction(9, 5000)],
            [0.0019897341919324425, 0.002, Fraction(1, 500)],
            [0.0041643509375105065, 0.0042, Fraction(21, 5000)],
            [0.030607346091352187, 0.031, Fraction(31, 1000)],
            [1.0076973439575128, 1.0, Fraction(1, 1)],
            [1.0, 1.0, Fraction(1, 1)],
            [84.54360269093455, 85.0, Fraction(85, 1)],
            [494.86581234602096, 490.0, Fraction(490, 1)],
            [987.9441243998718, 990.0, Fraction(990, 1)],
            [1757.368043916636, 1800.0, Fraction(1800, 1)],
            [1143506.2928512183, 1100000.0, Fraction(1100000, 1)],
            [217971970.75235566, 220000000.0, Fraction(220000000, 1)]
        ]
        self._assert_values(values, dict())

    def test_3_025(self):
        actual = round_to_fraction(1, digits=1, resolution=0.25)
        self.assertEqual(Fraction(1, 1), actual)
        values = [
            [-1.0, -1.0, Fraction(-1, 1)],
            [0.0, 0.0, Fraction(0, 1)],
            [5.247476065426347e-09, 5.2475e-09, Fraction(2099, 400000000000)],
            [3.427467229408875e-06, 3.4275e-06, Fraction(1371, 400000000)],
            [4.501758583626108e-06, 4.5025e-06, Fraction(1801, 400000000)],
            [1.1351705264714663e-05, 1.135e-05, Fraction(227, 20000000)],
            [0.00048171747406886744, 0.00048175, Fraction(1927, 4000000)],
            [0.0018032657496927416, 0.0018025, Fraction(721, 400000)],
            [0.0019897341919324425, 0.00199, Fraction(199, 100000)],
            [0.0041643509375105065, 0.004165, Fraction(833, 200000)],
            [0.030607346091352187, 0.0306, Fraction(153, 5000)],
            [1.0076973439575128, 1.0075, Fraction(403, 400)],
            [1.0, 1.0, Fraction(1, 1)],
            [84.54360269093455, 84.55, Fraction(1691, 20)],
            [494.86581234602096, 494.75, Fraction(1979, 4)],
            [987.9441243998718, 988.0, Fraction(988, 1)],
            [1757.368043916636, 1757.5, Fraction(3515, 2)],
            [1143506.2928512183, 1142500.0, Fraction(1142500, 1)],
            [217971970.75235566, 218000000.0, Fraction(218000000, 1)]
        ]
        self._assert_values(values, dict(digits=3, resolution=0.25))

    def test_2_5(self):
        values = [
            [-1.0, -1.0, Fraction(-1, 1)],
            [0.0, 0.0, Fraction(0, 1)],
            [5.247476065426347e-09, 5.25e-09, Fraction(21, 4000000000)],
            [3.427467229408875e-06, 3.45e-06, Fraction(69, 20000000)],
            [4.501758583626108e-06, 4.5e-06, Fraction(9, 2000000)],
            [1.1351705264714663e-05, 1.15e-05, Fraction(23, 2000000)],
            [0.00048171747406886744, 0.00048, Fraction(3, 6250)],
            [0.0018032657496927416, 0.0018, Fraction(9, 5000)],
            [0.0019897341919324425, 0.002, Fraction(1, 500)],
            [0.0041643509375105065, 0.00415, Fraction(83, 20000)],
            [0.030607346091352187, 0.0305, Fraction(61, 2000)],
            [1.0076973439575128, 1.0, Fraction(1, 1)],
            [1.0, 1.0, Fraction(1, 1)],
            [84.54360269093455, 84.5, Fraction(169, 2)],
            [494.86581234602096, 495.0, Fraction(495, 1)],
            [987.9441243998718, 990.0, Fraction(990, 1)],
            [1757.368043916636, 1750.0, Fraction(1750, 1)],
            [1143506.2928512183, 1150000.0, Fraction(1150000, 1)],
            [217971970.75235566, 220000000.0, Fraction(220000000, 1)]
        ]
        self._assert_values(values, dict(digits=2, resolution=0.5))

    def _assert_values(self, values, kwargs):
        if self.dump:
            results = []
            for value, _, _ in values:
                actual_fraction = round_to_fraction(value, **kwargs)
                results.append([value, float(actual_fraction), actual_fraction])
            import pprint
            pprint.pprint(results)

        for value, expected_float, expected_fraction in values:
            actual_fraction = round_to_fraction(value, **kwargs)
            self.assertEqual(expected_fraction, actual_fraction)
            self.assertAlmostEqual(expected_float, float(actual_fraction))
