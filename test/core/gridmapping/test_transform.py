import unittest

import pyproj.crs

from test.sampledata import create_s2plus_dataset
from xcube.core.gridmapping import CRS_CRS84
from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping

CRS_UTM_32N = pyproj.crs.CRS(32632)

import numpy as np


# noinspection PyMethodMayBeStatic
class TransformTest(unittest.TestCase):

    def test_transform(self):
        gm = GridMapping.regular(size=(3, 3), xy_min=(10, 53), xy_res=0.1, crs=CRS_CRS84)
        gm_t = gm.transform(crs=CRS_UTM_32N)

        self.assertEqual(CRS_UTM_32N, gm_t.crs)
        self.assertEqual(False, gm_t.is_regular)
        self.assertEqual(('transformed_x', 'transformed_y'), gm_t.xy_var_names)
        self.assertEqual(('lon', 'lat'), gm_t.xy_dim_names)
        np.testing.assert_almost_equal(
            np.array([
                [570057.076286, 576728.9360228, 583400.7295284],
                [570220.3304187, 576907.7404859, 583595.0849538],
                [570383.3684844, 577086.3083212, 583789.1831954]
            ]),
            gm_t.xy_coords[0])
        np.testing.assert_almost_equal(
            np.array([
                [5900595.928991, 5900698.5746648, 5900810.5532744],
                [5889471.9033896, 5889574.6540572, 5889686.7472201],
                [5878348.0594403, 5878450.9138481, 5878563.1201969]
            ]),
            gm_t.xy_coords[1])

    def test_transform_xy_var_names(self):
        gm = GridMapping.regular(size=(3, 3), xy_min=(10, 53), xy_res=0.1, crs=CRS_CRS84)
        gm_t = gm.transform(crs=CRS_UTM_32N, xy_var_names=('x', 'y'))
        self.assertEqual(CRS_UTM_32N, gm_t.crs)
        self.assertEqual(('x', 'y'), gm_t.xy_var_names)
        self.assertEqual(('lon', 'lat'), gm_t.xy_dim_names)

    def test_transform_s2(self):
        dataset = create_s2plus_dataset()

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=True)
        # Assert we've picked the projected one which is regular
        self.assertEqual("Projected CRS", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm_t = gm.transform(CRS_CRS84)
        self.assertEqual(CRS_CRS84, gm_t.crs)

        gm_t = gm.transform(CRS_WGS84)
        self.assertEqual(CRS_WGS84, gm_t.crs)

    def test_transform_no_op(self):
        dataset = create_s2plus_dataset()

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=True)
        # Assert we've picked the projected one which is regular
        self.assertEqual("Projected CRS", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm_t = gm.transform(gm.crs)
        self.assertIs(gm, gm_t)

        # Almost no op
        gm = GridMapping.regular(size=(3, 3), xy_min=(10, 53), xy_res=0.1, crs=CRS_CRS84)
        gm_t = gm.transform(crs=gm.crs, xy_var_names=('x', 'y'))
        self.assertEqual(('x', 'y'), gm_t.xy_var_names)

