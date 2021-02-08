import unittest

from test.sampledata import create_highroc_dataset
from xcube.core.select import select_spatial_subset, select_variables_subset


class SelectVariablesSubsetTest(unittest.TestCase):
    def test_select_variables_subset_all(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_variables_subset(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = select_variables_subset(ds1, ds1.data_vars.keys())
        self.assertIs(ds2, ds1)

    def test_select_variables_subset_none(self):
        ds1 = create_highroc_dataset()
        ds2 = select_variables_subset(ds1, [])
        self.assertEqual(0, len(ds2.data_vars))
        ds2 = select_variables_subset(ds1, ['bibo'])
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_subset_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_variables_subset(ds1, ['conc_chl', 'c2rcc_flags', 'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))


class SelectSpatialSubsetTest(unittest.TestCase):
    def test_select_spatial_subset_all_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(0, 0, 4, 3))
        self.assertIs(ds2, ds1)

    def test_select_spatial_subset_some_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(1, 1, 4, 3))
        self.assertEqual((2, 3), ds2.conc_chl.shape)

    def test_select_spatial_subset_none_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(5, 6, 7, 8))
        self.assertEqual(None, ds2)
        ds2 = select_spatial_subset(ds1, ij_bbox=(-6, -4, 2, 2))
        self.assertEqual(None, ds2)

    def test_select_spatial_subset_all_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(7.9, 53.9, 12., 56.4))
        self.assertIs(ds2, ds1)

    def test_select_spatial_subset_some_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(8., 55, 10., 56.))
        self.assertEqual((3, 3), ds2.conc_chl.shape)

    def test_select_spatial_subset_none_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(13., 57., 15., 60.))
        self.assertEqual(None, ds2)
        ds2 = select_spatial_subset(ds1, xy_bbox=(5.5, 55, 6.5, 56))
        self.assertEqual(None, ds2)

    def test_select_spatial_subset_invalid_params(self):
        ds1 = create_highroc_dataset()
        with self.assertRaises(ValueError) as cm:
            select_spatial_subset(ds1, ij_bbox=(5, 6, 7, 8), xy_bbox=(0., 0., 1., 2.))
        self.assertEqual("Only one of ij_bbox and xy_bbox can be given", f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            select_spatial_subset(ds1)
        self.assertEqual("One of ij_bbox and xy_bbox must be given", f'{cm.exception}')
