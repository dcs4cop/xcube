import os
from typing import List
import xarray as xr
from test.cli.helpers import CliDataTest, TEST_ZARR_DIR
from xcube.core.verify import assert_cube


class RectifyTest(CliDataTest):

    def outputs(self) -> List[str]:
        return ['out.zarr']

    def test_rectify_without_vars(self):
        """Test that rectify selects all variables when --var is not given."""

        # For now, specify the image geometry explicitly with --size, --point,
        # and --res to avoid triggering an "invalid y_min" ValueError when
        # ImageGeom tries to determine it automatically. Once Issue #303 has
        # been fixed, these options can be omitted.

        result = self.invoke_cli(['rectify',
                                  '--size', '508,253',
                                  '--point', '-179.5,-89.5',
                                  '--res', '0.7071067811865475',
                                  '--crs', 'EPSG:4326',
                                  TEST_ZARR_DIR])
        self.assertEqual(0, result.exit_code)
        self.assertEqual('Opening dataset from \'test.zarr\'...\n'
                         'Rectifying...\n'
                         'Writing rectified dataset to \'out.zarr\'...\n'
                         'Done.\n',
                         result.stdout)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        self.assertIn('precipitation', ds)
        self.assertIn('temperature', ds)
        self.assertIn('soil_moisture', ds)

    def test_rectify_multiple_comma_separated_vars(self):
        """Test that rectify selects the desired variables when
        multiple --var options, some with multiple comma-separated
        variable names as an argument, are passed."""

        # For now, specify the image geometry explicitly with --size, --point,
        # and --res to avoid triggering an "invalid y_min" ValueError when
        # ImageGeom tries to determine it automatically. Once Issue #303 has
        # been fixed, these options can be omitted.

        result = self.invoke_cli(['rectify',
                                  '--size', '508,253',
                                  '--point', '-179.5,-89.5',
                                  '--res', '0.7071067811865475',
                                  '--crs', 'EPSG:4326',
                                  '--var', 'precipitation,temperature',
                                  '--var', 'soil_moisture',
                                  TEST_ZARR_DIR])
        self.assertEqual(0, result.exit_code)
        self.assertEqual('Opening dataset from \'test.zarr\'...\n'
                         'Rectifying...\n'
                         'Writing rectified dataset to \'out.zarr\'...\n'
                         'Done.\n',
                         result.stdout)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        self.assertIn('precipitation', ds)
        self.assertIn('temperature', ds)
        self.assertIn('soil_moisture', ds)
