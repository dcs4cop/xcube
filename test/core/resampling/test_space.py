import unittest

import numpy as np

from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.resampling import resample_in_space

nan = np.nan


class ResampleInSpaceTest(unittest.TestCase):
    # TODO: write more tests

    # noinspection PyMethodMayBeStatic
    def test_affine_transform_dataset(self):
        source_ds = new_cube()
        source_gm = GridMapping.from_dataset(source_ds)
        target_gm = GridMapping.regular(size=(8, 4),
                                        xy_min=(0, 0),
                                        xy_res=2,
                                        crs=CRS_WGS84)

        # TODO: assert stuff
        resample_in_space(source_ds, source_gm, target_gm)
