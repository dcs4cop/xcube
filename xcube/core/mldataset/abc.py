# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
from abc import abstractmethod, ABCMeta
from functools import cached_property
from typing import Sequence, Any, Dict, Callable, Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.tilingscheme import TilingScheme
from xcube.util.types import ScalarOrPair
from xcube.util.types import normalize_scalar_or_pair


class MultiLevelDataset(metaclass=ABCMeta):
    """A multi-level dataset of decreasing spatial resolutions
    (a multi-resolution pyramid).

    The pyramid level at index zero provides the original spatial dimensions.
    The size of the spatial dimensions in subsequent levels
    is computed by the formula `size[index + 1] = (size[index] + 1) // 2`
    with `size[index]` being the maximum size of the spatial dimensions
    at level zero.

    Any dataset chunks are assumed to be the same in all levels. Usually,
    the number of chunks is one in one of the spatial dimensions of the
    highest level.
    """

    @property
    @abstractmethod
    def ds_id(self) -> str:
        """Returns:
        the dataset identifier.
        """

    @ds_id.setter
    @abstractmethod
    def ds_id(self, ds_id: str):
        """Set the dataset identifier."""

    @property
    @abstractmethod
    def grid_mapping(self) -> GridMapping:
        """Returns:
        the CF-conformal grid mapping
        """

    @property
    @abstractmethod
    def num_levels(self) -> int:
        """Returns:
        the number of pyramid levels.
        """

    @cached_property
    def resolutions(self) -> Sequence[Tuple[float, float]]:
        """Returns:
        the x,y resolutions for each level given in the spatial
        units of the dataset's CRS (i.e. `self.grid_mapping.crs`).
        """
        x_res_0, y_res_0 = self.grid_mapping.xy_res
        return [
            (x_res_0 * (1 << level), y_res_0 * (1 << level))
            for level in range(self.num_levels)
        ]

    @cached_property
    def avg_resolutions(self) -> Sequence[float]:
        """Returns:
        the average x,y resolutions for each level given in the
        spatial units of the dataset's CRS (i.e.
        `self.grid_mapping.crs`).
        """
        x_res_0, y_res_0 = self.grid_mapping.xy_res
        xy_res_0 = math.sqrt(x_res_0 * y_res_0)
        return [xy_res_0 * (1 << level) for level in range(self.num_levels)]

    @property
    def base_dataset(self) -> xr.Dataset:
        """Returns:
        the base dataset for lowest level at index 0.
        """
        return self.get_dataset(0)

    @property
    def datasets(self) -> Sequence[xr.Dataset]:
        """Get datasets for all levels.

        Calling this method will trigger any lazy dataset instantiation.

        Returns:
            the datasets for all levels.
        """
        return [self.get_dataset(index) for index in range(self.num_levels)]

    @abstractmethod
    def get_dataset(self, index: int) -> xr.Dataset:
        """Args:
            index: the level index

        Returns:
            the dataset for the level at *index*.
        """

    def close(self):
        """Close all datasets. Default implementation does nothing."""

    def apply(
        self,
        function: Callable[[xr.Dataset, Dict[str, Any]], xr.Dataset],
        kwargs: Dict[str, Any] = None,
        ds_id: str = None,
    ) -> "MultiLevelDataset":
        """Apply function to all level datasets
        and return a new multi-level dataset.
        """
        from .mapped import MappedMultiLevelDataset

        return MappedMultiLevelDataset(
            self, function, ds_id=ds_id, mapper_params=kwargs
        )

    def derive_tiling_scheme(self, tiling_scheme: TilingScheme):
        """Derive a new tiling scheme for the given one with defined
        minimum and maximum level indices.
        """
        min_level, max_level = tiling_scheme.get_levels_for_resolutions(
            self.avg_resolutions, self.grid_mapping.spatial_unit_name
        )
        return tiling_scheme.derive(min_level=min_level, max_level=max_level)

    def get_level_for_resolution(self, xy_res: ScalarOrPair[float]) -> int:
        """Get the index of the level that best represents the given resolution.

        Args:
            xy_res: the resolution in x- and y-direction

        Returns:
            a level ranging from 0 to self.num_levels - 1
        """
        given_x_res, given_y_res = normalize_scalar_or_pair(xy_res, item_type=float)
        for level, (x_res, y_res) in enumerate(self.resolutions):
            if x_res > given_x_res and y_res > given_y_res:
                return max(0, level - 1)
        return self.num_levels - 1
