# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
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
import pathlib
from typing import Optional, Tuple, Dict, Any

import fsspec
import rasterio
import rioxarray
import xarray as xr

from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.store.fs.helpers import get_fs_path_class


class GeoTIFFMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 root: Optional[str],
                 data_id: str,
                 **open_params: Dict[str, Any]):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._path = data_id
        self._open_params = open_params
        self._path_class = get_fs_path_class(fs)

    def _get_num_levels_lazily(self) -> int:
        with rasterio.open(self._get_full_path()) as rio_dataset:
            overviews = rio_dataset.overviews(1)
            # TODO validate overviews/ resolution must increase by factor of 2
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) \
            -> xr.Dataset:
        # TODO get tile size from parameters
        return self.open_dataset(self._get_full_path(), (512, 512),
                                 overview_level=index)

    @classmethod
    def open_dataset(cls,
                     file_spec: str,
                     tile_size: Tuple[int, int],
                     overview_level: Optional[int] = 0) -> xr.Dataset:
        """

        @param overview_level: int
        @param tile_size: tuple
        @type file_spec: object
        """
        array: xr.DataArray = rioxarray.open_rasterio(
            file_spec,
            overview_level=overview_level - 1 if overview_level > 0 else None,
            chunks=dict(zip(('x', 'y'), tile_size))
        )
        arrays = {}
        if array.ndim == 3:
            for i in range(array.shape[0]):
                name = f'{array.name or "band"}_{i + 1}'
                dims = array.dims[-2:]
                coords = {n: v
                          for n, v in array.coords.items()
                          if n in dims or n == 'spatial_ref'}
                band_data = array.data[i, :, :]
                arrays[name] = xr.DataArray(band_data,
                                            coords=coords,
                                            dims=dims,
                                            attrs=dict(**array.attrs))
        elif array.ndim == 2:
            name = f'{array.name or "band"}'
            arrays[name] = array
        else:
            raise RuntimeError('number of dimensions must be 2 or 3')

        dataset = xr.Dataset(arrays, attrs=dict(source=file_spec))
        # For CRS, rioxarray uses variable "spatial_ref" by default
        if 'spatial_ref' in array.coords:
            for data_var in dataset.data_vars.values():
                data_var.attrs['grid_mapping'] = 'spatial_ref'

        return dataset

    def _get_full_path(self):
        # TODO:change to complete path
        return self._path

    def _get_path(self, *args) -> pathlib.PurePath:
        return self._path_class(*args)