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

from typing import Sequence, Any, Dict, Callable, Optional

import xarray as xr

from .abc import MultiLevelDataset
from .lazy import LazyMultiLevelDataset


class CombinedMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset that is a combination of other
    multi-level datasets.

    Args:
        ml_datasets: The multi-level datasets to be combined. At least
            two must be provided.
        ds_id: Optional dataset identifier.
        combiner_function: An optional function used to combine the
            datasets, for example `xarray.merge`. If given, it
            receives a list of datasets (`xarray.Dataset` instances)
            and *combiner_params* as keyword arguments. If not given or
            `None` is passed, a copy of the first dataset is made,
            which is then subsequently updated by the remaining datasets
            using `xarray.Dataset.update()`.
        combiner_params: Parameters to the *combiner_function* passed as
            keyword arguments.
    """

    def __init__(
        self,
        ml_datasets: Sequence[MultiLevelDataset],
        ds_id: Optional[str] = None,
        combiner_function: Optional[Callable] = None,
        combiner_params: Optional[Dict[str, Any]] = None,
    ):
        if not ml_datasets or len(ml_datasets) < 2:
            raise ValueError("ml_datasets must have at least two elements")
        super().__init__(ds_id=ds_id, parameters=combiner_params)
        self._ml_datasets = ml_datasets
        self._combiner_function = combiner_function

    def _get_num_levels_lazily(self) -> int:
        return self._ml_datasets[0].num_levels

    def _get_dataset_lazily(
        self, index: int, combiner_params: Dict[str, Any]
    ) -> xr.Dataset:
        datasets = [ml_dataset.get_dataset(index) for ml_dataset in self._ml_datasets]
        if self._combiner_function is None:
            combined_dataset = datasets[0].copy()
            for dataset in datasets[1:]:
                combined_dataset.update(dataset)
            return combined_dataset
        else:
            return self._combiner_function(datasets, **combiner_params)

    def close(self):
        for ml_dataset in self._ml_datasets:
            ml_dataset.close()
