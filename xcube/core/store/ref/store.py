# The MIT License (MIT)
# Copyright (c) 2020-2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings
from typing import Iterator, Any, Tuple, Container, Union, Dict, List

import fsspec
import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema

from ..datatype import DataTypeLike
from ..descriptor import DataDescriptor, DatasetDescriptor
from ..descriptor import new_data_descriptor
from ..store import DataStore
from .schema import REF_STORE_SCHEMA


class ReferenceDataStore(DataStore):
    """A data store that uses kerchunk reference files to
    open referred NetCDF datasets as if they were Zarr datasets.

    Args:
        refs: The list of reference JSON files to use for this instance.
            Files (URLs or local paths) are used in conjunction with
            target_options and target_protocol to open and parse JSON at
            this location.
        target_protocol: Used for loading the reference files. If not
            given, protocol will be derived from the given path.
        target_options: Extra filesystem options for loading the
            reference files.
        remote_protocol: The protocol of the filesystem on which the
            references will be evaluated. If not given, will be derived
            from the first URL in the references that has a protocol.
        remote_options: Extra filesystem options for loading the
            referenced data.
        max_gap: See `max_block`.
        max_block: For merging multiple concurrent requests to the same
            remote file. Neighboring byte ranges will only be merged
            when their inter-range gap is <= `max_gap`. Default is 64KB.
            Set to 0 to only merge when it requires no extra bytes. Pass
            a negative number to disable merging, appropriate for local
            target files. Neighboring byte ranges will only be merged
            when the size of the aggregated range is <= `max_block`.
            Default is 256MB.
        cache_size: Maximum size of LRU cache, where
            cache_size*record_size denotes the total number of
            references that can be loaded in memory at once. Only used
            for lazily loaded references.
    """

    def __init__(
        self,
        refs: List[str],
        **ref_kwargs,
    ):
        self._refs = dict(self._normalize_ref(ref) for ref in refs)
        self._ref_kwargs = ref_kwargs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return REF_STORE_SCHEMA

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return ("dataset",)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        return iter(self._refs.keys())

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        return data_id in self._refs

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DataDescriptor:
        data_descriptor = self._refs[data_id].get("data_descriptor")
        if data_descriptor is None:
            dataset = self.open_data(data_id)
            data_descriptor = new_data_descriptor(data_id, dataset)
            self._refs[data_id]["data_descriptor"] = data_descriptor
        return data_descriptor

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        return ("dataset:zarr:reference",)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        # We do not have open parameters yet
        return JsonObjectSchema()

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        if open_params:
            warnings.warn(
                f"open_params are not supported yet,"
                f" but passing forward {', '.join(open_params.keys())}"
            )
        ref_path = self._refs[data_id]["ref_path"]
        open_params.pop("consolidated", False)
        ref_mapping = fsspec.get_mapper("reference://", fo=ref_path, **self._ref_kwargs)
        return xr.open_zarr(ref_mapping, consolidated=False, **open_params)

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        # We do not have search parameters yet
        return JsonObjectSchema()

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        if search_params:
            warnings.warn(
                f"search_params are not supported yet,"
                f" but received {', '.join(search_params.keys())}"
            )
        return (self.describe_data(data_id) for data_id in self.get_data_ids())

    @classmethod
    def _normalize_ref(
        cls, ref: Union[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        if isinstance(ref, str):
            ref_path = ref
            data_id = cls._ref_path_to_data_id(ref_path)
            return data_id, dict(ref_path=ref_path, data_descriptor=None)

        if isinstance(ref, dict):
            ref_path = ref.get("ref_path") or None
            if not ref_path:
                raise ValueError("missing key ref_path in refs item")

            data_descriptor = ref.get("data_descriptor") or None
            if isinstance(data_descriptor, dict):
                data_descriptor = DatasetDescriptor.from_dict(data_descriptor)
            elif data_descriptor is not None:
                raise TypeError(
                    "value of data_descriptor key"
                    " in refs item must be a dict or None"
                )

            data_id = ref.get("data_id") or None
            if data_id is None and data_descriptor is not None:
                data_id = data_descriptor.data_id
            if data_id is None:
                data_id = cls._ref_path_to_data_id(ref_path)

            return data_id, dict(ref_path=ref_path, data_descriptor=data_descriptor)
        raise TypeError("item in refs must be a str or a dict")

    @classmethod
    def _ref_path_to_data_id(cls, ref_path: str) -> str:
        protocol, path = fsspec.core.split_protocol(ref_path)
        if protocol in (None, "file", "local") and os.path.sep != "/":
            path = path.replace(os.path.sep, "/")
        name = path.rsplit("/", maxsplit=1)[-1]
        return name[:-5] if name.endswith(".json") else name
