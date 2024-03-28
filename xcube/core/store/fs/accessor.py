# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import fsspec

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from ..accessor import DataOpener
from ..accessor import DataWriter
from ..datatype import DataType
from ..error import DataStoreError

COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES = dict(
    # passed to `DirCache`, if the implementation supports
    # directory listing caching. Pass use_listings_cache=False
    # to disable such caching.
    use_listings_cache=JsonBooleanSchema(),
    listings_expiry_time=JsonNumberSchema(),
    max_paths=JsonIntegerSchema(),
    # If this is a cachable implementation, pass True here to force
    # creating a new instance even if a matching instance exists, and prevent
    # storing this instance.
    skip_instance_cache=JsonBooleanSchema(),
    asynchronous=JsonBooleanSchema(),
)

PROTOCOL_PARAM_NAME = "protocol"
STORAGE_OPTIONS_PARAM_NAME = "storage_options"
FS_PARAM_NAME = "fs"
ROOT_PARAM_NAME = "root"


class FsAccessor:
    """
    Base class for accessing some filesystem.
    """

    @classmethod
    def get_protocol(cls) -> str:
        """Get the filesystem protocol."""
        return "abstract"

    @classmethod
    def get_storage_options_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema of the filesystem parameters."""
        return JsonObjectSchema(
            properties=COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
            additional_properties=True,
        )

    @classmethod
    def load_fs(
        cls, params: Dict[str, Any]
    ) -> Tuple[fsspec.AbstractFileSystem, Optional[str], Dict[str, Any]]:
        """Load a filesystem instance from *params*.

        Pops useful parameters from a copy of *params* to get
        or instantiate the filesystem.
        Returns the filesystem and *params* reduced by
        the used parameters.

        Args:
            params: Parameters passed to a filesystem
                data store, opener, or writer call.
        Returns:
            A tuple comprising the filesystem, an optional root path,
            and the modified *params*.
        """
        params = dict(params)

        # Filesystem data-stores pass "fs" and "root" kwargs to
        # data opener and writer calls.
        fs = params.pop(FS_PARAM_NAME, None)
        root = params.pop(ROOT_PARAM_NAME, None)
        if fs is not None:
            assert_instance(fs, fsspec.AbstractFileSystem, name=FS_PARAM_NAME)
        if root is not None:
            assert_instance(root, str, name=ROOT_PARAM_NAME)
        if fs:
            return fs, root, params

        protocol = cls.get_protocol()
        if protocol == "abstract":
            protocol = params.pop(PROTOCOL_PARAM_NAME, None)
            if protocol is None:
                raise DataStoreError(
                    f"Cannot determine filesystem,"
                    f" try using parameter"
                    f" {PROTOCOL_PARAM_NAME!r}"
                )

        storage_options = params.pop(STORAGE_OPTIONS_PARAM_NAME, None)
        if storage_options is not None:
            assert_instance(storage_options, dict, name=STORAGE_OPTIONS_PARAM_NAME)
            storage_options = dict(storage_options)

        # Note, by default, filesystem data stores are writable and hence
        # SHALL NOT cache any directory listings!
        use_listings_cache = bool(
            storage_options.pop("use_listings_cache", False)
            if storage_options
            else False
        )

        try:
            return (
                fsspec.filesystem(
                    protocol,
                    use_listings_cache=use_listings_cache,
                    **(storage_options or {}),
                ),
                root,
                params,
            )
        except (ValueError, ImportError) as error:
            raise DataStoreError(f"Cannot instantiate filesystem {protocol!r}: {error}")

    @classmethod
    def add_storage_options_to_params_schema(
        cls, params_schema: JsonObjectSchema
    ) -> JsonObjectSchema:
        """
        Utility method to be used by subclasses to add the schema
        for the parameter "storage_options" to given *param_schema*.
        """
        params_schema = copy.deepcopy(params_schema)
        params_schema.properties[
            STORAGE_OPTIONS_PARAM_NAME
        ] = cls.get_storage_options_schema()
        return params_schema

    @classmethod
    def remove_storage_options_from_params_schema(
        cls, params_schema: JsonObjectSchema
    ) -> JsonObjectSchema:
        """
        Utility method to be used by subclasses to remove the schema
        for the parameter "storage_options" from given *param_schema*.
        """
        if STORAGE_OPTIONS_PARAM_NAME in params_schema.properties:
            params_schema = copy.deepcopy(params_schema)
            del params_schema.properties[STORAGE_OPTIONS_PARAM_NAME]
        return params_schema


class FsDataAccessor(DataOpener, DataWriter, FsAccessor, ABC):
    """Abstract base class for data accessors that
    use an underlying filesystem.

    A `FsDataAccessor` is responsible for exactly one data type
    and one data format.

    Note, for all filesystem based accessors the parameter
    *data_id* is an absolute path that is build by the store
    according to `f"{store.root}/{data_id}"`.
    The store's root may therefore only be useful to
    compute the original, relative store's *data_id*.
    """

    @classmethod
    @abstractmethod
    def get_data_type(cls) -> DataType:
        """Get the supported data type."""

    @classmethod
    @abstractmethod
    def get_format_id(cls) -> str:
        """Get the format identifier,
        for example "zarr" or "geojson".
        """

    def get_delete_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                recursive=JsonBooleanSchema(),
                maxdepth=JsonIntegerSchema(),
                storage_options=self.get_storage_options_schema(),
            ),
            additional_properties=False,
        )

    def delete_data(self, data_id: str, **delete_params):
        fs, _, delete_params = self.load_fs(delete_params)
        fs.delete(data_id, **delete_params)
