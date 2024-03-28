# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from abc import abstractmethod, ABC
from typing import Any, List, Optional

import xarray as xr

from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.util.assertions import assert_given
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry
from .datatype import DataType
from .datatype import DataTypeLike
from .error import DataStoreError


#######################################################
# Data accessor instantiation and registry query
#######################################################


def new_data_opener(
    opener_id: str,
    extension_registry: Optional[ExtensionRegistry] = None,
    **opener_params,
) -> "DataOpener":
    """Get an instance of the data opener identified by *opener_id*.

    The optional, extra opener parameters *opener_params* may
    be used by data store (`xcube.core.store.DataStore`)
    implementations so they can share their internal state with the opener.

    Args:
        opener_id: The data opener identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.
        **opener_params: Extra opener parameters.

    Returns:
        A data opener instance.
    """
    assert_given(opener_id, "opener_id")
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_OPENERS, opener_id):
        raise DataStoreError(f"A data opener named" f" {opener_id!r} is not registered")
    return extension_registry.get_component(EXTENSION_POINT_DATA_OPENERS, opener_id)(
        **opener_params
    )


def new_data_writer(
    writer_id: str,
    extension_registry: Optional[ExtensionRegistry] = None,
    **writer_params,
) -> "DataWriter":
    """Get an instance of the data writer identified by *writer_id*.

    The optional, extra writer parameters *writer_params* may be used by
    data store (`xcube.core.store.DataStore`) implementations so they
    can share their internal state with the writer.

    Args:
        writer_id: The data writer identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.
        **writer_params: Extra writer parameters.

    Returns:
        A data writer instance.
    """
    assert_given(writer_id, "writer_id")
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_WRITERS, writer_id):
        raise DataStoreError(f"A data writer named" f" {writer_id!r} is not registered")
    return extension_registry.get_component(EXTENSION_POINT_DATA_WRITERS, writer_id)(
        **writer_params
    )


def find_data_opener_extensions(
    predicate: ExtensionPredicate = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> List[Extension]:
    """Get registered data opener extensions using the optional
    filter function *predicate*.

    Args:
        predicate: An optional filter function.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(
        EXTENSION_POINT_DATA_OPENERS, predicate=predicate
    )


def find_data_writer_extensions(
    predicate: ExtensionPredicate = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> List[Extension]:
    """Get registered data writer extensions using the optional filter
    function *predicate*.

    Args:
        predicate: An optional filter function.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(
        EXTENSION_POINT_DATA_WRITERS, predicate=predicate
    )


def get_data_accessor_predicate(
    data_type: DataTypeLike = None, format_id: str = None, storage_id: str = None
) -> ExtensionPredicate:
    """Get a predicate that checks if a data accessor extensions's name is
    compliant with *data_type*, *format_id*, *storage_id*.

    Args:
        data_type: Optional data data type to be supported. May be given
            as type alias name, as a type, or as a DataType instance.
        format_id: Optional data format identifier to be supported.
        storage_id: Optional data storage identifier to be supported.

    Returns:
        A filter function.

    Raises:
        DataStoreError: If an error occurs.
    """
    if any((data_type, format_id, storage_id)):
        data_type = DataType.normalize(data_type) if data_type is not None else None

        def _predicate(extension: Extension) -> bool:
            extension_parts = extension.name.split(":", maxsplit=4)
            if storage_id is not None:
                ext_storage_id = extension_parts[2]
                if ext_storage_id != "*" and ext_storage_id != storage_id:
                    return False
            if format_id is not None:
                ext_format_id = extension_parts[1]
                if ext_format_id != "*" and ext_format_id != format_id:
                    return False
            if data_type is not None:
                ext_data_type = DataType.normalize(extension_parts[0])
                if not data_type.is_super_type_of(ext_data_type):
                    return False
            return True

    else:
        # noinspection PyUnusedLocal
        def _predicate(extension: Extension) -> bool:
            return True

    return _predicate


#######################################################
# Classes
#######################################################


class DataOpener(ABC):
    """An interface that specifies a parameterized `open_data()` operation.

    Possible open parameters are implementation-specific and
    are described by a JSON Schema.

    Note this interface uses the term "opener" to underline the expected
    laziness of the operation. For example, when a xarray.Dataset is
    returned from a Zarr directory, the actual data is represented by
    Dask arrays and will be loaded only on-demand.
    """

    @abstractmethod
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *open_params* to
        `open_data(data_id, open_params)`.
        If *data_id* is given, the returned schema will be tailored
        to the constraints implied by the identified data resource.
        Some openers might not support this, therefore *data_id*
        is optional, and if it is omitted, the returned schema will be
        less restrictive.

        Args:
            data_id: An optional data resource identifier.

        Returns:
            The schema for the parameters in *open_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def open_data(self, data_id: str, **open_params) -> Any:
        """Open the data resource given by the data resource identifier
        *data_id* using the supplied *open_params*.

        Raises if *data_id* does not exist.

        Args:
            data_id: The data resource identifier.
            **open_params: Opener-specific parameters.

        Returns:
            An xarray.Dataset instance.

        Raises:
            DataStoreError: If an error occurs.
        """


class DataDeleter(ABC):
    """An interface that specifies a parameterized `delete_data()` operation.

    Possible delete parameters are implementation-specific and
    are described by a JSON Schema.
    """

    @abstractmethod
    def get_delete_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *delete_params*
        to `delete_data()`.
        If *data_id* is given, the returned schema will be tailored to
        the constraints implied by the identified data resource.
        Some deleters might not support this, therefore *data_id*
        is optional, and if it is omitted, the returned schema will
        be less restrictive.

        Args:
            data_id: An optional data resource identifier.

        Returns:
            The schema for the parameters in *delete_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def delete_data(self, data_id: str, **delete_params):
        """Delete a data resource. Raises if *data_id* does not exist.

        Args:
            data_id: A data resource identifier known to exist.
            **delete_params: Deleter-specific parameters.

        Raises:
            DataStoreError: If an error occurs.
        """


class DataWriter(DataDeleter, ABC):
    """An interface that specifies a parameterized `write_data()` operation.

    Possible write parameters are implementation-specific and
    are described by a JSON Schema.
    """

    @abstractmethod
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *write_params* to
        `write_data(data resource, data_id, open_params)`.

        Returns:
            The schema for the parameters in *write_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def write_data(
        self, data: Any, data_id: str, replace: bool = False, **write_params
    ) -> str:
        """Write a data resource using the supplied *data_id* and *write_params*.

        Args:
            data: The data resource's in-memory representation to be
                written.
            data_id: A unique data resource identifier.
            replace: Whether to replace an existing data resource.
            **write_params: Writer-specific parameters.

        Returns:
            The data resource identifier used to write the data
            resource.

        Raises:
            DataStoreError: If an error occurs.
        """


class DataTimeSliceUpdater(DataWriter, ABC):
    """An interface that specifies writing of time slice data."""

    @abstractmethod
    def append_data_time_slice(self, data_id: str, time_slice: xr.Dataset):
        """Append a time slice to the identified data resource.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def insert_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """Insert a time slice into the identified data resource at given index.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.
            time_index: The time index.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def replace_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """Replace a time slice in the identified data resource at given index.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.
            time_index: The time index.

        Raises:
            DataStoreError: If an error occurs.
        """
