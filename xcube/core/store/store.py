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
from typing import Iterator, Tuple, Any, Optional, List, Type, Dict, Union, Container

from xcube.constants import EXTENSION_POINT_DATA_STORES
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry
from .accessor import DataOpener
from .accessor import DataWriter
from .assertions import assert_valid_params
from .datatype import DataTypeLike
from .descriptor import DataDescriptor
from .error import DataStoreError
from .search import DataSearcher


#######################################################
# Data store instantiation and registry query
#######################################################


def new_data_store(
    data_store_id: str,
    extension_registry: Optional[ExtensionRegistry] = None,
    **data_store_params,
) -> Union["DataStore", "MutableDataStore"]:
    """Create a new data store instance for given
    *data_store_id* and *data_store_params*.

    Args:
        data_store_id: A data store identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.
        **data_store_params: Data store specific parameters.

    Returns:
        A new data store instance
    """
    data_store_class = get_data_store_class(
        data_store_id, extension_registry=extension_registry
    )
    data_store_params_schema = data_store_class.get_data_store_params_schema()
    assert_valid_params(
        data_store_params, name="data_store_params", schema=data_store_params_schema
    )
    # noinspection PyArgumentList
    return data_store_class(**data_store_params)


def get_data_store_class(
    data_store_id: str, extension_registry: Optional[ExtensionRegistry] = None
) -> Union[Type["DataStore"], Type["MutableDataStore"]]:
    """Get the class for the data store identified by *data_store_id*.

    Args:
        data_store_id: A data store identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        The class for the data store.
    """
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_STORES, data_store_id):
        raise DataStoreError(
            f'Unknown data store "{data_store_id}"'
            f" (may be due to missing xcube plugin)"
        )
    return extension_registry.get_component(EXTENSION_POINT_DATA_STORES, data_store_id)


def get_data_store_params_schema(
    data_store_id: str, extension_registry: Optional[ExtensionRegistry] = None
) -> JsonObjectSchema:
    """Get the JSON schema for instantiating a new data store
    identified by *data_store_id*.

    Args:
        data_store_id: A data store identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        The JSON schema for the data store's parameters.
    """
    data_store_class = get_data_store_class(
        data_store_id, extension_registry=extension_registry
    )
    return data_store_class.get_data_store_params_schema()


def find_data_store_extensions(
    predicate: ExtensionPredicate = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> List[Extension]:
    """Find data store extensions using the optional filter
    function *predicate*.

    Args:
        predicate: An optional filter function.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        List of data store extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(
        EXTENSION_POINT_DATA_STORES, predicate=predicate
    )


#######################################################
# Classes
#######################################################


class DataStore(DataOpener, DataSearcher, ABC):
    """A data store represents a collection of data resources that
    can be enumerated, queried, and opened in order to obtain
    in-memory representations of the data. The same data resource may be
    made available using different data types. Therefore, many methods
    allow specifying a *data_type* parameter.

    A store implementation may use any existing openers/writers,
    or define its own, or not use any openers/writers at all.

    Store implementers should follow the conventions outlined in
    https://xcube.readthedocs.io/en/latest/storeconv.html .

    The `DataStore` is an abstract base class that both read-only and
    mutable data stores must implement.
    """

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        """Get descriptions of parameters that must or can be used to
        instantiate a new DataStore object.
        Parameters are named and described by the properties of the
        returned JSON object schema.
        The default implementation returns JSON object schema that
        can have any properties.
        """
        return JsonObjectSchema()

    @classmethod
    @abstractmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        """Get alias names for all data types supported by this store.
        The first entry in the tuple represents this store's
        default data type.

        Returns:
            The tuple of supported data types.
        """

    @abstractmethod
    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        """Get alias names for of data types that are supported
        by this store for the given *data_id*.

        Args:
            data_id: An identifier of data that is provided by this
                store

        Returns:
            A tuple of data types that apply to the given *data_id*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """Get an iterator over the data resource identifiers for the
        given type *data_type*. If *data_type* is omitted, all data
        resource identifiers are returned.

        If a store implementation supports only a single data type,
        it should verify that *data_type* is either None or
        compatible with the supported data type.

        If *include_attrs* is provided, it must be a sequence of names
        of metadata attributes. The store will then return extra metadata
        for each returned data resource identifier according to the
        names of the metadata attributes as tuples (*data_id*, *attrs*).

        Hence, the type of the returned iterator items depends on the
        value of *include_attrs*:

        - If *include_attrs* is None (the default), the method returns
          an iterator of dataset identifiers *data_id* of type `str`.
        - If *include_attrs* is a sequence of attribute names, even an
          empty one, the method returns an iterator of tuples
          (*data_id*, *attrs*) of type `Tuple[str, Dict]`, where *attrs*
          is a dictionary filled according to the names in *include_attrs*.
          If a store cannot provide a given attribute, it should simply
          ignore it. This may even yield to an empty dictionary for a given
          *data_id*.

        The individual attributes do not have to exist in the dataset's
        metadata, they may also be generated on-the-fly.
        An example for a generic attribute name is "title".
        A store should try to resolve `include_attrs=["title"]`
        by returning items such as
        `("ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.zarr",
        {"title": "Level-4 GHRSST Analysed Sea Surface Temperature"})`.

        Args:
            data_type: If given, only data identifiers that are
                available as this type are returned. If this is omitted,
                all available data identifiers are returned.
            include_attrs: A sequence of names of attributes to be
                returned for each dataset identifier. If given, the
                store will attempt to provide the set of requested
                dataset attributes in addition to the data ids. (added
                in xcube 0.8.0)

        Returns:
            An iterator over the identifiers and titles of data
            resources provided by this data store.

        Raises:
            DataStoreError: If an error occurs.
        """

    def list_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        """Convenience version of `get_data_ids()` that returns a list rather
        than an iterator.

        Args:
            data_type: If given, only data identifiers that are
                available as this type are returned. If this is omitted,
                all available data identifiers are returned.
            include_attrs: A sequence of names of attributes to be
                returned for each dataset identifier. If given, the
                store will attempt to provide the set of requested
                dataset attributes in addition to the data ids. (added
                in xcube 0.8.0)

        Returns:
            A list comprising the identifiers and titles of data
            resources provided by this data store.

        Raises:
            DataStoreError: If an error occurs.
        """
        return list(self.get_data_ids(data_type=data_type, include_attrs=include_attrs))

    @abstractmethod
    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        """Check if the data resource given by *data_id* is
        available in this store.

        Args:
            data_id: A data identifier
            data_type: An optional data type. If given, it will also be
                checked whether the data is available as the specified
                type. May be given as type alias name, as a type, or as
                a `DataType` instance.

        Returns:
            True, if the data resource is available in this store, False
            otherwise.
        """

    @abstractmethod
    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DataDescriptor:
        """Get the descriptor for the data resource given by *data_id*.

        Raises a `DataStoreError` if *data_id* does not
        exist in this store or the data is not available as the
        specified *data_type*.

        Args:
            data_id: An identifier of data provided by this store
            data_type: If given, the descriptor of the data will
                describe the data as specified by the data type. May be
                given as type alias name, as a type, or as a
                `DataType` instance.

        Returns:
            a data-type specific data descriptor

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        """Get identifiers of data openers that can be used to open data
        resources from this store.

        If *data_id* is given, data accessors are restricted to the ones
        that can open the identified data resource.
        Raises if *data_id* does not exist in this store.

        If *data_type* is given, only openers that are compatible with
        this data type are returned.

        If a store implementation supports only a single data type,
        it should verify that *data_type* is either None or equal to
        that single data type.

        Args:
            data_id: An optional data resource identifier that is known
                to exist in this data store.
            data_type: An optional data type that is known to be
                supported by this data store. May be given as type alias
                name, as a type, or as a `DataType` instance.

        Returns:
            A tuple of identifiers of data openers that can be used to
            open data resources.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *open_params* to
        `open_data(data_id, open_params)`.

        If *data_id* is given, the returned schema will be tailored
        to the constraints implied by the identified data resource.
        Some openers might not support this, therefore *data_id* is optional,
        and if it is omitted, the returned schema will be less restrictive.
        If given, the method raises if *data_id* does not exist in this store.

        If *opener_id* is given, the returned schema will be tailored to
        the constraints implied by the identified opener. Some openers
        might not support this, therefore *opener_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        For maximum compatibility of stores, it is strongly encouraged to
        apply the following conventions on parameter names, types,
        and their interpretation.

        Let P be the value of an optional, data constraining open parameter,
        then it should be interpreted as follows:

          * _if P is None_ means, parameter not given,
            hence no constraint applies, hence no additional restrictions
            on requested data.
          * _if not P_ means, we exclude data that would be
            included by default.
          * _else_, the given constraint applies.

        Given here are names, types, and descriptions of common,
        constraining open parameters for gridded datasets.
        Note, whether any of these is optional or mandatory depends
        on the individual data store. A store may also
        define other open parameters or support only a subset of the
        following. Note all parameters may be optional,
        the Python-types given here refer to _given_, non-Null parameters:

          * `variable_names: List[str]`: Included data variables.
            Available coordinate variables will be auto-included for
            any dimension of the data variables.
          * `bbox: Tuple[float, float, float, float]`: Spatial coverage
            as xmin, ymin, xmax, ymax.
          * `crs: str`: Spatial CRS, e.g. "EPSG:4326" or OGC CRS URI.
          * `spatial_res: float`: Spatial resolution in
            coordinates of the spatial CRS.
          * `time_range: Tuple[Optional[str], Optional[str]]`:
            Time range interval in UTC date/time units using ISO format.
            Start or end time may be missing which means everything until
            available start or end time.
          * `time_period: str`: Pandas-compatible period/frequency
            string, e.g. "8D", "2W".

        E.g. applied to an optional `variable_names` parameter, this means

          * `variable_names is None` - include all data variables
          * `variable_names == []` - do not include data variables
            (schema only)
          * `variable_names == ["<var_1>", "<var_2>", ...]` only
            include data variables named "<var_1>", "<var_2>", ...

        Args:
            data_id: An optional data identifier that is known to exist
                in this data store.
            opener_id: An optional data opener identifier.

        Returns:
            The schema for the parameters in *open_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def open_data(self, data_id: str, opener_id: str = None, **open_params) -> Any:
        """Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        The data type of the return value depends on the data opener
        used to open the data resource.

        If *opener_id* is given, the identified data opener will be used
        to open the data resource and *open_params* must comply with the
        schema of the opener's parameters. Note that some store
        implementations may not support using different openers or just
        support a single one.

        Raises if *data_id* does not exist in this store.

        Args:
            data_id: The data identifier that is known to exist in this
                data store.
            opener_id: An optional data opener identifier.
            **open_params: Opener-specific parameters.

        Returns:
            An in-memory representation of the data resources identified
            by *data_id* and *open_params*.

        Raises:
            DataStoreError: If an error occurs.
        """


class MutableDataStore(DataStore, DataWriter, ABC):
    """A mutable data store is a data store that also allows for adding,
    updating, and removing data resources.

    MutableDataStore is an abstract base class that any mutable data
    store must implement.
    """

    @abstractmethod
    def get_data_writer_ids(self, data_type: DataTypeLike = None) -> Tuple[str, ...]:
        """Get identifiers of data writers that can be used to write data
        resources to this store.

        If *data_type* is given, only writers that support this
        data type are returned.

        If a store implementation supports only a single data type,
        it should verify that *data_type* is either None or equal to
        that single data type.

        Args:
            data_type: An optional data type specifier that is known to
                be supported by this data store. May be given as type
                alias name, as a type, or as a `DataType` instance.

        Returns:
            A tuple of identifiers of data writers that can be used to
            write data resources.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *write_params* to
        `write_data(data, data_id, open_params)`.

        If *writer_id* is given, the returned schema will be tailored
        to the constraints implied by the identified writer.
        Some writers might not support this, therefore *writer_id*
        is optional, and if it is omitted, the returned schema will
        be less restrictive.

        Given here is a pseudo-code implementation for stores that support
        multiple writers and where the store has common parameters with
        the writer:

        ```python
            store_params_schema = self.get_data_store_params_schema()
            writer_params_schema = get_writer(writer_id).get_write_data_params_schema()
            return subtract_param_schemas(writer_params_schema, store_params_schema)
        ```

        Args:
            writer_id: An optional data writer identifier.

        Returns:
            The schema for the parameters in *write_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def write_data(
        self,
        data: Any,
        data_id: str = None,
        writer_id: str = None,
        replace: bool = False,
        **write_params,
    ) -> str:
        """Write a data in-memory instance using the supplied *data_id*
         and *write_params*.

        If data identifier *data_id* is not given, a writer-specific default
        will be generated, used, and returned.

        If *writer_id* is given, the identified data writer will be used to
        write the data resource and *write_params* must comply with the
        schema of writers's parameters. Note that some store
        implementations may not support using different writers or
        just support a single one.

        Given here is a pseudo-code implementation for stores that support
        multiple writers:

            writer_id = writer_id or self.gen_data_id()
            path = self.resolve_data_id_to_path(data_id)
            write_params = add_params(self.get_data_store_params(), write_params)
            get_writer(writer_id).write_data(data, path, **write_params)
            self.register_data(data_id, data)

        Raises if *data_id* does not exist in this store.

        Args:
            data: The data in-memory instance to be written.
            data_id: An optional data identifier that is known to be
                unique in this data store.
            writer_id: An optional data writer identifier.
            replace: Whether to replace an existing data resource.
            **write_params: Writer-specific parameters.

        Returns:
            The data identifier used to write the data.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def delete_data(self, data_id: str, **delete_params):
        """Delete the data resource identified by *data_id*.

        Typically, an implementation would delete the data resource
        from the physical storage and also remove any registered metadata
        from an associated database.

        Raises if *data_id* does not exist in this store.

        Args:
            data_id: An data identifier that is known to exist in this
                data store.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def register_data(self, data_id: str, data: Any):
        """Register the in-memory representation of a data resource *data*
        using the given data resource identifier *data_id*.

        This method can be used to register data resources that are
        already physically stored in the data store, but are not yet
        searchable or otherwise accessible by the given *data_id*.

        Typically, an implementation would extract metadata from
        *data* and store it in a store-specific database.
        An implementation should just store the metadata of *data*.
        It should not write *data*.

        Args:
            data_id: A data resource identifier that is known to be
                unique in this data store.
            data: An in-memory representation of a data resource.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def deregister_data(self, data_id: str):
        """De-register a data resource identified by *data_id* from
        this data store.

        This method can be used to de-register data resources so it
        will be no longer searchable or otherwise accessible by
        the given *data_id*.

        Typically, an implementation would extract metadata from
        *data* and store it in a store-specific database.
        An implementation should only remove a data resource's metadata.
        It should not delete *data* from its physical storage space.

        Raises if *data_id* does not exist in this store.

        Args:
            data_id: A data resource identifier that is known to exist
                in this data store.

        Raises:
            DataStoreError: If an error occurs.
        """
