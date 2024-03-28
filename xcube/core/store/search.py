from abc import abstractmethod, ABC
from typing import Iterator

from xcube.util.jsonschema import JsonObjectSchema
from .assertions import assert_valid_params
from .datatype import DataTypeLike
from .descriptor import DataDescriptor


class DataSearcher(ABC):
    """Allow searching data in a data store."""

    @classmethod
    @abstractmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        """Get the schema for the parameters that can be passed
        as *search_params* to `search_data()`.
        Parameters are named and described by the properties of the
        returned JSON object schema.

        Args:
            data_type: If given, the search parameters will be tailored
                to search for data for the given *data_type*.

        Returns:
            A JSON object schema whose properties describe this store's
            search parameters.
        """

    @abstractmethod
    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        """Search this store for data resources.
        If *data_type* is given, the search is restricted
        to data resources of that type.

        Returns an iterator over the search results which are
        returned as `DataDescriptor` objects.

        If a store implementation supports only a single data type,
        it should verify that *data_type*
        is either None or compatible with the supported data type
        specifier.

        Args:
            data_type: An optional data type that is known to be
                supported by this data store.
            search_params: The search parameters.

        Returns:
            An iterator of data descriptors for the found data
            resources.

        Raises:
            DataStoreError: If an error occurs.
        """


# noinspection PyUnresolvedReferences
class DefaultSearchMixin(DataSearcher):
    """A mixin for data store implementations that implements default
    search behaviour.

    It is expected that such data stores have no dedicated search
    parameters.
    """

    # noinspection PyUnusedLocal
    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        """Get search parameters JSON object schema.

        The default implementation returns a schema that does
        not allow for any search parameters.

        Args:
            data_type

        Returns:
            a JSON object schema for the search parameters
        """
        return JsonObjectSchema(additional_properties=False)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        """Search the data store.

        The default implementation returns all data resources that
        may be filtered using the optional *data_type*.

        Args:
            data_type: Type specifier to filter returned data resources.
            **search_params: Not supported (yet)

        Returns:
            an iterator of `DataDescriptor` instances
        """
        search_params_schema = self.get_search_params_schema(data_type=data_type)
        assert_valid_params(
            search_params, name="search_params", schema=search_params_schema
        )
        for data_id in self.get_data_ids(data_type=data_type):
            data_descriptor = self.describe_data(data_id, data_type=data_type)
            yield data_descriptor
