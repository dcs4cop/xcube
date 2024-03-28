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

import os
from abc import ABC, abstractmethod

from typing import Sequence, Tuple, Iterator, Generic, TypeVar, Any
from typing import Type, List
from typing import Union

import geopandas
import xarray

from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonStringSchema

# A data type name or a DataType
DataTypeLike = Union[str, None, type, "DataType"]


class DataType:
    """A well-known Python data type that also has a representation
    using alias names.

    For example, the aliases "dataset" and "xarray.Dataset" both
    refer to the Python data type `xarray.Dataset```.

    Args:
        dtype: The Python data type.
        alias: An alias name or list of aliases.
    """

    _READTHEDOCS = os.environ.get("READTHEDOCS") == "True"
    _REGISTERED_DATA_TYPES: List["DataType"] = []

    @classmethod
    def register_data_type(cls, data_type: "DataType"):
        assert_instance(data_type, DataType, name="data_type")
        cls._REGISTERED_DATA_TYPES.append(data_type)

    def __init__(self, dtype: Type, alias: Union[None, str, Sequence[str]] = None):
        assert_instance(dtype, type if not self._READTHEDOCS else object, name="dtype")
        if alias is not None:
            assert_instance(alias, (str, tuple, list), name="alias")
        self._dtype = dtype
        self._aliases = (
            [] if alias is None else [alias] if isinstance(alias, str) else list(alias)
        ) + [self._get_fully_qualified_type_name(dtype)]
        self._alias_set = set(self._aliases)  # for faster lookup

    def __str__(self) -> str:
        return self.alias

    def __repr__(self) -> str:
        return repr(self.alias)

    def __eq__(self, other: Any):
        return self is other or (
            isinstance(other, DataType)
            and self.dtype is other.dtype
            and self.aliases == other.aliases
        )

    def __hash__(self):
        return hash(self.dtype) + 16 * hash(self.aliases)

    @property
    def dtype(self) -> Type:
        """The Python data type."""
        return self._dtype

    @property
    def alias(self) -> str:
        """The primary alias name."""
        return self._aliases[0]

    @property
    def aliases(self) -> Tuple[str, ...]:
        """All alias names."""
        return tuple(self._aliases)

    @classmethod
    def normalize(cls, data_type: DataTypeLike) -> "DataType":
        """Normalize the given *data_type* value into
        an instance of `DataType`.

        Args:
            data_type: Another data type, maybe given as type alias name,
                as a type, or as a DataType instance.
        Returns:
            a DataType instance
        """
        if isinstance(data_type, DataType):
            return data_type
        if data_type is None or data_type is type(None):
            return ANY_TYPE
        if isinstance(data_type, str):
            for dt in cls._REGISTERED_DATA_TYPES:
                if data_type in dt._alias_set:
                    return dt
            raise ValueError(f"unknown data type {data_type!r}")
        if isinstance(data_type, type):
            for dt in cls._REGISTERED_DATA_TYPES:
                if data_type is dt.dtype:
                    return dt
            return DataType(data_type)
        raise ValueError(f"cannot convert {data_type!r} into a data type")

    def is_sub_type_of(self, data_type: DataTypeLike) -> bool:
        """Tests whether this data type is a sub-type of or the same
        as another data type.

        Args:
            data_type: The other data type, may be given as type
                alias name, as a type, or as a DataType instance.
        Returns:
            Whether this data type satisfies another data type.
        """
        return issubclass(self.dtype, self._normalize_dtype(data_type))

    def is_super_type_of(self, data_type: DataTypeLike) -> bool:
        """Tests whether this data type is a super-type of or the same
        as another data type.

        Args:
            data_type: The other data type, may be given as type
                alias name, as a type, or as a DataType instance.
        Returns:
            Whether this data type satisfies another data type.
        """
        return issubclass(self._normalize_dtype(data_type), self.dtype)

    @classmethod
    def get_schema(cls) -> JsonStringSchema:
        """Get the JSON Schema."""
        return JsonStringSchema(min_length=1, factory=cls.normalize, serializer=str)

    @classmethod
    def _normalize_dtype(cls, data_type: DataTypeLike) -> Type:
        if isinstance(data_type, DataType):
            return data_type.dtype
        if isinstance(data_type, type):
            return data_type
        return cls.normalize(data_type).dtype

    @staticmethod
    def _get_fully_qualified_type_name(data_type: Type) -> str:
        return f"{data_type.__module__}.{data_type.__name__}"


T = TypeVar("T")


class DataIterator(Generic[T], Iterator[T], ABC):
    """An iterator for data items.
    This class is a marker type for data stores that can return data iterators
    from their `open_data()` method.
    Data stores that support data iterators are not required to yield
    instances of this class. They just have to return iterator objects
    that yield instances of the desired item type, e.g. `xarray.Dataset`.
    """

    @abstractmethod
    def __next__(self) -> T:
        """Yield the next data item."""


ANY_TYPE = DataType(object, ["any", "*", "object"])

DATASET_TYPE = DataType(xarray.Dataset, ["dataset", "xarray.Dataset"])

DATASET_ITERATOR_TYPE = DataType(
    DataIterator,  # Unfortunately we cannot use DataIterator[xarray.Dataset]
    ["dsiter", "DataIterator[xarray.Dataset]"],
)

MULTI_LEVEL_DATASET_TYPE = DataType(
    MultiLevelDataset,
    ["mldataset", "xcube.MultiLevelDataset", "xcube.core.mldataset.MultiLevelDataset"],
)

MULTI_LEVEL_DATASET_ITERATOR_TYPE = DataType(
    DataIterator,  # Unfortunately we cannot use DataIterator[MultiLevelDataset]
    ["mldsiter", "DataIterator[xcube.MultiLevelDataset]"],
)

GEO_DATA_FRAME_TYPE = DataType(
    geopandas.GeoDataFrame, ["geodataframe", "geopandas.GeoDataFrame"]
)

GEO_DATA_FRAME_ITERATOR_TYPE = DataType(
    DataIterator,  # Unfortunately we cannot use DataIterator[GeoDataFrame]
    ["gdfiter", "DataIterator[geopandas.GeoDataFrame]"],
)


def register_default_data_types():
    for data_type in [
        ANY_TYPE,
        DATASET_TYPE,
        DATASET_ITERATOR_TYPE,
        MULTI_LEVEL_DATASET_TYPE,
        MULTI_LEVEL_DATASET_ITERATOR_TYPE,
        GEO_DATA_FRAME_TYPE,
        GEO_DATA_FRAME_ITERATOR_TYPE,
    ]:
        DataType.register_data_type(data_type)


register_default_data_types()
