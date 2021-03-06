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

from typing import Any
from typing import Optional
from typing import Set
from typing import Union

import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.verify import verify_cube
from xcube.util.assertions import assert_given, assert_condition
from xcube.util.jsonschema import JsonStringSchema


class TypeSpecifier:
    """
    A type specifier denotes a type of data. It is used to group similar types of data and
    distinguish between different types of data. It can be used by stores to state what
    types of data can be read from and/or written to them.

    A type specifier consists of a name and an arbitrary number of flags.
    Flags can be used to further refine a type specifier if needed.

    :param name: The name of the type of data
    :param flags: An arbitrary set of flags that further refine the type
    """

    def __init__(self, name: str, flags: Set[str] = None):
        assert_given(name, 'name')
        if name == '*' and flags:
            raise ValueError('flags are not allowed if name is "*" (any)')
        self._name = name
        self._flags = flags if flags is not None else set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def flags(self) -> Set[str]:
        return self._flags

    def __str__(self) -> str:
        if len(self.flags) == 0:
            return self.name
        flag_part = ','.join(sorted(self.flags))
        return f'{self.name}[{flag_part}]'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        try:
            other_type = self.normalize(other)
        except TypeError:
            return False
        if self.name != other_type.name:
            return False
        return self.flags == other_type.flags

    def __hash__(self) -> int:
        return hash(self.name) + 16 * hash(frozenset(self.flags))

    def satisfies(self, other: Union[str, "TypeSpecifier"]) -> bool:
        """
        Tests whether this type specifier satisfies (the requirements of) another type specifier.

        This type specifier satisfies type specifier *other*

        1. if either this or *other* is "*" (= any) or
        2. if the type names of this and *other* are equal and
          - if *other* has no flags or
          - if all the flags of *other* are a subset of the flags (if any) of this type specifier.

        :param other: Another type specifier, as string or *TypeSpecifier*.
        :return: Whether this type specifier satisfies another type specifier.
        """
        other_type = self.normalize(other)
        if self.name == '*' or other_type.name == '*':
            return True
        if self.name != other_type.name:
            return False
        if not other_type.flags:
            return True
        return other_type.flags.issubset(self.flags)

    def is_satisfied_by(self, other: Union[str, "TypeSpecifier"]) -> bool:
        """
        Tests whether this type specifier is satisfied by another type specifier.

        This is the inverse operation of :meth:satisfies and may be more handy
        or intuitive in some situations. It is equivalent to:

            self.normalize(other).satisfies(self)

        :param other: Another type specifier, as string or *TypeSpecifier*.
        :return: Whether this type specifier satisfies another type specifier.
        """
        return self.normalize(other).satisfies(self)

    @classmethod
    def normalize(cls, type_specifier: Union[str, "TypeSpecifier"]) -> "TypeSpecifier":
        if isinstance(type_specifier, TypeSpecifier):
            return type_specifier
        if isinstance(type_specifier, str):
            return cls.parse(type_specifier)
        raise TypeError('type_specifier must be of type "str" or "TypeSpecifier"')

    @classmethod
    def parse(cls, type_specifier: str) -> "TypeSpecifier":
        if '[' not in type_specifier:
            return TypeSpecifier(type_specifier)
        if not type_specifier.endswith(']'):
            raise SyntaxError(f'"{type_specifier}" cannot be parsed: No end brackets found')
        name = type_specifier.split('[')[0]
        flags = type_specifier.split('[')[1].split(']')[0].split(',')
        return TypeSpecifier(name, flags=set(flags))

    @classmethod
    def get_schema(cls) -> JsonStringSchema:
        return JsonStringSchema(
            min_length=1,
            factory=TypeSpecifier.parse,
            serializer=str
        )

    def assert_satisfies(self, other: Union[str, 'TypeSpecifier'], name: str = None):
        assert_condition(self.satisfies(other),
                         f'{name or "type_specifier"} must satisfy type specifier "{other}",'
                         f' but was "{self}"')


TYPE_SPECIFIER_ANY = TypeSpecifier('*')
TYPE_SPECIFIER_DATASET = TypeSpecifier('dataset')
TYPE_SPECIFIER_CUBE = TypeSpecifier('dataset', flags={'cube'})
TYPE_SPECIFIER_MULTILEVEL_DATASET = TypeSpecifier('dataset', flags={'multilevel'})
TYPE_SPECIFIER_MULTILEVEL_CUBE = TypeSpecifier('dataset', flags={'multilevel', 'cube'})
TYPE_SPECIFIER_GEODATAFRAME = TypeSpecifier('geodataframe')


def get_type_specifier(data: Any) -> Optional[TypeSpecifier]:
    if isinstance(data, MultiLevelDataset):
        if not verify_cube(data.get_dataset(0)):
            return TYPE_SPECIFIER_MULTILEVEL_CUBE
        return TYPE_SPECIFIER_MULTILEVEL_DATASET
    if isinstance(data, xr.Dataset):
        if not verify_cube(data):
            return TYPE_SPECIFIER_CUBE
        return TYPE_SPECIFIER_DATASET
    if isinstance(data, gpd.GeoDataFrame):
        return TYPE_SPECIFIER_GEODATAFRAME
    return None
