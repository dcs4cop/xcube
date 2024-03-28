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

import collections.abc
from abc import abstractmethod, ABC
from typing import Generic, TypeVar, Tuple, Any, Iterable, Dict, List, Sequence, Mapping

K = TypeVar("K")
V = TypeVar("V")

_DICT_IS_READONLY = "dict is read-only"
_LIST_IS_READONLY = "list is read-only"


class Frozen(ABC):
    """A frozen, that is, an immutable object."""

    @classmethod
    @abstractmethod
    def freeze(cls, obj: Any) -> "Frozen":
        """Deeply freeze object *obj*, i.e.,
        return an immutable version of it.
        """

    @abstractmethod
    def defrost(self) -> Any:
        """Deeply defrost this frozen object, i.e.,
        return its mutable counterpart.
        """


class FrozenDict(Dict[K, V], Frozen, Generic[K, V]):
    """A frozen version of a standard `dict`."""

    @classmethod
    def freeze(cls, other: Mapping) -> "FrozenDict":
        return FrozenDict({key: freeze_value(value) for key, value in other.items()})

    def defrost(self) -> Dict[K, V]:
        return {key: defrost_value(value) for key, value in self.items()}

    ###########################################################
    # dict overrides

    def clear(self) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def pop(self, *args, **kwargs) -> V:
        raise TypeError(_DICT_IS_READONLY)

    def popitem(self) -> Tuple[K, V]:
        raise TypeError(_DICT_IS_READONLY)

    def update(self, *args, **kwargs) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __setitem__(self, key: K, value: V) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __delitem__(self, key: K) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __delattr__(self, name: str) -> None:
        raise TypeError(_DICT_IS_READONLY)


class FrozenList(List[V], Frozen, Generic[V]):
    """A frozen version of a standard `list`."""

    @classmethod
    def freeze(cls, other: Sequence[V]) -> "FrozenList":
        return FrozenList(freeze_value(item) for item in other)

    def defrost(self) -> List[V]:
        return [defrost_value(item) for item in self]

    ###########################################################
    # list overrides

    def append(self, element: V):
        raise TypeError(_LIST_IS_READONLY)

    def extend(self, elements: Iterable[V]):
        raise TypeError(_LIST_IS_READONLY)

    def clear(self):
        raise TypeError(_LIST_IS_READONLY)

    def insert(self, index: int, elements: Iterable[V]):
        raise TypeError(_LIST_IS_READONLY)

    def pop(self, *args, **kwargs):
        raise TypeError(_LIST_IS_READONLY)

    def remove(self, value: V):
        raise TypeError(_LIST_IS_READONLY)

    def reverse(self):
        raise TypeError(_LIST_IS_READONLY)

    def sort(self, *args, **kwargs):
        raise TypeError(_LIST_IS_READONLY)

    def __iadd__(self, arg: Any):
        raise TypeError(_LIST_IS_READONLY)

    def __imul__(self, arg: Any):
        raise TypeError(_LIST_IS_READONLY)

    def __setitem__(self, index: int, value: V):
        raise TypeError(_LIST_IS_READONLY)


# Standard Python immutables
_IMMUTABLES = type(None), bool, int, float, complex, str, bytes


def freeze_value(value: Any) -> Any:
    """Freeze given *value*, that is, return a deeply
    immutable version of it."""
    # Note, _IMMUTABLES also includes str and bytes which are
    # both a collections.abc.Sequence.
    if isinstance(value, _IMMUTABLES):
        return value
    if isinstance(value, collections.abc.Mapping):
        return FrozenDict.freeze(value)
    if isinstance(value, collections.abc.Sequence):
        return FrozenList.freeze(value)
    return value


def defrost_value(value: Any) -> Any:
    """Defrost given *value*, that is, return a deeply
    mutable version of it.
    """
    if isinstance(value, Frozen):
        return value.defrost()
    return value
