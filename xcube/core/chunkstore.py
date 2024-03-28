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

import itertools
import json
from collections.abc import MutableMapping
from typing import Dict, Tuple, Callable, Any, Sequence
from typing import Union

import numpy as np

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

from deprecated import deprecated
from xcube.constants import LOG_LEVEL_TRACE
from xcube.core.zarrstore import LoggingZarrStore
import collections.abc
from logging import Logger
from typing import Iterator, Iterable, KeysView, Optional

from xcube.constants import LOG
from xcube.util.assertions import assert_instance

GetChunk = Callable[["ChunkStore", str, Tuple[int, ...]], bytes]


@deprecated(
    reason="This class shall no longer be used."
    " If similar functionality is needed,"
    " use xcube.core.zarrstore.GenericZarrStore",
    version="0.12.1",
)
class ChunkStore(MutableMapping):
    """A Zarr Store that generates datasets by allowing data variables to
    fetch or compute their chunks by a user-defined function *get_chunk*.
    Implements the standard Python `MutableMapping` interface.

    This is how the *get_chunk* function is called:

    ```python
        data = get_chunk(chunk_store, var_name, chunk_indexes)
    ```

    where `chunk_store` is this store, `var_name` is the name of
    the variable for which data is fetched, and `chunk_indexes` is
    a tuple of zero-based, integer chunk indexes. The result must be
    a Python *bytes* object.

    Args:
        dims: Dimension names of all data variables,
            e.g. ('time', 'lat', 'lon').
        shape: Shape of all data variables according to *dims*,
            e.g. (512, 720, 1480).
        chunks: Chunk sizes of all data variables according to *dims*,
            e.g. (128, 180, 180).
        attrs: Global dataset attributes.
        get_chunk: Default chunk fetching/computing function.
        trace_store_calls: Whether to log calls
            into the `MutableMapping` interface.
    """

    def __init__(
        self,
        dims: Sequence[str],
        shape: Sequence[int],
        chunks: Sequence[int],
        attrs: Dict[str, Any] = None,
        get_chunk: GetChunk = None,
        trace_store_calls: bool = False,
    ):
        self._ndim = len(dims)
        self._dims = tuple(dims)
        self._shape = tuple(shape)
        self._chunks = tuple(chunks)
        self._get_chunk = get_chunk
        self._trace_store_calls = trace_store_calls

        # setup Virtual File System (vfs)
        self._vfs = {
            ".zgroup": _dict_to_bytes(dict(zarr_format=2)),
            ".zattrs": _dict_to_bytes(attrs or dict()),
        }

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def chunks(self) -> Tuple[int, ...]:
        return self._chunks

    def add_array(self, name: str, array: np.ndarray, attrs: Dict):
        shape = list(map(int, array.shape))
        dtype = str(array.dtype.str)
        array_metadata = {
            "zarr_format": 2,
            "chunks": shape,
            "shape": shape,
            "dtype": dtype,
            "fill_value": None,
            "compressor": None,
            "filters": None,
            "order": "C",
        }
        self._vfs[name] = _str_to_bytes("")
        self._vfs[name + "/.zarray"] = _dict_to_bytes(array_metadata)
        self._vfs[name + "/.zattrs"] = _dict_to_bytes(attrs)
        self._vfs[name + "/" + (".".join(["0"] * array.ndim))] = bytes(array)

    def add_lazy_array(
        self,
        name: str,
        dtype: str,
        fill_value: Union[int, float] = None,
        compressor: Dict[str, Any] = None,
        filters=None,
        order: str = "C",
        attrs: Dict[str, Any] = None,
        get_chunk: GetChunk = None,
    ):
        get_chunk = get_chunk or self._get_chunk
        if get_chunk is None:
            raise ValueError("get_chunk must be given as there is no default")

        array_metadata = dict(
            zarr_format=2,
            shape=self._shape,
            chunks=self._chunks,
            compressor=compressor,
            dtype=dtype,
            fill_value=fill_value,
            filters=filters,
            order=order,
        )

        self._vfs[name] = _str_to_bytes("")
        self._vfs[name + "/.zarray"] = _dict_to_bytes(array_metadata)
        self._vfs[name + "/.zattrs"] = _dict_to_bytes(
            dict(_ARRAY_DIMENSIONS=self._dims, **(attrs or dict()))
        )

        nums = np.array(self._shape) // np.array(self._chunks)
        indexes = itertools.product(*tuple(map(range, map(int, nums))))
        for index in indexes:
            filename = ".".join(map(str, index))
            # noinspection PyTypeChecker
            self._vfs[name + "/" + filename] = name, index, get_chunk

    @property
    def _class_name(self):
        return self.__module__ + "." + self.__class__.__name__

    ##########################################################################
    # Zarr Store (MutableMapping) implementation
    ##########################################################################

    def keys(self) -> KeysView[str]:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.keys()")
        return self._vfs.keys()

    def listdir(self, key: str) -> Iterable[str]:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.listdir(key={key!r})")
        if key == "":
            return (k for k in self._vfs.keys() if "/" not in k)
        else:
            prefix = key + "/"
            start = len(prefix)
            return (
                k
                for k in self._vfs.keys()
                if k.startswith(prefix) and k.find("/", start) == -1
            )

    def getsize(self, key: str) -> int:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.getsize(key={key!r})")
        return len(self._vfs[key])

    def __iter__(self) -> Iterator[str]:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__iter__()")
        return iter(self._vfs.keys())

    def __len__(self) -> int:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__len__()")
        return len(self._vfs.keys())

    def __contains__(self, key) -> bool:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__contains__(key={key!r})")
        return key in self._vfs

    def __getitem__(self, key: str) -> bytes:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__getitem__(key={key!r})")
        value = self._vfs[key]
        if isinstance(value, tuple):
            name, index, get_chunk = value
            return get_chunk(self, name, index)
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__setitem__(key={key!r}, value={value!r})")
        raise TypeError(f"{self._class_name} is read-only")

    def __delitem__(self, key: str) -> None:
        if self._trace_store_calls:
            _trace(f"{self._class_name}.__delitem__(key={key!r})")
        raise TypeError(f"{self._class_name} is read-only")


def _trace(msg: str):
    LOG.log(LOG_LEVEL_TRACE, msg)


def _dict_to_bytes(d: Dict):
    return _str_to_bytes(json.dumps(d, indent=2))


def _str_to_bytes(s: str):
    return bytes(s, encoding="utf-8")


@deprecated(
    reason="This class has been moved,"
    " use xcube.core.zarrstore.LoggingZarrStore"
    " instead.",
    version="0.12.1",
)
class LoggingStore(LoggingZarrStore):
    """A Zarr Store that logs all method calls on another store *other*
    including execution time.
    """

    @classmethod
    def new(
        cls,
        other: collections.abc.MutableMapping,
        logger: Logger = LOG,
        name: Optional[str] = None,
    ):
        assert_instance(other, collections.abc.MutableMapping)
        return cls(other, logger=logger, name=name)


@deprecated(
    reason="This class has been moved,"
    " use xcube.core.zarrstore.LoggingZarrStore"
    " instead.",
    version="0.12.1",
)
class MutableLoggingStore(LoggingStore):
    """Mutable version of `LoggingStore`."""
