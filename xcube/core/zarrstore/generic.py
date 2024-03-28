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
import inspect
import itertools
import json
import math
import threading
import warnings
from typing import Iterator, Dict, Tuple, Any, Callable, Optional, List, Sequence
from typing import Union

import numcodecs.abc
import numpy as np
import xarray as xr
import zarr.storage

from xcube.util.assertions import assert_instance, assert_true

GetData = Callable[[Tuple[int]], Union[bytes, np.ndarray]]

OnClose = Callable[[Dict[str, Any]], None]


class GenericArray(Dict[str, any]):
    """Represent a generic array in the `GenericZarrStore` as
    dictionary of properties.

    Although all properties of this class are optional,
    some of them are mandatory when added to the `GenericZarrStore`.

    When added to the store using `GenericZarrStore.add_array()`,
    the array *name* and *dims* must always be present.
    Other mandatory properties depend on
    the *data* and *get_data* properties, which are mutually exclusive:

    * *get_data* is called for a requested data chunk of an array.
      It must return a bytes object or a numpy nd-array and is passed
      the chunk index, the chunk shape, and this array info dictionary.
      *get_data* requires the following properties to be present too:
      *name*, *dims*, *dtype*, *shape*.
      *chunks* is optional and defaults to *shape*.
    * *data* must be a bytes object or a numpy nd-array.
      *data* requires the following properties to be present too:
      *name*, *dims*. *chunks* must be same as *shape*.

    The function *get_data* receives only keyword-arguments which
    comprises the ones passed by *get_data_params*, if any, and
    two special ones which may occur in the signature of *get_data*:

    * The keyword argument *chunk_info*, if given, provides a dictionary
      that holds information about the current chunk:
      - `index: tuple[int, ...]` - the chunk's index
      - `shape: tuple[int, ...]` - the chunk's shape
      - `slices: tuple[slice, ...]` - the chunk's array slices

    * The keyword argument *array_info*, if given, provides a dictionary
      that holds information about the overall array. It contains
      all array properties passed to the constructor of `GenericArray`
      plus
      - `ndim: int` - number of dimensions
      - `num_chunks: tuple[int, ...]` - number of chunks in every dimension

    `GenericZarrStore` will convert a Numpy array returned
    by *get_data* or given by *data* into a bytes object.
    It will also be compressed, if a *compressor* is given.
    It is important that the array chunks always See also
    https://zarr.readthedocs.io/en/stable/spec/v2.html#chunks

    Note that if the value of a named keyword argument is None,
    it will not be stored.

    Args:
        array: Optional array info dictionary
        name: Optional array name
        data: Optional array data. Mutually exclusive with *get_data*.
            Must be a bytes object or a numpy array.
        get_data: Optional array data chunk getter. Mutually exclusive
            with *data*. Called for a requested data chunk of an array.
            Must return a bytes object or a numpy array.
        get_data_params: Optional keyword-arguments passed to
            *get_data*.
        dtype: Optional array data type. Either a string using syntax of
            the Zarr spec or a `numpy.dtype`. For string encoded data
            types, see
            https://zarr.readthedocs.io/en/stable/spec/v2.html#data-
            type-encoding
        dims: Optional sequence of dimension names.
        shape: Optional sequence of shape sizes for each dimension.
        chunks: Optional sequence of chunk sizes for each dimension.
        fill_value: Optional fill value, see
            https://zarr.readthedocs.io/en/stable/spec/v2.html#fill-
            value-encoding
        compressor: Optional compressor. If given, it must be an
            instance of `numcodecs.abc.Codec`.
        filters: Optional sequence of filters, see
            https://zarr.readthedocs.io/en/stable/spec/v2.html#filters.
        order: Optional array endian ordering. If given, must be "C" or
            "F". Defaults to "C".
        attrs: Optional array attributes. If given, must be JSON-
            serializable.
        on_close: Optional array close handler. Called if the store is
            closed.
        chunk_encoding: Optional encoding type of the chunk data
            returned for the array. Can be "bytes" (the default) or
            "ndarray" for array chunks that are numpy.ndarray instances.
        kwargs: Other keyword arguments passed directly to the
            dictionary constructor.
    """

    def __init__(
        self,
        array: Optional[Dict[str, any]] = None,
        name: Optional[str] = None,
        get_data: Optional[GetData] = None,
        get_data_params: Optional[Dict[str, Any]] = None,
        data: Optional[np.ndarray] = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        dims: Optional[Union[str, Sequence[str]]] = None,
        shape: Optional[Sequence[int]] = None,
        chunks: Optional[Sequence[int]] = None,
        fill_value: Optional[Union[bool, int, float, str]] = None,
        compressor: Optional[numcodecs.abc.Codec] = None,
        filters: Optional[Sequence[numcodecs.abc.Codec]] = None,
        order: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        on_close: Optional[OnClose] = None,
        chunk_encoding: Optional[str] = None,
        **kwargs,
    ):
        array = dict(array) if array is not None else dict()
        array.update(
            {
                k: v
                for k, v in dict(
                    name=name,
                    dtype=dtype,
                    dims=dims,
                    shape=shape,
                    chunks=chunks,
                    fill_value=fill_value,
                    compressor=compressor,
                    filters=filters,
                    order=order,
                    attrs=attrs,
                    data=data,
                    get_data=get_data,
                    get_data_params=get_data_params,
                    on_close=on_close,
                    chunk_encoding=chunk_encoding,
                ).items()
                if v is not None
            }
        )
        super().__init__(array, **kwargs)

    def finalize(self) -> "GenericArray":
        """Normalize and validate array properties and return a valid
        array info dictionary to be stored in the `GenericZarrStore`.
        """
        name = self.get("name")
        if not name:
            raise ValueError("missing array name")

        data = self.get("data")
        get_data = self.get("get_data")
        if data is None and get_data is None:
            raise ValueError(
                f"array {name!r}:" f" either data or get_data must be defined"
            )
        if get_data is not None:
            if data is not None:
                raise ValueError(
                    f"array {name!r}:"
                    f" data and get_data cannot"
                    f" be defined together"
                )
            if not callable(get_data):
                raise TypeError(f"array {name!r}:" f" get_data must be a callable")
            sig = inspect.signature(get_data)
            get_data_info = {
                "has_array_info": "array_info" in sig.parameters,
                "has_chunk_info": "chunk_info" in sig.parameters,
            }
            get_data_params = dict(self.get("get_data_params") or {})
        else:
            get_data_info = None
            get_data_params = None

        dims = self.get("dims")
        dims = [dims] if isinstance(dims, str) else dims
        if dims is None:
            raise ValueError(f"array {name!r}: missing dims")

        ndim = len(dims)

        if isinstance(data, np.ndarray):
            # forman: maybe warn if dtype or shape is given,
            #  but does not match data.dtype and data.shape
            dtype = str(data.dtype.str)
            shape = data.shape
            chunks = data.shape
        else:
            dtype = self.get("dtype")
            shape = self.get("shape")
            chunks = self.get("chunks", shape)

        if not dtype:
            raise ValueError(f"array {name!r}: missing dtype")
        elif isinstance(dtype, np.dtype):
            dtype = dtype.str

        if shape is None:
            raise ValueError(f"array {name!r}: missing shape")
        if len(shape) != ndim:
            raise ValueError(
                f"array {name!r}:" f" dims and shape must have same length"
            )
        if len(chunks) != ndim:
            raise ValueError(
                f"array {name!r}:" f" dims and chunks must have same length"
            )

        num_chunks = tuple(map(lambda x: math.ceil(x[0] / x[1]), zip(shape, chunks)))

        filters = self.get("filters")
        if filters:
            filters = list(filters)
            for f in filters:
                if not isinstance(f, numcodecs.abc.Codec):
                    raise TypeError(
                        f"array {name!r}:"
                        f" filter items must be an"
                        f" instance of numcodecs.abc.Codec"
                    )
        else:
            filters = None

        compressor = self.get("compressor")
        if compressor is not None:
            if not isinstance(compressor, numcodecs.abc.Codec):
                raise TypeError(
                    f"array {name!r}:"
                    f" compressor must be an"
                    f" instance of numcodecs.abc.Codec"
                )

        fill_value = self.get("fill_value")
        if isinstance(fill_value, np.ndarray):
            fill_value = fill_value.item()
        allowed_fill_value_types = (type(None), bool, int, float, str)
        if not isinstance(fill_value, allowed_fill_value_types):
            raise TypeError(
                f"array {name!r}:"
                f" fill_value type must be one of"
                f" {tuple(t.__name__ for t in allowed_fill_value_types)},"
                f" was {type(fill_value).__name__}"
            )

        order = self.get("order") or "C"
        allowed_orders = ("C", "F")
        if order not in allowed_orders:
            raise ValueError(
                f"array {name!r}:"
                f" order must be one of {allowed_orders},"
                f" was {order!r}"
            )

        chunk_encoding = self.get("chunk_encoding") or "bytes"
        allowed_chunk_encodings = ("bytes", "ndarray")
        if chunk_encoding not in allowed_chunk_encodings:
            raise ValueError(
                f"array {name!r}:"
                f" chunk_encoding must be one of {allowed_chunk_encodings},"
                f" was {chunk_encoding!r}"
            )

        attrs = self.get("attrs")
        if attrs is not None:
            if not isinstance(attrs, dict):
                raise TypeError(
                    f"array {name!r}:"
                    f" attrs must be dict, was {type(attrs).__name__}"
                )

        # Note: passing the properties as dictionary
        # will prevent removing them if their value is None,
        # see GenericArray constructor.
        return GenericArray(
            {
                "name": name,
                "dtype": dtype,
                "dims": tuple(dims),
                "shape": tuple(shape),
                "chunks": tuple(chunks),
                "fill_value": fill_value,
                "filters": filters,
                "compressor": compressor,
                "order": order,
                "attrs": attrs,
                "data": data,
                "get_data": get_data,
                "get_data_params": get_data_params,
                "on_close": self.get("on_close"),
                "chunk_encoding": chunk_encoding,
                # Computed properties
                "ndim": len(dims),
                "num_chunks": num_chunks,
                "get_data_info": get_data_info,
            }
        )


GenericArrayLike = Union[GenericArray, Dict[str, Any]]


class GenericZarrStore(zarr.storage.Store):
    """A Zarr store that maintains generic arrays in a flat, top-level
    hierarchy. The root of the store is a Zarr group
    conforming to the Zarr spec v2.

    It is designed to serve as a Zarr store for xarray datasets
    that compute their data arrays dynamically.

    See class `GenericArray` for specifying the arrays' properties.

    The array data of this store's arrays are either retrieved from
    static (numpy) arrays or from a callable that provides the
    array's data chunks as bytes or numpy arrays.

    Args:
        arrays: Arrays to be added.
            Typically, these will be instances of `GenericArray`.
        attrs: Optional attributes of the top-level group.
            If given, it must be JSON serializable.
        array_defaults: Optional array defaults for
            array properties not passed to `add_array`.
            Typically, this will be an instance of `GenericArray`.
    """

    # Shortcut for GenericArray
    Array = GenericArray

    def __init__(
        self,
        *arrays: GenericArrayLike,
        attrs: Optional[Dict[str, Any]] = None,
        array_defaults: Optional[GenericArrayLike] = None,
    ):
        self._attrs = dict(attrs) if attrs is not None else {}
        self._array_defaults = array_defaults
        self._dim_sizes: Dict[str, int] = {}
        self._arrays: Dict[str, GenericArray] = {}
        for array in arrays:
            self.add_array(array)

    def add_array(
        self, array: Optional[GenericArrayLike] = None, **array_kwargs
    ) -> None:
        """
        Add a new array to this store.

        array: Optional array properties.
            Typically, this will be an instance of `GenericArray`.
        array_kwargs: Keyword arguments form
            for the properties of `GenericArray`.
        """
        effective_array = GenericArray(self._array_defaults or {})
        if array:
            effective_array.update(array)
        if array_kwargs:
            effective_array.update(array_kwargs)
        effective_array = effective_array.finalize()

        name = effective_array["name"]
        if name in self._arrays:
            raise ValueError(f"array {name!r} is already defined")

        dims = effective_array["dims"]
        shape = effective_array["shape"]
        for dim_name, dim_size in zip(dims, shape):
            old_dim_size = self._dim_sizes.get(dim_name)
            if old_dim_size is None:
                self._dim_sizes[name] = dim_size
            elif old_dim_size != dim_size:
                # Dimensions must have same lengths for all arrays
                # in this store
                raise ValueError(
                    f"array {name!r}"
                    f" defines dimension {dim_name!r}"
                    f" with size {dim_size},"
                    f" but existing size is {old_dim_size}"
                )

        self._arrays[name] = effective_array

    ##########################################################################
    # Zarr Store implementation
    ##########################################################################

    def is_writeable(self) -> bool:
        """Return False, because arrays in this store are generative."""
        return False

    def listdir(self, path: str = "") -> List[str]:
        """List a store path.

        Args:
            path: The path.

        Returns:
            List of sorted directory entries.
        """
        if path == "":
            return sorted([".zmetadata", ".zgroup", ".zattrs", *self._arrays.keys()])
        elif "/" not in path:
            return sorted(self._get_array_keys(path))
        raise ValueError(f"{path} is not a directory")

    def rmdir(self, path: str = "") -> None:
        """The general form removes store paths.
        This implementation can remove entire arrays only.
        path: The array's name.
        """
        if path not in self._arrays:
            raise ValueError(f"{path}: can only remove existing arrays")
        array = self._arrays.pop(path)
        dims = array["dims"]
        for i, dim_name in enumerate(dims):
            dim_used = False
            for array_name, array in self._arrays.items():
                if dim_name in array["dims"]:
                    dim_used = True
                    break
            if not dim_used:
                del self._dim_sizes[dim_name]

    def rename(self, src_path: str, dst_path: str) -> None:
        """The general form renames store paths.
        This implementation can rename arrays only.

        src_path: Source array name.
        dst_path: Target array name.
        """
        array = self._arrays.get(src_path)
        if array is None:
            raise ValueError(
                f"can only rename arrays, but {src_path!r}" f" is not an array"
            )
        if dst_path in self._arrays:
            raise ValueError(
                f"cannot rename array {src_path!r} into"
                f" {dst_path!r} because it already exists"
            )
        if "/" in dst_path:
            raise ValueError(f"cannot rename array {src_path!r}" f" into {dst_path!r}")
        array["name"] = dst_path
        self._arrays[dst_path] = array
        del self._arrays[src_path]

    def close(self) -> None:
        """Calls the "on_close" handlers, if any, of arrays."""
        for array in self._arrays.values():
            on_close = array.get("on_close")
            if on_close is not None:
                on_close(array)

    # Note, getsize is not implemented by intention as it requires
    # actual computation of arrays.
    #
    # def getsize(self, key: str) -> int:
    #    pass

    ##########################################################################
    # MutableMapping implementation
    ##########################################################################

    def __iter__(self) -> Iterator[str]:
        """Get an iterator of all keys in this store."""
        yield ".zmetadata"
        yield ".zgroup"
        yield ".zattrs"
        for array_name in self._arrays.keys():
            yield from self._get_array_keys(array_name)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __contains__(self, key: str) -> bool:
        if key in (".zmetadata", ".zgroup", ".zattrs"):
            return True
        try:
            array_name, value_id = self._parse_array_key(key)
        except KeyError:
            return False
        if value_id in (".zarray", ".zattrs"):
            return True
        try:
            self._get_array_chunk_index(array_name, value_id)
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str) -> Union[bytes, np.ndarray]:
        item = self._get_item(key)
        if isinstance(item, dict):
            return dict_to_bytes(item)
        elif isinstance(item, str):
            return str_to_bytes(item)
        return item

    def __setitem__(self, key: str, value: bytes) -> None:
        class_name = self.__module__ + "." + self.__class__.__name__
        raise TypeError(f"{class_name} is read-only")

    def __delitem__(self, key: str) -> None:
        self.rmdir(key)

    ########################################################################
    # Utilities
    ##########################################################################

    @classmethod
    def from_dataset(
        cls, dataset: xr.Dataset, array_defaults: Optional[GenericArrayLike] = None
    ) -> "GenericZarrStore":
        """Create a Zarr store for given *dataset*.
        to the *dataset*'s attributes.
        The following *array_defaults* properties can be provided
        (other properties are prescribed by the *dataset*):

        * `fill_value`- defaults to None
        * `compressor`- defaults to None
        * `filters`- defaults to None
        * `order`- defaults to "C"
        * `chunk_encoding` - defaults to "bytes"

        Args:
            dataset: The dataset
            array_defaults: Array default values.

        Returns:
            A new Zarr store instance.
        """

        def _get_dataset_data(ds=None, chunk_info=None, array_info=None) -> np.ndarray:
            array_name = array_info["name"]
            chunk_slices = chunk_info["slices"]
            return ds[array_name][chunk_slices].values

        arrays = []
        for var_name, var in dataset.variables.items():
            arrays.append(
                GenericArray(
                    name=str(var_name),
                    dtype=np.dtype(var.dtype).str,
                    dims=[str(dim) for dim in var.dims],
                    shape=var.shape,
                    chunks=(
                        [(max(*c) if len(c) > 1 else c[0]) for c in var.chunks]
                        if var.chunks
                        else var.shape
                    ),
                    attrs={str(k): v for k, v in var.attrs.items()},
                    get_data=_get_dataset_data,
                    get_data_params=dict(ds=dataset),
                )
            )

        attrs = {str(k): v for k, v in dataset.attrs.items()}
        return GenericZarrStore(*arrays, attrs=attrs, array_defaults=array_defaults)

    ########################################################################
    # Helpers
    ##########################################################################

    def _get_item(self, key: str) -> Union[dict, str, bytes]:
        if key == ".zmetadata":
            return self._get_metadata_item()
        if key == ".zgroup":
            return self._get_group_item()
        if key == ".zattrs":
            return self._get_attrs_item()

        array_name, value_id = self._parse_array_key(key)
        array = self._arrays[array_name]

        if value_id == ".zarray":
            return self._get_array_spec_item(array)
        if value_id == ".zattrs":
            return self._get_array_attrs_item(array)

        chunk_index = self._get_array_chunk_index(array_name, value_id)
        return self._get_array_data_item(array, chunk_index)

    def _get_metadata_item(self):
        metadata = {
            ".zgroup": self._get_item(".zgroup"),
            ".zattrs": self._get_item(".zattrs"),
        }
        for array_name in self._arrays.keys():
            key = array_name + "/.zarray"
            metadata[key] = self._get_item(key)
            key = array_name + "/.zattrs"
            metadata[key] = self._get_item(key)
        return {"zarr_consolidated_format": 1, "metadata": metadata}

    # noinspection PyMethodMayBeStatic
    def _get_group_item(self):
        return {"zarr_format": 2}

    def _get_attrs_item(self):
        return self._attrs or {}

    # noinspection PyMethodMayBeStatic
    def _get_array_spec_item(self, array: GenericArray):

        # JSON-encode fill_value
        fill_value = array["fill_value"]
        if isinstance(fill_value, float):
            if math.isnan(fill_value):
                fill_value = "NaN"
            elif math.isinf(fill_value):
                if fill_value < 0:
                    fill_value = "-Infinity"
                else:
                    fill_value = "Infinity"

        # JSON-encode compressor
        compressor = array["compressor"]
        if compressor is not None:
            compressor = compressor.get_config()

        # JSON-encode filters
        filters = array["filters"]
        if filters is not None:
            filters = list(f.get_config() for f in filters)

        return {
            "zarr_format": 2,
            "dtype": array["dtype"],
            "shape": list(array["shape"]),
            "chunks": list(array["chunks"]),
            "fill_value": fill_value,
            "compressor": compressor,
            "filters": filters,
            "order": array["order"],
        }

    # noinspection PyMethodMayBeStatic
    def _get_array_attrs_item(self, array: GenericArray):
        dims = array["dims"]
        attrs = array["attrs"]
        return {"_ARRAY_DIMENSIONS": dims, **(attrs or {})}

    # noinspection PyMethodMayBeStatic
    def _get_array_data_item(
        self, array: Dict[str, Any], chunk_index: Tuple[int]
    ) -> Union[bytes, np.ndarray]:
        # Note, here array is expected to be "finalized",
        # that is, validated and normalized

        shape = array["shape"]
        chunks = array["chunks"]
        chunk_shape = None

        data = array["data"]
        if data is None:
            get_data = array["get_data"]
            assert callable(get_data)  # Has been ensured before
            get_data_params = array["get_data_params"]
            get_data_kwargs = dict(get_data_params)
            get_data_info = array["get_data_info"]
            if get_data_info["has_chunk_info"]:
                chunk_shape = get_chunk_shape(shape, chunks, chunk_index)
                array_slices = get_array_slices(shape, chunks, chunk_index)
                get_data_kwargs["chunk_info"] = {
                    "index": chunk_index,
                    "shape": chunk_shape,
                    "slices": array_slices,
                }
            if get_data_info["has_array_info"]:
                get_data_kwargs["array_info"] = dict(array)

            data = get_data(**get_data_kwargs)

        chunk_encoding = array["chunk_encoding"]
        if isinstance(data, np.ndarray):
            # As of Zarr 2.0, all chunks of an array
            # must have the same shape (= chunks)
            if data.shape != chunks:
                # This commonly happens if array shape sizes
                # are not integer multiple of chunk shape sizes.
                if chunk_shape is None:
                    # Compute expected chunk shape.
                    chunk_shape = get_chunk_shape(shape, chunks, chunk_index)
                # We will only pad the data if the data shape
                # corresponds to the expected chunk's shape.
                if data.shape == chunk_shape:
                    padding = get_chunk_padding(shape, chunks, chunk_index)
                    fill_value = array["fill_value"]
                    data = np.pad(
                        data, padding, mode="constant", constant_values=fill_value or 0
                    )
                else:
                    key = format_chunk_key(array["name"], chunk_index)
                    raise ValueError(
                        f"{key}:"
                        f" data chunk at {chunk_index}"
                        f" must have shape {chunk_shape},"
                        f" but was {data.shape}"
                    )
            if chunk_encoding == "bytes":
                # Convert to bytes, filter and compress
                data = ndarray_to_bytes(
                    data,
                    order=array["order"],
                    filters=array["filters"],
                    compressor=array["compressor"],
                )

        # Sanity check
        if (chunk_encoding == "bytes" and not isinstance(data, bytes)) or (
            chunk_encoding == "ndarray" and not isinstance(data, np.ndarray)
        ):
            key = format_chunk_key(array["name"], chunk_index)
            expected_type = "numpy.ndarray" if chunk_encoding == "ndarray" else "bytes"
            raise TypeError(
                f"{key}:"
                f" data must be encoded as {expected_type},"
                f" but was {type(data).__name__}"
            )

        return data

    def _parse_array_key(self, key: str) -> Tuple[str, str]:
        array_name_and_value_id = key.rsplit("/", maxsplit=1)
        if len(array_name_and_value_id) != 2:
            raise KeyError(key)
        array_name, value_id = array_name_and_value_id
        if array_name not in self._arrays:
            raise KeyError(key)
        return array_name, value_id

    def _get_array_chunk_index(self, array_name: str, index_id: str) -> Tuple[int]:
        try:
            chunk_index = tuple(map(int, index_id.split(".")))
        except (ValueError, TypeError):
            raise KeyError(f"{array_name}/{index_id}")
        array = self._arrays[array_name]
        shape = array["shape"]
        if len(chunk_index) != len(shape):
            raise KeyError(f"{array_name}/{index_id}")
        num_chunks = array["num_chunks"]
        for i, n in zip(chunk_index, num_chunks):
            if not (0 <= i < n):
                raise KeyError(f"{array_name}/{index_id}")
        return chunk_index

    def _get_array_keys(self, array_name: str) -> Iterator[str]:
        yield array_name + "/.zarray"
        yield array_name + "/.zattrs"
        array = self._arrays[array_name]
        num_chunks = array["num_chunks"]
        yield from get_chunk_keys(array_name, num_chunks)


def get_array_slices(
    shape: Tuple[int, ...], chunks: Tuple[int, ...], chunk_index: Tuple[int, ...]
) -> Tuple[slice, ...]:
    return tuple(
        slice(i * c, i * c + (c if (i + 1) * c <= s else s % c))
        for s, c, i in zip(shape, chunks, chunk_index)
    )


def get_chunk_shape(
    shape: Tuple[int, ...], chunks: Tuple[int, ...], chunk_index: Tuple[int, ...]
) -> Tuple[int, ...]:
    return tuple(
        c if (i + 1) * c <= s else s % c for s, c, i in zip(shape, chunks, chunk_index)
    )


def get_chunk_padding(
    shape: Tuple[int, ...], chunks: Tuple[int, ...], chunk_index: Tuple[int, ...]
):
    return tuple(
        (0, 0 if (i + 1) * c <= s else c - s % c)
        for s, c, i in zip(shape, chunks, chunk_index)
    )


def get_chunk_indexes(num_chunks: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    if not num_chunks:
        yield 0,
    else:
        yield from itertools.product(*tuple(map(range, map(int, num_chunks))))


def get_chunk_keys(array_name: str, num_chunks: Tuple[int, ...]) -> Iterator[str]:
    for chunk_index in get_chunk_indexes(num_chunks):
        yield format_chunk_key(array_name, chunk_index)


def format_chunk_key(array_name: str, chunk_index: Tuple[int, ...]) -> str:
    chunk_id = ".".join(map(str, chunk_index))
    return f"{array_name}/{chunk_id}"


def dict_to_bytes(d: Dict) -> bytes:
    return str_to_bytes(json.dumps(d, indent=2))


def str_to_bytes(s: str) -> bytes:
    return bytes(s, encoding="utf-8")


def ndarray_to_bytes(
    data: np.ndarray,
    order: Optional[str] = None,
    filters: Optional[Sequence[Any]] = None,
    compressor: Optional[numcodecs.abc.Codec] = None,
) -> bytes:
    data = data.tobytes(order=order or "C")
    if filters:
        for f in filters:
            data = f.encode(data)
    if compressor is not None:
        data = compressor.encode(data)
    return data
