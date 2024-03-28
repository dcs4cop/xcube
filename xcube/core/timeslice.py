# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import tempfile
from collections.abc import MutableMapping
from typing import Dict, Union, Tuple

import numpy as np
import xarray as xr
import zarr

from xcube.core.chunk import chunk_dataset
from xcube.core.unchunk import unchunk_dataset

DEFAULT_TIME_EPS = np.array(1000 * 1000, dtype="timedelta64[ns]")


def find_time_slice(
    store: Union[str, MutableMapping],
    time_stamp: Union[np.datetime64, np.ndarray],
    time_eps: np.timedelta64 = DEFAULT_TIME_EPS,
) -> Tuple[int, str]:
    """
    Find time index and update mode for *time_stamp* in Zarr
    dataset given by *store*.

    Args:
        store: A zarr store.
        time_stamp: Time stamp to find index for.
        time_eps: Time epsilon for equality comparison,
            defaults to 1 millisecond.
    Returns:
        A tuple (time_index, 'insert') or (time_index, 'replace')
        if an index was found, (-1, 'create') or (-1, 'append')
        otherwise.
    """
    try:
        cube = xr.open_zarr(store)
    except (FileNotFoundError, ValueError):
        # FileNotFoundError is raised as by Zarr since 2.13,
        # before GroupNotFoundError (extends ValueError) was raised.
        # Keep ValueError for backward compatibility.
        try:
            cube = xr.open_dataset(store)
        except (FileNotFoundError, ValueError):
            # If the zarr directory does not exist, open_dataset raises a
            # FileNotFoundError (with xarray <= 0.17.0) or a ValueError
            # (with xarray 0.18.0).
            return -1, "create"

    # TODO (forman): optimise following naive search by bi-sectioning or so
    for i in range(cube.time.size):
        time = cube.time[i]
        if abs(time_stamp - time) < time_eps:
            return i, "replace"
        if time_stamp < time:
            return i, "insert"

    return -1, "append"


def append_time_slice(
    store: Union[str, MutableMapping],
    time_slice: xr.Dataset,
    chunk_sizes: Dict[str, int] = None,
):
    """Append time slice to existing zarr dataset.

    Args:
        store: A zarr store.
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name="zarr")

    # Unfortunately time_slice.to_zarr(store, mode='a', append_dim='time') will replace global attributes of store
    # with attributes of time_slice (xarray bug?), which are usually empty in our case.
    # Hence, we must save our old attributes in a copy of time_slice.
    ds = zarr.open_group(store, mode="r")
    time_slice = time_slice.copy()
    time_slice.attrs.update(ds.attrs)
    if "coordinates" in time_slice.attrs:
        # Remove 'coordinates', otherwise we get
        # ValueError: cannot serialize coordinates because the global attribute 'coordinates' already exists
        # from next time_slice.to_zarr(...) call.
        time_slice.attrs.pop("coordinates")

    time_slice.to_zarr(store, mode="a", append_dim="time", consolidated=True)

    unchunk_dataset(store, coords_only=True)


def insert_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    chunk_sizes: Dict[str, int] = None,
):
    """Insert time slice into existing zarr dataset.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    update_time_slice(
        store, insert_index, time_slice, "insert", chunk_sizes=chunk_sizes
    )


def replace_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    chunk_sizes: Dict[str, int] = None,
):
    """Replace time slice in existing zarr dataset.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    update_time_slice(
        store, insert_index, time_slice, "replace", chunk_sizes=chunk_sizes
    )


def update_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    mode: str,
    chunk_sizes: Dict[str, int] = None,
):
    """Update existing zarr dataset by new time slice.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        mode: Update mode, 'insert' or 'replace'
        chunk_sizes: desired chunk sizes
    """

    if mode not in ("insert", "replace"):
        raise ValueError(f"illegal mode value: {mode!r}")

    insert_mode = mode == "insert"

    time_var_names = []
    encoding = {}
    with xr.open_zarr(store) as cube:
        for var_name in cube.variables:
            var = cube[var_name]
            if var.ndim >= 1 and "time" in var.dims:
                if var.dims[0] != "time":
                    raise ValueError(
                        f"dimension 'time' of variable {var_name!r} must be first dimension"
                    )
                time_var_names.append(var_name)
                enc = dict(cube[var_name].encoding)
                # xarray 0.17+ supports engine preferred chunks if exposed by the backend
                # zarr does that, but when we use the new 'preferred_chunks' when writing to zarr
                # it raises and says, 'preferred_chunks' is an unsupported encoding
                if "preferred_chunks" in enc:
                    del enc["preferred_chunks"]
                encoding[var_name] = enc

    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name="zarr")
    temp_dir = tempfile.TemporaryDirectory(prefix="xcube-time-slice-", suffix=".zarr")
    time_slice.to_zarr(temp_dir.name, encoding=encoding)
    slice_root_group = zarr.open(temp_dir.name, mode="r")
    slice_arrays = dict(slice_root_group.arrays())

    cube_root_group = zarr.open(store, mode="r+")
    for var_name, var_array in cube_root_group.arrays():
        if var_name in time_var_names:
            slice_array = slice_arrays[var_name]
            if insert_mode:
                # Add one empty time step
                empty = zarr.creation.empty(slice_array.shape, dtype=var_array.dtype)
                var_array.append(empty, axis=0)
                # Shift contents
                var_array[insert_index + 1 :, ...] = var_array[insert_index:-1, ...]
            # Replace slice
            var_array[insert_index, ...] = slice_array[0]

    unchunk_dataset(store, coords_only=True)
