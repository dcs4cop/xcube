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

import os.path
import shutil
from typing import Type, Union, Sequence

import zarr

from xcube.core.unchunk import unchunk_dataset


def optimize_dataset(
    input_path: str,
    output_path: str = None,
    in_place: bool = False,
    unchunk_coords: Union[bool, str, Sequence[str]] = False,
    exception_type: Type[Exception] = ValueError,
):
    """Optimize a dataset for faster access.

    Reduces the number of metadata and coordinate data files in xcube dataset given by given by *dataset_path*.
    Consolidated cubes open much faster from remote locations, e.g. in object storage,
    because obviously much less HTTP requests are required to fetch initial cube meta
    information. That is, it merges all metadata files into a single top-level JSON file ".zmetadata".

    If *unchunk_coords* is given, it also removes any chunking of coordinate variables
    so they comprise a single binary data file instead of one file per data chunk.
    The primary usage of this function is to optimize data cubes for cloud object storage.
    The function currently works only for data cubes using Zarr format.
    *unchunk_coords* can be a name, or list of names of the coordinate variable(s) to be consolidated.
    If boolean `True` is used, coordinate all variables will be consolidated.

    Args:
        input_path: Path to input dataset with ZARR format.
        output_path: Path to output dataset with ZARR format. May
            contain "{input}" template string, which is replaced by the
            input path's file name without file name extension.
        in_place: Whether to modify the dataset in place. If False, a
            copy is made and *output_path* must be given.
        unchunk_coords: The name of a coordinate variable or a list of
            coordinate variables whose chunks should be consolidated.
            Pass `True` to consolidate chunks of all coordinate
            variables.
        exception_type: Type of exception to be used on value errors.
    """

    if not os.path.isfile(os.path.join(input_path, ".zgroup")):
        raise exception_type("Input path must point to ZARR dataset directory.")

    input_path = os.path.abspath(os.path.normpath(input_path))

    if in_place:
        output_path = input_path
    else:
        if not output_path:
            raise exception_type(f"Output path must be given.")
        if "{input}" in output_path:
            base_name, _ = os.path.splitext(os.path.basename(input_path))
            output_path = output_path.format(input=base_name)
        output_path = os.path.abspath(os.path.normpath(output_path))
        if os.path.exists(output_path):
            raise exception_type(f"Output path already exists.")

    if not in_place:
        shutil.copytree(input_path, output_path)

    if unchunk_coords:
        if isinstance(unchunk_coords, str):
            var_names = (unchunk_coords,)
        elif isinstance(unchunk_coords, bool):
            var_names = None
        else:
            var_names = tuple(unchunk_coords)
        unchunk_dataset(output_path, var_names=var_names, coords_only=True)

    zarr.convenience.consolidate_metadata(output_path)
