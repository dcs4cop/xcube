# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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
from typing import List, Callable, Sequence, Optional, Tuple

import xarray as xr
from deprecated import deprecated

from xcube.core.dsio import open_dataset

PyramidLevelCallback = Callable[[xr.Dataset, int, int], Optional[xr.Dataset]]

_DEPRECATED_READ = (
    "No longer in use. To read multi-level datasets, use:\n"
    '>>> data_store = new_data_store("file")\n'
    '>>> ml_dataset = data_store.open_data(path + ".levels")'
)
_DEPRECATED_WRITE = (
    "No longer in use. To write multi-level datasets, use:\n"
    '>>> data_store = new_data_store("file")\n'
    '>>> data_store.write_data(dataset, path + ".levels")'
)


@deprecated(version="0.10.2", reason=_DEPRECATED_WRITE)
def compute_levels(
    dataset: xr.Dataset,
    spatial_dims: Tuple[str, str] = None,
    spatial_shape: Tuple[int, int] = None,
    spatial_tile_shape: Tuple[int, int] = None,
    var_names: Sequence[str] = None,
    num_levels_max: int = None,
    post_process_level: PyramidLevelCallback = None,
    progress_monitor: PyramidLevelCallback = None,
) -> List[xr.Dataset]:
    """
    Transform the given *dataset* into the levels of a multi-level
    pyramid with spatial resolution decreasing by a factor of two
    in both spatial dimensions.

    It is assumed that the spatial dimensions of each variable are
    the innermost, that is, the last two elements of a variable's
    shape provide the spatial dimension sizes.

    Args:
        dataset: The input dataset to be turned into
            a multi-level pyramid.
        spatial_dims: If given, only variables are considered
            whose last to dimension elements match the given *spatial_dims*.
        spatial_shape: If given, only variables are
            considered whose last to shape elements match the
            given *spatial_shape*.
        spatial_tile_shape: If given, chunking will match the
            provided *spatial_tile_shape*.
        var_names: Variables to consider. If None, all variables
            with at least two dimensions are considered.
        num_levels_max: If given, the maximum number
            of pyramid levels.
        post_process_level: If given, the function will be
            called for each level and must return a dataset.
        progress_monitor: If given, the function will be
            called for each level.
    Returns:
        A list of dataset instances representing
        the multi-level pyramid.
    """
    dropped_vars, spatial_shape, spatial_tile_shape = _filter_level_source_dataset(
        dataset, var_names, spatial_dims, spatial_shape, spatial_tile_shape
    )
    if dropped_vars:
        dataset = dataset.drop_vars(dropped_vars)

    if not tuple(dataset.data_vars):
        raise ValueError(
            "cannot compute pyramid levels because "
            "no suitable data variables were found"
        )

    if spatial_tile_shape is None:
        tile_w = min(spatial_shape[0], 512)
        tile_h = min(spatial_shape[1], 512)
        spatial_tile_shape = tile_w, tile_h

    # Count num_levels
    level_shapes = _compute_level_shapes(
        spatial_shape, spatial_tile_shape, num_levels_max=num_levels_max
    )
    num_levels = len(level_shapes)

    # Compute levels
    level_dataset = dataset
    level_datasets = []
    for level in range(num_levels):
        if level > 0:
            # Down-sample levels
            downsampled_vars = {}
            for var_name in level_dataset.data_vars:
                var = level_dataset.data_vars[var_name]
                # For time being, we use the simplest and
                # likely fastest downsampling I can think of
                downsampled_var = var[..., ::2, ::2]
                if downsampled_var.shape[-2:] != level_shapes[level]:
                    import warnings

                    warnings.warn(
                        f"unexpected spatial shape for down-sampled"
                        f" variable {var_name!r}:"
                        f" expected {level_shapes[level]},"
                        f" but found {downsampled_var.shape[-2:]}"
                    )
                downsampled_vars[var_name] = downsampled_var
            level_dataset = xr.Dataset(downsampled_vars, attrs=level_dataset.attrs)

        level_dataset = _tile_level_dataset(level_dataset, spatial_tile_shape)

        # Apply post processor, if any
        if post_process_level is not None:
            level_dataset = post_process_level(
                level_dataset, len(level_datasets), num_levels
            )

        # Inform progress monitor, if any
        if progress_monitor is not None:
            progress_monitor(level_dataset, len(level_datasets), num_levels)

        # Collect level dataset
        level_datasets.append(level_dataset)

    return level_datasets


@deprecated(version="0.10.2", reason=_DEPRECATED_WRITE)
def write_levels(
    output_path: str,
    dataset: xr.Dataset = None,
    input_path: str = None,
    link_input: bool = False,
    progress_monitor: PyramidLevelCallback = None,
    **kwargs,
) -> List[xr.Dataset]:
    """Transform the given dataset given by a *dataset* instance
    or *input_path* string into the levels of a multi-level pyramid
    with spatial resolution decreasing by a factor of two in both
    spatial dimensions and write them to *output_path*.

    One of *dataset* and *input_path* must be given.

    Args:
        output_path: Output path
        dataset: Dataset to be converted and written as levels.
        input_path: Input path to a dataset to be transformed and
            written as levels.
        link_input: Just link the dataset at level zero instead of
            writing it.
        progress_monitor: An optional progress monitor.
        **kwargs: Keyword-arguments accepted by the `compute_levels()`
            function.

    Returns:
        A list of dataset instances representing the multi-level
        pyramid.
    """
    if dataset is None and input_path is None:
        raise ValueError("at least one of dataset or input_path must be given")

    if link_input and input_path is None:
        raise ValueError("input_path must be provided to link input")

    _post_process_level = (
        kwargs.pop("post_process_level") if "post_process_level" in kwargs else None
    )

    def post_process_level(level_dataset, index, num_levels):
        if _post_process_level is not None:
            level_dataset = _post_process_level(level_dataset, index, num_levels)

        if index == 0 and link_input:
            with open(os.path.join(output_path, f"{index}.link"), "w") as fp:
                fp.write(input_path)
        else:
            path = os.path.join(output_path, f"{index}.zarr")
            level_dataset.to_zarr(path)
            level_dataset.close()
            level_dataset = xr.open_zarr(path)

        if progress_monitor is not None:
            progress_monitor(level_dataset, index, num_levels)

        return level_dataset

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dataset is None:
        dataset = open_dataset(input_path)

    return compute_levels(dataset, post_process_level=post_process_level, **kwargs)


@deprecated(version="0.10.2", reason=_DEPRECATED_READ)
def read_levels(
    dir_path: str, progress_monitor: PyramidLevelCallback = None
) -> List[xr.Dataset]:
    """Read the of a multi-level pyramid with spatial resolution
    decreasing by a factor of two in both spatial dimensions.

    Args:
        dir_path: The directory path.
        progress_monitor: An optional progress monitor.

    Returns:
        A list of dataset instances representing the multi-level
        pyramid.
    """
    file_paths = os.listdir(dir_path)
    level_paths = {}
    num_levels = -1
    for filename in file_paths:
        file_path = os.path.join(dir_path, filename)
        basename, ext = os.path.splitext(filename)
        if basename.isdigit():
            index = int(basename)
            num_levels = max(num_levels, index + 1)
            if os.path.isfile(file_path) and ext == ".link":
                level_paths[index] = (ext, file_path)
            elif os.path.isdir(file_path) and ext == ".zarr":
                level_paths[index] = (ext, file_path)

    if num_levels != len(level_paths):
        raise ValueError(
            f"Inconsistent pyramid directory:"
            f" expected {num_levels} but found"
            f" {len(level_paths)} entries:"
            f" {dir_path}"
        )

    levels = []
    for index in range(num_levels):
        ext, file_path = level_paths[index]
        if ext == ".link":
            with open(file_path, "r") as fp:
                link_file_path = fp.read()
            if not os.path.isabs(link_file_path):
                parent_dir_path = os.path.abspath(os.path.dirname(dir_path) or ".")
                link_file_path = os.path.join(parent_dir_path, link_file_path)
            dataset = xr.open_zarr(link_file_path)
        else:
            dataset = xr.open_zarr(file_path)
        if progress_monitor is not None:
            progress_monitor(dataset, index, num_levels)
        levels.append(dataset)
    return levels


def _tile_chunk(size, tile_size):
    last_tile_size = size % tile_size
    if last_tile_size != 0:
        return (tile_size,) * (size // tile_size) + (last_tile_size,)
    return tile_size


def _tile_level_dataset(level_dataset, spatial_tile_shape):
    tile_height, tile_width = spatial_tile_shape

    # Chunk variables in level dataset according to spatial_tile_shape
    chunked_vars = {}

    # Chunk data variables according to tile size
    for var_name in level_dataset.data_vars:
        var = level_dataset.data_vars[var_name]
        height, width = var.shape[-2:]
        zarr_chunks = (1,) * (var.ndim - 2) + (tile_height, tile_width)
        dask_chunks = (1,) * (var.ndim - 2) + (
            _tile_chunk(height, tile_height),
            _tile_chunk(width, tile_width),
        )
        dask_chunks = {var.dims[i]: dask_chunks[i] for i in range(var.ndim)}
        chunked_var = var.chunk(chunks=dask_chunks)
        chunked_var.encoding.update(chunks=zarr_chunks)
        chunked_vars[var_name] = chunked_var

    # Make coordinate variable chunks equal to their shape
    # TODO (forman): find out if chunking the spatial coordinates
    #  according to tile size improves performance
    for var_name in level_dataset.coords:
        var = level_dataset.coords[var_name]
        zarr_chunks = var.shape
        dask_chunks = {var.dims[i]: var.shape[i] for i in range(var.ndim)}
        chunked_var = var.chunk(chunks=dask_chunks)
        chunked_var.encoding.update(chunks=zarr_chunks)
        chunked_vars[var_name] = chunked_var

    return level_dataset.assign(variables=chunked_vars)


def _compute_level_shapes(
    spatial_shape, spatial_tile_shape, num_levels_max=None
) -> List[Tuple[int, int]]:
    height, width = spatial_shape
    tile_height, tile_width = spatial_tile_shape
    num_levels_max = num_levels_max or -1
    level_shapes = [(height, width)]
    while True:
        width = (width + 1) // 2
        height = (height + 1) // 2
        if (
            width < tile_width
            or height < tile_height
            or num_levels_max == len(level_shapes)
        ):
            break
        level_shapes.append((height, width))
    return level_shapes


def _filter_level_source_dataset(
    dataset,
    var_names=None,
    spatial_dims=None,
    spatial_shape=None,
    spatial_tile_shape=None,
):
    if var_names:
        var_names = set(var_names)
        dropped_vars = list(set(dataset.data_vars).difference(var_names))
    else:
        var_names = set(dataset.data_vars)
        dropped_vars = []

    # Collect data variables to be dropped,
    # derive missing information from spatial data variables
    for var_name in var_names:
        if var_name not in dataset.data_vars:
            raise ValueError(f"variable {var_name} not found")

        var = dataset[var_name]

        if var.ndim < 2:
            # Must have at least the two spatial dimensions
            dropped_vars.append(var_name)
            continue

        if spatial_dims is None:
            spatial_dims = var.dims[-2:]
        elif spatial_dims != var.dims[-2:]:
            # Spatial dimensions don't fit
            dropped_vars.append(var_name)
            continue

        if spatial_shape is None:
            spatial_shape = var.shape[-2:]
        elif spatial_shape != var.shape[-2:]:
            # Spatial dimension sizes don't fit
            dropped_vars.append(var_name)
            continue

        if spatial_tile_shape is None and var.chunks is not None:

            def chunk_to_int(chunk):
                return chunk if isinstance(chunk, int) else max(chunk)

            spatial_tile_shape = tuple(map(chunk_to_int, var.chunks[-2:]))

    return dropped_vars, spatial_shape, spatial_tile_shape
