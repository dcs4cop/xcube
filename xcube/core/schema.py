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

from typing import Tuple, Sequence, Dict, Optional, Mapping, Union, Hashable

import numpy as np
import xarray as xr

from xcube.core.gridmapping import GridMapping


class CubeSchema:
    """A schema that can be used to create new xcube datasets.
    The given *shape*, *dims*, and *chunks*, *coords* apply to all data variables.

    Args:
        shape: A tuple of dimension sizes.
        coords: A dictionary of coordinate variables. Must have values
            for all *dims*.
        dims: A sequence of dimension names. Defaults to `('time',
            'lat', 'lon')`.
        chunks: A tuple of chunk sizes in each dimension.
    """

    def __init__(
        self,
        shape: Sequence[int],
        coords: Mapping[str, xr.DataArray],
        x_name: str = "lon",
        y_name: str = "lat",
        time_name: str = "time",
        dims: Sequence[str] = None,
        chunks: Sequence[int] = None,
    ):
        if not shape:
            raise ValueError("shape must be a sequence of integer sizes")
        if not coords:
            raise ValueError(
                "coords must be a mapping from dimension names to label arrays"
            )
        if not x_name:
            raise ValueError("x_name must be given")
        if not y_name:
            raise ValueError("y_name must be given")
        if not time_name:
            raise ValueError("time_name must be given")

        ndim = len(shape)
        if ndim < 3:
            raise ValueError("shape must have at least three dimensions")
        dims = tuple(dims) or (time_name, y_name, x_name)
        if dims and len(dims) != ndim:
            raise ValueError("dims must have same length as shape")
        if x_name not in coords or y_name not in coords or time_name not in coords:
            raise ValueError(
                f"missing variables {x_name!r}, {y_name!r}, {time_name!r} in coords"
            )
        x_var, y_var, time_var = (
            coords.get(x_name),
            coords.get(y_name),
            coords.get(time_name),
        )
        if x_var.ndim != 1 or y_var.ndim != 1 or time_var.ndim != 1:
            raise ValueError(
                f"variables {x_name!r}, {y_name!r}, {time_name!r} in coords must be 1-D"
            )
        x_dim, y_dim, time_dim = x_var.dims[0], y_var.dims[0], time_var.dims[0]
        if dims[0] != time_dim:
            raise ValueError(f"the first dimension in dims must be {time_dim!r}")
        if dims[-2:] != (y_dim, x_dim):
            raise ValueError(
                f"the last two dimensions in dims must be {y_dim!r} and {x_dim!r}"
            )
        if chunks and len(chunks) != ndim:
            raise ValueError("chunks must have same length as shape")
        for i in range(ndim):
            dim_name = dims[i]
            dim_size = shape[i]
            if dim_name not in coords:
                raise ValueError(f"missing dimension {dim_name!r} in coords")
            dim_labels = coords[dim_name]
            if len(dim_labels.shape) != 1:
                raise ValueError(
                    f"labels of {dim_name!r} in coords must be one-dimensional"
                )
            if len(dim_labels) != dim_size:
                raise ValueError(
                    f"number of labels of {dim_name!r} in coords does not match shape"
                )

        self._shape = tuple(shape)
        self._x_name = x_name
        self._y_name = y_name
        self._time_name = time_name
        self._dims = dims
        self._chunks = tuple(chunks) if chunks else None
        self._coords = dict(coords)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._dims)

    @property
    def dims(self) -> Tuple[str, ...]:
        """Tuple of dimension names."""
        return self._dims

    @property
    def x_name(self) -> str:
        """Name of the spatial x coordinate variable."""
        return self._x_name

    @property
    def y_name(self) -> str:
        """Name of the spatial y coordinate variable."""
        return self._y_name

    @property
    def time_name(self) -> str:
        """Name of the time coordinate variable."""
        return self._time_name

    @property
    def x_var(self) -> xr.DataArray:
        """Spatial x coordinate variable."""
        return self._coords[self._x_name]

    @property
    def y_var(self) -> xr.DataArray:
        """Spatial y coordinate variable."""
        return self._coords[self._y_name]

    @property
    def time_var(self) -> xr.DataArray:
        """Time coordinate variable."""
        return self._coords[self._time_name]

    @property
    def x_dim(self) -> str:
        """Name of the spatial x dimension."""
        return self._dims[-1]

    @property
    def y_dim(self) -> str:
        """Name of the spatial y dimension."""
        return self._dims[-2]

    @property
    def time_dim(self) -> str:
        """Name of the time dimension."""
        return self._dims[0]

    @property
    def x_size(self) -> int:
        """Size of the spatial x dimension."""
        return self._shape[-1]

    @property
    def y_size(self) -> int:
        """Size of the spatial y dimension."""
        return self._shape[-2]

    @property
    def time_size(self) -> int:
        """Size of the time dimension."""
        return self._shape[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of dimension sizes."""
        return self._shape

    @property
    def chunks(self) -> Optional[Tuple[int]]:
        """Tuple of dimension chunk sizes."""
        return self._chunks

    @property
    def coords(self) -> Dict[str, xr.DataArray]:
        """Dictionary of coordinate variables."""
        return self._coords

    @classmethod
    def new(cls, cube: xr.Dataset) -> "CubeSchema":
        """Create a cube schema from given *cube*."""
        return get_cube_schema(cube)

    def _repr_html_(self):
        """Return a HTML representation for Jupyter Notebooks."""
        return (
            f"<table>"
            f"<tr><td>Shape:</td><td>{self.shape}</td></tr>"
            f"<tr><td>Chunk sizes:</td><td>{self.chunks}</td></tr>"
            f"<tr><td>Dimensions:</td><td>{self.dims}</td></tr>"
            f"</table>"
        )


# TODO (forman): code duplication with xcube.core.verify._check_data_variables(), line 76
def get_cube_schema(cube: xr.Dataset) -> CubeSchema:
    """Derive cube schema from given *cube*.

    Args:
        cube: The data cube.

    Returns:
        The cube schema.
    """

    xy_var_names = get_dataset_xy_var_names(
        cube, must_exist=True, dataset_arg_name="cube"
    )
    time_var_name = get_dataset_time_var_name(
        cube, must_exist=True, dataset_arg_name="cube"
    )

    first_dims = None
    first_shape = None
    first_chunks = None
    first_coords = None

    for var_name, var in cube.data_vars.items():
        dims = var.dims
        if first_dims is None:
            first_dims = dims
        elif first_dims != dims:
            raise ValueError(
                f"all variables must have same dimensions, but variable {var_name!r} "
                f"has dimensions {dims!r}"
            )

        shape = var.shape
        if first_shape is None:
            first_shape = shape
        elif first_shape != shape:
            raise ValueError(
                f"all variables must have same shape, but variable {var_name!r} "
                f"has shape {shape!r}"
            )

        coords = var.coords
        if first_coords is None:
            first_coords = coords

        dask_chunks = var.chunks
        if dask_chunks:
            chunks = []
            for i in range(var.ndim):
                dim_name = var.dims[i]
                dim_chunk_sizes = dask_chunks[i]
                first_size = dim_chunk_sizes[0]
                if any(size != first_size for size in dim_chunk_sizes[1:-1]):
                    raise ValueError(
                        f"dimension {dim_name!r} of variable {var_name!r} has chunks of different sizes: "
                        f"{dim_chunk_sizes!r}"
                    )
                chunks.append(first_size)
            chunks = tuple(chunks)
            if first_chunks is None:
                first_chunks = chunks
            elif first_chunks != chunks:
                raise ValueError(
                    f"all variables must have same chunks, but variable {var_name!r} "
                    f"has chunks {chunks!r}"
                )

    if first_dims is None:
        raise ValueError("cube is empty")

    return CubeSchema(
        first_shape,
        first_coords,
        x_name=xy_var_names[0],
        y_name=xy_var_names[1],
        time_name=time_var_name,
        dims=tuple(str(d) for d in first_dims),
        chunks=first_chunks,
    )


def get_dataset_xy_var_names(
    coords: Union[xr.Dataset, xr.DataArray, Mapping[Hashable, xr.DataArray]],
    must_exist: bool = False,
    dataset_arg_name: str = "dataset",
) -> Optional[Tuple[str, str]]:
    if hasattr(coords, "coords"):
        coords = coords.coords
    x_var_name = None
    y_var_name = None
    for var_name, var in coords.items():
        if (
            var.attrs.get("standard_name") == "projection_x_coordinate"
            or var.attrs.get("long_name") == "x coordinate of projection"
        ):
            if var.ndim == 1:
                x_var_name = var_name
        if (
            var.attrs.get("standard_name") == "projection_y_coordinate"
            or var.attrs.get("long_name") == "y coordinate of projection"
        ):
            if var.ndim == 1:
                y_var_name = var_name
        if x_var_name and y_var_name:
            return str(x_var_name), str(y_var_name)

    x_var_name = None
    y_var_name = None
    for var_name, var in coords.items():
        if var.attrs.get("long_name") == "longitude":
            if var.ndim == 1:
                x_var_name = var_name
        if var.attrs.get("long_name") == "latitude":
            if var.ndim == 1:
                y_var_name = var_name
        if x_var_name and y_var_name:
            return str(x_var_name), str(y_var_name)

    for x_var_name, y_var_name in (
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("x", "y"),
    ):
        if x_var_name in coords and y_var_name in coords:
            x_var = coords[x_var_name]
            y_var = coords[y_var_name]
            if x_var.ndim == 1 and y_var.ndim == 1:
                return x_var_name, y_var_name

    if must_exist:
        raise ValueError(
            f"{dataset_arg_name}" f" has no valid spatial coordinate variables"
        )


def get_dataset_time_var_name(
    dataset: Union[xr.Dataset, xr.DataArray],
    must_exist: bool = False,
    dataset_arg_name: str = "dataset",
) -> Optional[str]:
    time_var_name = "time"
    if time_var_name in dataset.coords:
        time_var = dataset.coords[time_var_name]
        if time_var.ndim == 1 and np.issubdtype(time_var.dtype, np.datetime64):
            return time_var_name

    if must_exist:
        raise ValueError(f"{dataset_arg_name} has no valid time coordinate variable")

    return None


def get_dataset_bounds_var_name(
    dataset: Union[xr.Dataset, xr.DataArray],
    var_name: str,
    must_exist: bool = False,
    dataset_arg_name: str = "dataset",
) -> Optional[str]:
    if var_name in dataset.coords:
        var = dataset[var_name]
        bounds_var_name = var.attrs.get("bounds", f"{var_name}_bnds")
        if bounds_var_name in dataset:
            bounds_var = dataset[bounds_var_name]
            if (
                bounds_var.ndim == 2
                and bounds_var.shape[0] == var.shape[0]
                and bounds_var.shape[1] == 2
            ):
                return bounds_var_name

    if must_exist:
        raise ValueError(
            f"{dataset_arg_name} has no valid bounds variable for variable {var_name!r}"
        )

    return None


def get_dataset_chunks(dataset: xr.Dataset) -> Dict[Hashable, int]:
    """Get the most common chunk sizes for each
    chunked dimension of *dataset*.

    Note: Only data variables are considered.

    Args:
        dataset: A dataset.

    Returns:
        A dictionary that maps dimension names to common chunk sizes.
    """

    # Record the frequencies of chunk sizes for
    # each dimension d in each data variable var
    dim_size_counts: Dict[Hashable, Dict[int, int]] = {}
    for var_name, var in dataset.data_vars.items():
        if var.chunks:
            for d, c in zip(var.dims, var.chunks):
                # compute max chunk size max_c from
                # e.g.  c = (512, 512, 512, 193)
                max_c = max(0, *c)
                # for dimension d, save the frequencies
                # of the different max_c
                if d not in dim_size_counts:
                    size_counts = {max_c: 1}
                    dim_size_counts[d] = size_counts
                else:
                    size_counts = dim_size_counts[d]
                    if max_c not in size_counts:
                        size_counts[max_c] = 1
                    else:
                        size_counts[max_c] += 1

    # For each dimension d, determine the most frequently
    # seen chunk size max_c
    dim_sizes: Dict[Hashable, int] = {}
    for d, size_counts in dim_size_counts.items():
        max_count = 0
        best_max_c = 0
        for max_c, count in size_counts.items():
            if count > max_count:
                # Should always come here, because count=1 is minimum
                max_count = count
                best_max_c = max_c
        assert best_max_c > 0
        dim_sizes[d] = best_max_c

    return dim_sizes


def rechunk_cube(
    cube: xr.Dataset,
    gm: GridMapping,
    chunks: Optional[Dict[str, int]] = None,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Tuple[xr.Dataset, GridMapping]:
    """Re-chunk data variables of *cube* so they all share the same chunk
    sizes for their dimensions.

    This functions rechunks *cube* for maximum compatibility with
    the Zarr format. Therefore it removes the "chunks" encoding
    from all variables.

    Args:
        cube: A data cube
        gm: The cube's grid mapping
        chunks: Optional mapping of dimension names to chunk sizes
        tile_size: Optional tile sizes, i.e. chunk size of spatial
            dimensions, given as (width, height)

    Returns:
        A potentially rechunked *cube* and adjusted grid mapping.
    """

    # get initial, common cube chunk sizes from given cube
    cube_chunks = get_dataset_chunks(cube)

    # Given chunks will overwrite initial values
    if chunks:
        for dim_name, size in chunks.items():
            cube_chunks[dim_name] = size

    # Given tile size will overwrite spatial dims
    x_dim_name, y_dim_name = gm.xy_dim_names
    if tile_size is not None:
        cube_chunks[x_dim_name] = tile_size[0]
        cube_chunks[y_dim_name] = tile_size[1]

    # Given grid mapping's tile size will overwrite
    # spatial dims only if missing still
    if gm.tile_size is not None:
        if x_dim_name not in cube_chunks:
            cube_chunks[x_dim_name] = gm.tile_size[0]
        if y_dim_name not in cube_chunks:
            cube_chunks[y_dim_name] = gm.tile_size[1]

    # If there is no chunking required, return identities
    if not cube_chunks:
        return cube, gm

    chunked_cube = xr.Dataset(attrs=cube.attrs)

    # Coordinate variables are always
    # chunked automatically
    chunked_cube = chunked_cube.assign_coords(
        coords={
            var_name: var.chunk({dim_name: "auto" for dim_name in var.dims})
            for var_name, var in cube.coords.items()
        }
    )

    # Data variables are chunked according to cube_chunks,
    # or if not specified, by the dimension size.
    chunked_cube = chunked_cube.assign(
        variables={
            var_name: var.chunk(
                {
                    dim_name: cube_chunks.get(dim_name, cube.dims[dim_name])
                    for dim_name in var.dims
                }
            )
            for var_name, var in cube.data_vars.items()
        }
    )

    # Update chunks encoding for Zarr
    for var_name, var in chunked_cube.variables.items():
        if "chunks" in var.encoding:
            del var.encoding["chunks"]
        # if var.chunks is not None:
        #     # sizes[0] is the first of
        #     # e.g. sizes = (512, 512, 71)
        #     var.encoding.update(chunks=[
        #         sizes[0] for sizes in var.chunks
        #     ])
        # elif 'chunks' in var.encoding:
        #     del var.encoding['chunks']
        # print(f"--> {var_name}: encoding={var.encoding.get('chunks')!r}, chunks={var.chunks!r}")

    # Test whether tile size has changed after re-chunking.
    # If so, we will change the grid mapping too.
    tile_width = cube_chunks.get(x_dim_name)
    tile_height = cube_chunks.get(y_dim_name)
    assert tile_width is not None
    assert tile_height is not None
    tile_size = (tile_width, tile_height)
    if tile_size != gm.tile_size:
        # Note, changing grid mapping tile size may
        # rechunk (2D) coordinates in chunked_cube too
        gm = gm.derive(tile_size=tile_size)

    return chunked_cube, gm
