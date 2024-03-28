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

import os
import shutil
import warnings
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Mapping,
)

import botocore.exceptions
import pandas as pd
import s3fs
import urllib3.util
import xarray as xr
import zarr
from deprecated import deprecated

from xcube.constants import EXTENSION_POINT_DATASET_IOS
from xcube.constants import (
    FORMAT_NAME_CSV,
    FORMAT_NAME_MEM,
    FORMAT_NAME_NETCDF4,
    FORMAT_NAME_ZARR,
)
from xcube.core.timeslice import (
    append_time_slice,
    insert_time_slice,
    replace_time_slice,
)
from xcube.core.verify import assert_cube
from xcube.util.plugin import ExtensionComponent, get_extension_registry

_DEPRECATION_REASON = "Functionality is redundant. Use xcube.core.store API instead."
_DEPRECATION_VERSION = "0.12.1"


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def open_cube(input_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
    """Open a xcube dataset from *input_path*.
    If *format* is not provided it will be guessed from *input_path*.

    Args:
        input_path: input path
        format_name: format, e.g. "zarr" or "netcdf4"
        **kwargs: format-specific keyword arguments

    Returns:
        xcube dataset
    """
    return open_dataset(input_path, format_name=format_name, is_cube=True, **kwargs)


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def write_cube(
    cube: xr.Dataset,
    output_path: str,
    format_name: str = None,
    cube_asserted: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Write a xcube dataset to *output_path*.
    If *format* is not provided it will be guessed from *output_path*.

    Args:
        cube: xcube dataset to be written.
        output_path: output path
        format_name: format, e.g. "zarr" or "netcdf4"
        **kwargs: format-specific keyword arguments
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        xcube dataset *cube*
    """
    if not cube_asserted:
        assert_cube(cube)
    return write_dataset(cube, output_path, format_name=format_name, **kwargs)


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def open_dataset(
    input_path: str, format_name: str = None, is_cube: bool = False, **kwargs
) -> xr.Dataset:
    """Open a dataset from *input_path*.
    If *format* is not provided it will be guessed from *output_path*.

    Args:
        input_path: input path
        format_name: format, e.g. "zarr" or "netcdf4"
        is_cube: Whether a ValueError will be raised, if the dataset
            read from *input_path* is not a xcube dataset.
        **kwargs: format-specific keyword arguments

    Returns:
        dataset object
    """
    format_name = format_name if format_name else guess_dataset_format(input_path)
    if format_name is None:
        raise ValueError("Unknown input format")
    dataset_io = find_dataset_io(format_name, modes=["r"])
    if dataset_io is None:
        raise ValueError(f"Unknown input format {format_name!r} for {input_path}")
    dataset = dataset_io.read(input_path, **kwargs)
    if is_cube:
        assert_cube(dataset)
    return dataset


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def write_dataset(
    dataset: xr.Dataset, output_path: str, format_name: str = None, **kwargs
) -> xr.Dataset:
    """Write dataset to *output_path*.
    If *format* is not provided it will be guessed from *output_path*.

    Args:
        dataset: Dataset to be written.
        output_path: output path
        format_name: format, e.g. "zarr" or "netcdf4"
        **kwargs: format-specific keyword arguments

    Returns:
        the input dataset
    """
    format_name = format_name if format_name else guess_dataset_format(output_path)
    if format_name is None:
        raise ValueError("Unknown output format")
    dataset_io = find_dataset_io(format_name, modes=["w"])
    if dataset_io is None:
        raise ValueError(f"Unknown output format {format_name!r} for {output_path}")
    dataset_io.write(dataset, output_path, **kwargs)
    return dataset


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
class DatasetIO(ExtensionComponent, metaclass=ABCMeta):
    """An abstract base class that represents dataset input/output.

    Args:
        name: A unique dataset I/O identifier.
    """

    def __init__(self, name: str):
        super().__init__(EXTENSION_POINT_DATASET_IOS, name)

    @property
    def description(self) -> str:
        """Returns:
        A description for this input processor
        """
        return self.get_metadata_attr("description", "")

    @property
    def ext(self) -> str:
        """The primary filename extension used by this dataset I/O."""
        return self.get_metadata_attr("ext", "")

    @property
    def modes(self) -> Set[str]:
        """A set describing the modes of this dataset I/O.
        Must be one or more of "r" (read), "w" (write), and "a" (append).
        """
        return self.get_metadata_attr("modes", set())

    @abstractmethod
    def fitness(self, path: str, path_type: str = None) -> float:
        """Compute a fitness of this dataset I/O in the interval [0 to 1]
        for reading/writing from/to the given *path*.

        Args:
            path: The path or URL.
            path_type: Either "file", "dir", "url", or None.

        Returns:
            the chance in range [0 to 1]
        """
        return 0.0

    def read(self, input_path: str, **kwargs) -> xr.Dataset:
        """Read a dataset from *input_path* using format-specific read parameters *kwargs*."""
        raise NotImplementedError()

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        """ "Write *dataset* to *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        """ "Append *dataset* to existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def insert(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        """ "Insert *dataset* at *index* into existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def replace(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        """ "Replace *dataset* at *index* in existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        """ "Update *dataset* at *output_path* using format-specific open parameters *kwargs*."""
        raise NotImplementedError()


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def get_extension(name: str):
    return get_extension_registry().get_extension(EXTENSION_POINT_DATASET_IOS, name)


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def find_dataset_io_by_name(name: str):
    extension = get_extension(name)
    if not extension:
        return None
    return extension.component


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def find_dataset_io(
    format_name: str, modes: Iterable[str] = None, default: DatasetIO = None
) -> Optional[DatasetIO]:
    modes = set(modes) if modes else None
    format_name = format_name.lower()
    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)
    for dataset_io in dataset_ios:
        # noinspection PyUnresolvedReferences
        if format_name == dataset_io.name.lower():
            # noinspection PyTypeChecker
            if not modes or modes.issubset(dataset_io.modes):
                return dataset_io
    for dataset_io in dataset_ios:
        # noinspection PyUnresolvedReferences
        if format_name == dataset_io.ext.lower():
            # noinspection PyTypeChecker
            if not modes or modes.issubset(dataset_io.modes):
                return dataset_io
    return default


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def guess_dataset_format(path: str) -> Optional[str]:
    """Guess a dataset format for a file system path or URL given by *path*.

    Args:
        path: A file system path or URL.

    Returns:
        The name of a dataset format guessed from *path*.
    """
    dataset_io_fitness_list = guess_dataset_ios(path)
    if dataset_io_fitness_list:
        return dataset_io_fitness_list[0][0].name
    return None


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def guess_dataset_ios(path: str) -> List[Tuple[DatasetIO, float]]:
    """Guess suitable DatasetIO objects for a file system path or URL given by *path*.
    Returns a list of (DatasetIO, fitness) tuples, sorted by descending fitness values.
    Fitness values are in the interval (0, 1].
    The first entry is the most appropriate DatasetIO object.

    Args:
        path: A file system path or URL.

    Returns:
        A list of (DatasetIO, fitness) tuples.
    """
    if os.path.isfile(path):
        input_type = "file"
    elif os.path.isdir(path):
        input_type = "dir"
    elif path.find("://") > 0:
        input_type = "url"
    else:
        input_type = None

    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)

    dataset_io_fitness_list = []
    for dataset_io in dataset_ios:
        fitness = dataset_io.fitness(path, path_type=input_type)
        if fitness > 0.0:
            dataset_io_fitness_list.append((dataset_io, fitness))

    dataset_io_fitness_list.sort(key=lambda item: -item[1])
    return dataset_io_fitness_list


def _get_ext(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    return ext.lower()


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def query_dataset_io(filter_fn: Callable[[DatasetIO], bool] = None) -> List[DatasetIO]:
    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)
    if filter_fn is None:
        return dataset_ios
    return list(filter(filter_fn, dataset_ios))


# noinspection PyAbstractClass
@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
class MemDatasetIO(DatasetIO):
    """An in-memory  dataset I/O. Keeps all datasets in a dictionary.

    Args:
        datasets: The initial datasets as a path to dataset mapping.
    """

    def __init__(self, datasets: Dict[str, xr.Dataset] = None):
        super().__init__(FORMAT_NAME_MEM)
        self._datasets = datasets or {}

    @property
    def datasets(self) -> Dict[str, xr.Dataset]:
        return self._datasets

    def fitness(self, path: str, path_type: str = None) -> float:
        if path in self._datasets:
            return 1.0
        ext_value = _get_ext(path) == ".mem"
        type_value = 0.0
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        if path in self._datasets:
            return self._datasets[path]
        raise FileNotFoundError(path)

    def write(self, dataset: xr.Dataset, path: str, **kwargs):
        self._datasets[path] = dataset

    def append(self, dataset: xr.Dataset, path: str, **kwargs):
        if path in self._datasets:
            old_ds = self._datasets[path]
            # noinspection PyTypeChecker
            self._datasets[path] = xr.concat(
                [old_ds, dataset],
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="equals",
            )
        else:
            self._datasets[path] = dataset.copy()

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            ds = self._datasets[output_path]
            ds.attrs.update(global_attrs)


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
class Netcdf4DatasetIO(DatasetIO):
    """A dataset I/O that reads from / writes to NetCDF files."""

    def __init__(self):
        super().__init__(FORMAT_NAME_NETCDF4)

    def fitness(self, path: str, path_type: str = None) -> float:
        ext = _get_ext(path)
        ext_value = ext in {".nc", ".hdf", ".h5"}
        type_value = 0.0
        if path_type == "file":
            type_value = 1.0
        elif path_type is None:
            type_value = 0.5
        else:
            ext_value = 0.0
        return (3 * ext_value + type_value) / 4

    def read(self, input_path: str, **kwargs) -> xr.Dataset:
        return xr.open_dataset(input_path, **kwargs)

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        dataset.to_netcdf(output_path)

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        import os

        temp_path = output_path + ".temp.nc"
        os.rename(output_path, temp_path)
        old_ds = xr.open_dataset(temp_path)
        new_ds = xr.concat(
            [old_ds, dataset],
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
        )
        # noinspection PyUnresolvedReferences
        new_ds.to_netcdf(output_path)
        old_ds.close()
        rimraf(temp_path)

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            import netCDF4

            ds = netCDF4.Dataset(output_path, "r+")
            ds.setncatts(global_attrs)
            ds.close()


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
class ZarrDatasetIO(DatasetIO):
    """A dataset I/O that reads from / writes to Zarr directories or archives."""

    def __init__(self):
        super().__init__(FORMAT_NAME_ZARR)

    def fitness(self, path: str, path_type: str = None) -> float:
        ext = _get_ext(path)
        ext_value = 0.0
        type_value = 0.0
        if ext == ".zarr":
            ext_value = 1.0
            if path_type == "dir":
                type_value = 1.0
            elif path_type == "url" or path_type is None:
                type_value = 0.5
            else:
                ext_value = 0.0
        else:
            lower_path = path.lower()
            if lower_path.endswith(".zarr.zip"):
                ext_value = 1.0
                if path_type == "file":
                    type_value = 1.0
                elif path_type is None:
                    type_value = 0.5
                else:
                    ext_value = 0.0
            else:
                if path_type == "dir":
                    type_value = 1.0
                elif path_type == "url":
                    type_value = 0.5
        return (3 * ext_value + type_value) / 4

    def read(
        self,
        path: str,
        s3_kwargs: Dict[str, Any] = None,
        s3_client_kwargs: Dict[str, Any] = None,
        max_cache_size: int = None,
        **kwargs,
    ) -> xr.Dataset:
        """Read dataset from some Zarr storage.

        Args:
            path: File path or object storage URL.
            s3_kwargs: if *path* is an object storage URL, keyword-
                arguments passed to S3 file system, that is
                `s3fs.S3FileSystem(**s3_kwargs, ...)`.
            s3_client_kwargs: if *path* is an object storage URL,
                keyword-arguments passed to S3 (boto3) client, that is
                `s3fs.S3FileSystem(..., client_kwargs=s3_client_kwargs)`.
            max_cache_size: if this is a positive integer, the store
                will be wrapped in an in-memory cache, that is
                `store = zarr.LRUStoreCache(store, max_size=max_cache_size)`.
            **kwargs: Keyword-arguments passed to xarray Zarr adapter,
                that is `xarray.open_zarr(..., **kwargs)`. In
                addition, the parameter **

        Returns:

        """
        path_or_store = path
        consolidated = False
        if isinstance(path, str):
            path_or_store, consolidated = get_path_or_s3_store(
                path_or_store,
                s3_kwargs=s3_kwargs,
                s3_client_kwargs=s3_client_kwargs,
                mode="r",
            )
            if max_cache_size is not None and max_cache_size > 0:
                path_or_store = zarr.LRUStoreCache(
                    path_or_store, max_size=max_cache_size
                )
        return xr.open_zarr(path_or_store, consolidated=consolidated, **kwargs)

    def write(
        self,
        dataset: xr.Dataset,
        output_path: str,
        compressor: Dict[str, Any] = None,
        chunksizes: Dict[str, int] = None,
        packing: Dict[str, Dict[str, Any]] = None,
        s3_kwargs: Dict[str, Any] = None,
        s3_client_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        path_or_store, _ = get_path_or_s3_store(
            output_path,
            s3_kwargs=s3_kwargs,
            s3_client_kwargs=s3_client_kwargs,
            mode="w",
        )
        encoding = self._get_write_encodings(dataset, compressor, chunksizes, packing)
        consolidated = kwargs.pop("consolidated", True)
        dataset.to_zarr(
            path_or_store,
            mode="w",
            encoding=encoding,
            consolidated=consolidated,
            **kwargs,
        )

    @classmethod
    def _get_write_encodings(cls, dataset, compressor, chunksizes, packing):
        encoding = None
        if chunksizes:
            encoding = {}
            for var_name in dataset.data_vars:
                var = dataset[var_name]
                chunks: List[int] = []
                for i in range(len(var.dims)):
                    dim_name = var.dims[i]
                    if dim_name in chunksizes:
                        chunks.append(chunksizes[dim_name])
                    else:
                        chunks.append(var.shape[i])
                encoding[var_name] = dict(chunks=chunks)
        if packing:
            if encoding:
                for var_name in packing.keys():
                    if var_name in encoding.keys():
                        encoding[var_name].update(dict(packing[var_name]))
                    else:
                        encoding[var_name] = dict(packing[var_name])
            else:
                encoding = {}
                for var_name in packing.keys():
                    encoding[var_name] = dict(packing[var_name])

        if compressor:
            compressor = zarr.Blosc(**compressor)

            if encoding:
                for var_name in encoding.keys():
                    encoding[var_name].update(compressor=compressor)
            else:
                encoding = {
                    var_name: dict(compressor=compressor)
                    for var_name in dataset.data_vars
                }
        return encoding

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        append_time_slice(output_path, dataset)

    def insert(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        insert_time_slice(output_path, index, dataset)

    def replace(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        replace_time_slice(output_path, index, dataset)

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            import zarr

            ds = zarr.open_group(output_path, mode="r+", **kwargs)
            ds.attrs.update(global_attrs)
            zarr.consolidate_metadata(output_path)


# noinspection PyAbstractClass
@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
class CsvDatasetIO(DatasetIO):
    """A dataset I/O that reads from / writes to CSV files."""

    def __init__(self):
        super().__init__(FORMAT_NAME_CSV)

    def fitness(self, path: str, path_type: str = None) -> float:
        if path_type == "dir":
            return 0.0
        ext = _get_ext(path)
        ext_value = {".csv": 1.0, ".txt": 0.5, ".dat": 0.2}.get(ext, 0.1)
        type_value = {"file": 1.0, "url": 0.5, None: 0.5}.get(path_type, 0.0)
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        return xr.Dataset.from_dataframe(pd.read_csv(path, **kwargs))

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        dataset.to_dataframe().to_csv(output_path, **kwargs)


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def rimraf(*paths: str):
    """The UNIX command `rm -rf` for xcube.
    Recursively remove directory or single file.

    Args:
        *paths: one or more directories or files
    """
    for path in paths:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except OSError:
                warnings.warn(f"failed to remove file {path}")
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                warnings.warn(f"failed to remove file {path}")
                pass


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def get_path_or_s3_store(
    path_or_url: str,
    s3_kwargs: Mapping[str, Any] = None,
    s3_client_kwargs: Mapping[str, Any] = None,
    mode: str = "r",
) -> Tuple[Union[str, Dict], bool]:
    """If *path_or_url* is an object storage URL, return a object storage
    Zarr store (mapping object) using *s3_client_kwargs* and *mode* and
    a flag indicating whether the Zarr datasets is consolidated.

    Otherwise *path_or_url* is interpreted as a local file system path,
    returned as-is plus a flag indicating whether the Zarr datasets
    is consolidated.

    Args:
        path_or_url: A path or a URL.
        s3_kwargs: keyword arguments for S3 file system.
        s3_client_kwargs: keyword arguments for S3 boto3 client.
        mode: access mode "r" or "w". "r" is default.

    Returns:
        A tuple (path_or_obs_store, consolidated).
    """
    if is_s3_url(path_or_url) or s3_kwargs is not None or s3_client_kwargs is not None:
        s3, root = parse_s3_fs_and_root(
            path_or_url,
            s3_kwargs=s3_kwargs,
            s3_client_kwargs=s3_client_kwargs,
            mode=mode,
        )
        consolidated = mode == "r" and s3.exists(f"{root}/.zmetadata")
        return (
            s3fs.S3Map(root=root, s3=s3, check=False, create=mode == "w"),
            consolidated,
        )
    else:
        consolidated = os.path.exists(os.path.join(path_or_url, ".zmetadata"))
        return path_or_url, consolidated


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def parse_s3_fs_and_root(
    s3_url: str,
    s3_kwargs: Mapping[str, Any] = None,
    s3_client_kwargs: Mapping[str, Any] = None,
    mode: str = "r",
) -> Tuple[s3fs.S3FileSystem, str]:
    """Parses *s3_url*, *s3_kwargs*, *s3_client_kwargs* and returns a
    new tuple (*obs_fs*, *root_path*). For example

    ```
    obs_fs, root_path = parse_s3_fs_and_root(s3_url, s3_kwargs, s3_client_kwargs)
    obs_map = s3fs.S3Map(root=root_path, s3=obs_fs)
    ```

    Args:
        s3_url: Object storage URL, e.g. "s3://bucket/root", or
            "https://bucket.s3.amazonaws.com/root".
        s3_kwargs: keyword arguments for S3 file system.
        s3_client_kwargs: keyword arguments for S3 boto3 client.
        mode: Access mode "r" or "w",  "r" is default.

    Returns:
        A tuple (*obs_fs*, *root_path*).
    """

    root, s3_kwargs, s3_client_kwargs = parse_s3_url_and_kwargs(
        s3_url, s3_kwargs=s3_kwargs, s3_client_kwargs=s3_client_kwargs
    )
    s3 = new_s3_file_system(
        s3_kwargs=s3_kwargs,
        s3_client_kwargs=s3_client_kwargs,
        check_path=root if mode == "r" else None,
    )
    return s3, root


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def new_s3_file_system(
    s3_kwargs: Mapping[str, Any] = None,
    s3_client_kwargs: Mapping[str, Any] = None,
    s3_config_param_name: str = "s3_kwargs",
    check_path: str = None,
) -> s3fs.S3FileSystem:
    """Wrapper for s3fs.S3FileSystem() constructor that issues warnings
    in case the file system can not be created.

    Args:
        s3_kwargs: keyword arguments for S3 file system.
        s3_client_kwargs: keyword arguments for S3 boto3 client.
        s3_config_param_name: the name of the configuration parameter.
        check_path: an optional root path. If given, we call
            s3.exists(check_path) as a check whether S3 file system is
            valid.

    Returns:
        A s3fs.S3FileSystem instance.
    """

    s3_kwargs = s3_kwargs or {}
    if "use_listings_cache" not in s3_kwargs:
        # The default is not to cache any directory listings
        # because we want current contents
        s3_kwargs["use_listings_cache"] = False
    client_kwargs = s3_kwargs.pop("client_kwargs", {})
    client_kwargs.update(s3_client_kwargs or {})
    try:
        s3 = s3fs.S3FileSystem(**s3_kwargs, client_kwargs=client_kwargs)
        if check_path is not None:
            # Force potential NoCredentialsError
            s3.exists(check_path)
        return s3
    except botocore.exceptions.NoCredentialsError:
        if not s3_kwargs.get("anon"):
            warnings.warn(
                "No object storage credentials were"
                " passed or found.\n"
                "If you intend to access a public object storage,"
                " please pass " + s3_config_param_name + '={"anon": True, ...}\n'
            )
        raise


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def parse_s3_url_and_kwargs(
    s3_url: str,
    s3_kwargs: Mapping[str, Any] = None,
    s3_client_kwargs: Mapping[str, Any] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Parses *obs_url*, *s3_kwargs*, *s3_client_kwargs* and returns a
    new tuple (*root*, *s3_kwargs*, *s3_client_kwargs*) with updated kwargs whose elements
    can be passed to the s3fs.S3FileSystem and s3fs.S3Map constructors as follows:

    ```python
        obs_fs = s3fs.S3FileSystem(**s3_kwargs, client_kwargs=s3_client_kwargs)
        obs_map = s3fs.S3Map(root=root, s3=obs_fs)
    ```

    Args:
        s3_url: Object storage URL, e.g. "s3://bucket/root", or
            "https://bucket.s3.amazonaws.com/root".
        s3_kwargs: keyword arguments for S3 file system.
        s3_client_kwargs: keyword arguments for S3 boto3 client.

    Returns:
        A tuple (root, s3_kwargs, s3_client_kwargs).
    """
    endpoint_url, root = split_s3_url(s3_url)

    new_s3_client_kwargs = dict(s3_client_kwargs) if s3_client_kwargs else dict()
    if endpoint_url:
        new_s3_client_kwargs["endpoint_url"] = endpoint_url

    # The following key + secret kwargs are no longer supported in client_kwargs and are now moved into s3_kwargs
    key = secret = None
    if "provider_access_key_id" in new_s3_client_kwargs:
        key = new_s3_client_kwargs.pop("provider_access_key_id")
    if "aws_access_key_id" in new_s3_client_kwargs:
        key = new_s3_client_kwargs.pop("aws_access_key_id")
    if "provider_secret_access_key" in new_s3_client_kwargs:
        secret = new_s3_client_kwargs.pop("provider_secret_access_key")
    if "aws_secret_access_key" in new_s3_client_kwargs:
        secret = new_s3_client_kwargs.pop("aws_secret_access_key")

    new_s3_kwargs = dict(key=key, secret=secret)
    if s3_kwargs:
        new_s3_kwargs.update(**s3_kwargs)

    return root, new_s3_kwargs, new_s3_client_kwargs


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def split_s3_url(path: str) -> Tuple[Optional[str], str]:
    """If *path* is a URL, return tuple (endpoint_url, root), otherwise (None, *path*)"""
    url = urllib3.util.parse_url(path)
    if all((url.scheme, url.host, url.path)) and url.scheme != "s3":
        if url.port is not None:
            endpoint_url = f"{url.scheme}://{url.host}:{url.port}"
        else:
            endpoint_url = f"{url.scheme}://{url.host}"
        root = url.path
        if root.startswith("/"):
            root = root[1:]
        return endpoint_url, root
    return None, path


@deprecated(_DEPRECATION_REASON, version=_DEPRECATION_VERSION)
def is_s3_url(path_or_url: str) -> bool:
    """Test if *path_or_url* is a potential object storage URL.

    Args:
        path_or_url: Path or URL to test.

    Returns:
        True, if so.
    """
    return (
        path_or_url.startswith("https://")
        or path_or_url.startswith("http://")
        or path_or_url.startswith("s3://")
    )
