# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from abc import ABC
from typing import Tuple, Optional

import fsspec
import rasterio
import rioxarray
import s3fs
import xarray as xr
import zarr
from rasterio.session import AWSSession

from xcube.core.zarrstore import LoggingZarrStore

# Note, we need the following reference to register the
# xarray property accessor
# noinspection PyUnresolvedReferences
from xcube.core.zarrstore import ZarrStoreHolder
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.fspath import is_local_fs
from xcube.util.jsonencoder import to_json_value
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.temp import new_temp_file
from ..accessor import FsDataAccessor
from ...datatype import DATASET_TYPE
from ...datatype import DataType
from ...error import DataStoreError

ZARR_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        log_access=JsonBooleanSchema(default=False),
        cache_size=JsonIntegerSchema(
            minimum=0,
        ),
        group=JsonStringSchema(
            description="Group path." " (a.k.a. path in zarr terminology.).",
            min_length=1,
        ),
        chunks=JsonObjectSchema(
            description="Optional chunk sizes along each dimension."
            ' Chunk size values may be None, "auto"'
            " or an integer value.",
            examples=[
                {"time": None, "lat": "auto", "lon": 90},
                {"time": 1, "y": 512, "x": 512},
            ],
            additional_properties=True,
        ),
        decode_cf=JsonBooleanSchema(
            description="Whether to decode these variables,"
            " assuming they were saved according to"
            " CF conventions.",
            default=True,
        ),
        mask_and_scale=JsonBooleanSchema(
            description="If True, replace array values equal"
            ' to attribute "_FillValue" with NaN. '
            ' Use "scale_factor" and "add_offset"'
            " attributes to compute actual values.",
            default=True,
        ),
        decode_times=JsonBooleanSchema(
            description="If True, decode times encoded in the"
            " standard NetCDF datetime format "
            "into datetime objects. Otherwise,"
            " leave them encoded as numbers.",
            default=True,
        ),
        decode_coords=JsonBooleanSchema(
            description='If True, decode the "coordinates"'
            " attribute to identify coordinates in "
            "the resulting dataset.",
            default=True,
        ),
        drop_variables=JsonArraySchema(
            items=JsonStringSchema(min_length=1),
        ),
        consolidated=JsonBooleanSchema(
            description="Whether to open the store using"
            " Zarr's consolidated metadata "
            "capability. Only works for stores that"
            " have already been consolidated.",
            default=False,
        ),
    ),
    required=[],
    additional_properties=False,
)

ZARR_WRITE_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        group=JsonStringSchema(
            description="Group path." " (a.k.a. path in zarr terminology.).",
            min_length=1,
        ),
        encoding=JsonObjectSchema(
            description="Nested dictionary with variable"
            " names as keys and dictionaries of"
            " variable specific encodings as values.",
            examples=[{"my_variable": {"dtype": "int16", "scale_factor": 0.1}}],
            additional_properties=True,
        ),
        consolidated=JsonBooleanSchema(
            description="If True (the default), consolidate all metadata"
            ' files ("**/.zarray", "**/.zattrs")'
            ' into a single top-level file ".zmetadata"',
            default=True,
        ),
        append_dim=JsonStringSchema(
            description="If set, the dimension on which the" " data will be appended.",
            min_length=1,
        ),
    ),
    additional_properties=False,
)


class DatasetFsDataAccessor(FsDataAccessor, ABC):
    """Opener/writer extension name: 'dataset:<format>:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return DATASET_TYPE


class DatasetZarrFsDataAccessor(DatasetFsDataAccessor):
    """Opener/writer extension name: 'dataset:zarr:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "zarr"

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(ZARR_OPEN_DATA_PARAMS_SCHEMA)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)
        zarr_store = fs.get_mapper(data_id)
        cache_size = open_params.pop("cache_size", None)
        if isinstance(cache_size, int) and cache_size > 0:
            zarr_store = zarr.LRUStoreCache(zarr_store, max_size=cache_size)
        log_access = open_params.pop("log_access", None)
        if log_access:
            zarr_store = LoggingZarrStore(zarr_store, name=f"zarr_store({data_id!r})")
        consolidated = open_params.pop(
            "consolidated", fs.exists(f"{data_id}/.zmetadata")
        )
        try:
            dataset = xr.open_zarr(zarr_store, consolidated=consolidated, **open_params)
        except ValueError as e:
            raise DataStoreError(f"Failed to open" f" dataset {data_id!r}: {e}") from e

        dataset.zarr_store.set(zarr_store)
        return dataset

    # noinspection PyMethodMayBeStatic
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(ZARR_WRITE_DATA_PARAMS_SCHEMA)

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        assert_instance(data_id, str, name="data_id")
        fs, root, write_params = self.load_fs(write_params)
        zarr_store = fs.get_mapper(data_id, create=True)
        log_access = write_params.pop("log_access", None)
        if log_access:
            zarr_store = LoggingZarrStore(zarr_store, name=f"zarr_store({data_id!r})")
        consolidated = write_params.pop("consolidated", True)
        try:
            data.to_zarr(
                zarr_store,
                mode="w" if replace else None,
                consolidated=consolidated,
                **write_params,
            )
        except ValueError as e:
            raise DataStoreError(f"Failed to write" f" dataset {data_id!r}: {e}") from e
        return data_id

    def delete_data(self, data_id: str, **delete_params):
        fs, root, delete_params = self.load_fs(delete_params)
        delete_params.pop("recursive", None)
        fs.delete(data_id, recursive=True, **delete_params)


NETCDF_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        # TODO: add more from xr.open_dataset()
    ),
    additional_properties=True,
)

NETCDF_WRITE_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        # TODO: add more from ds.to_netcdf()
    ),
    additional_properties=True,
)


class DatasetNetcdfFsDataAccessor(DatasetFsDataAccessor):
    """Opener/writer extension name: 'dataset:netcdf:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "netcdf"

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(NETCDF_OPEN_DATA_PARAMS_SCHEMA)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = open_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'rb') as file:
        #     return xr.open_dataset(file, engine=engine, **open_params)

        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file(suffix=".nc")
            fs.get_file(data_id, file_path)
        engine = open_params.pop("engine", "netcdf4")
        return xr.open_dataset(file_path, engine=engine, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(
            NETCDF_WRITE_DATA_PARAMS_SCHEMA
        )

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        assert_instance(data_id, str, name="data_id")
        fs, root, write_params = self.load_fs(write_params)
        if not replace and fs.exists(data_id):
            raise DataStoreError(f"Data resource {data_id} already exists")

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = write_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'wb') as file:
        #     data.to_netcdf(file, engine=engine, **write_params)

        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file(suffix=".nc")
        engine = write_params.pop("engine", "netcdf4")
        data.to_netcdf(file_path, engine=engine, **write_params)
        if not is_local:
            fs.put_file(file_path, data_id)
        return data_id


GEOTIFF_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=256, default=512),
                JsonNumberSchema(minimum=256, default=512),
            ),
            default=[512, 512],
        ),
        overview_level=JsonIntegerSchema(
            default=None,
            nullable=True,
            description="GeoTIFF overview level. 0 is the first overview.",
        ),
    ),
    additional_properties=False,
)


class DatasetGeoTiffFsDataAccessor(DatasetFsDataAccessor):
    """
    Opener/writer extension name: 'dataset:geotiff:<protocol>'.
    """

    @classmethod
    def get_format_id(cls) -> str:
        return "geotiff"

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return GEOTIFF_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)

        if isinstance(fs.protocol, str):
            protocol = fs.protocol
        else:
            protocol = fs.protocol[0]
        if root is not None:
            file_path = protocol + "://" + root + "/" + data_id
        else:
            file_path = protocol + "://" + data_id
        tile_size = open_params.get("tile_size", (512, 512))
        overview_level = open_params.get("overview_level", None)
        return self.open_dataset(
            fs, file_path, tile_size, overview_level=overview_level
        )

    @classmethod
    def create_env_session(cls, fs):
        if isinstance(fs, s3fs.S3FileSystem):
            aws_unsigned = bool(fs.anon)
            aws_session = AWSSession(
                aws_unsigned=aws_unsigned,
                aws_secret_access_key=fs.secret,
                aws_access_key_id=fs.key,
                aws_session_token=fs.token,
                region_name=fs.client_kwargs.get("region_name", "eu-central-1"),
            )
            return rasterio.env.Env(
                session=aws_session, aws_no_sign_request=aws_unsigned
            )
        return rasterio.env.NullContextManager()

    @classmethod
    def open_dataset_with_rioxarray(
        cls, file_path, overview_level, tile_size
    ) -> rioxarray.raster_array:
        return rioxarray.open_rasterio(
            file_path,
            overview_level=overview_level,
            chunks=dict(zip(("x", "y"), tile_size)),
        )

    @classmethod
    def open_dataset(
        cls,
        fs,
        file_path: str,
        tile_size: Tuple[int, int],
        overview_level: Optional[int] = None,
    ) -> xr.Dataset:
        """A method to open the cog/geotiff dataset using rioxarray.

        Args:
            fs: abstract file system
            file_path: path of the file
            overview_level: the overview level of GeoTIFF, 0 is the first
               overview and None means full resolution.
            tile_size: tile size as tuple.

        Returns:
            An xarray.Dataset instance
        """

        if isinstance(fs, fsspec.AbstractFileSystem):
            with cls.create_env_session(fs):
                array = cls.open_dataset_with_rioxarray(
                    file_path, overview_level, tile_size
                )
        else:
            assert_true(fs is None, message="invalid type for fs")
        arrays = {}
        if array.ndim == 3:
            for i in range(array.shape[0]):
                name = f'{array.name or "band"}_{i + 1}'
                dims = array.dims[-2:]
                coords = {
                    n: v
                    for n, v in array.coords.items()
                    if n in dims or n == "spatial_ref"
                }
                band_data = array.data[i, :, :]
                arrays[name] = xr.DataArray(
                    band_data, coords=coords, dims=dims, attrs=dict(**array.attrs)
                )
        elif array.ndim == 2:
            name = f'{array.name or "band"}'
            arrays[name] = array
        else:
            raise RuntimeError("number of dimensions must be 2 or 3")

        dataset = xr.Dataset(arrays, attrs=dict(source=file_path))
        # For CRS, rioxarray uses variable "spatial_ref" by default
        if "spatial_ref" in array.coords:
            for data_var in dataset.data_vars.values():
                data_var.attrs["grid_mapping"] = "spatial_ref"

        # rioxarray may return non-JSON-serializable metadata
        # attribute values.
        # We have seen _FillValue of type np.uint8
        cls._sanitize_dataset_attrs(dataset)

        return dataset

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        raise NotImplementedError("Writing of GeoTIFF not yet supported")

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        raise NotImplementedError("Writing of GeoTIFF not yet supported")

    @classmethod
    def _sanitize_dataset_attrs(cls, dataset):
        dataset.attrs.update(to_json_value(dataset.attrs))
        for var in dataset.variables.values():
            var.attrs.update(to_json_value(var.attrs))
