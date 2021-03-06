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

import warnings
from typing import Tuple, Sequence, Mapping, Optional, Dict, Any, Union, Hashable

import dask.array
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.geom import get_dataset_bounds
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store.typespecifier import TYPE_SPECIFIER_ANY
from xcube.core.store.typespecifier import TYPE_SPECIFIER_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store.typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store.typespecifier import TypeSpecifier
from xcube.core.store.typespecifier import get_type_specifier
from xcube.core.timecoord import get_end_time_from_attrs
from xcube.core.timecoord import get_start_time_from_attrs
from xcube.core.timecoord import get_time_range_from_data
from xcube.core.timecoord import remove_time_part_from_isoformat
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_not_none
from xcube.util.ipython import register_json_formatter
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonDateSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.schema.CubeSchema class
#   xcube.webapi.context.DatasetDescriptor type
#   responses of xcube.webapi.controllers.catalogue
# TODO: write tests
# TODO: validate params


def new_data_descriptor(data_id: str, data: Any, require: bool = False) -> 'DataDescriptor':
    if isinstance(data, xr.Dataset):
        coords = _build_variable_descriptor_dict(data.coords)
        data_vars = _build_variable_descriptor_dict(data.data_vars)
        spatial_res = _determine_spatial_res(data)
        bbox = _determine_bbox(data)
        time_coverage_start, time_coverage_end = _determine_time_coverage(data)
        time_period = _determine_time_period(data)
        return DatasetDescriptor(data_id=data_id,
                                 type_specifier=get_type_specifier(data),
                                 dims={str(k): v for k, v in data.dims.items()},
                                 coords=coords,
                                 data_vars=data_vars,
                                 bbox=bbox,
                                 time_range=(time_coverage_start, time_coverage_end),
                                 time_period=time_period,
                                 spatial_res=spatial_res,
                                 attrs=data.attrs)
    elif isinstance(data, MultiLevelDataset):
        # TODO: implement me: data -> MultiLevelDatasetDescriptor
        return MultiLevelDatasetDescriptor(data_id=data_id, num_levels=5)
    elif isinstance(data, gpd.GeoDataFrame):
        # TODO: implement me: data -> GeoDataFrameDescriptor
        return GeoDataFrameDescriptor(data_id=data_id)
    elif not require:
        return DataDescriptor(data_id=data_id, type_specifier=TYPE_SPECIFIER_ANY)
    raise NotImplementedError()


class DataDescriptor(JsonObject):
    """
    A generic descriptor for any data.
    Also serves as a base class for more specific data descriptors.

    :param data_id: An identifier for the data
    :param type_specifier: A type specifier for the data
    :param crs: A coordinate reference system identifier, as an EPSG, PROJ or WKT string
    :param bbox: A bounding box of the data
    :param time_range: Start and end time delimiting this data's temporal extent
    :param time_period: The data's periodicity if it is evenly temporally resolved.
    :param open_params_schema: A JSON schema describing the parameters that may be used to open
    this data.
    """

    def __init__(self,
                 data_id: str,
                 type_specifier: Union[str, TypeSpecifier],
                 *,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 open_params_schema: JsonObjectSchema = None,
                 **additional_properties):
        assert_given(data_id, 'data_id')
        assert_instance(data_id, (str, TypeSpecifier))
        if additional_properties:
            warnings.warn(f'Additional properties received;'
                          f' will be ignored: {additional_properties}')
        self.data_id = data_id
        self.type_specifier = TypeSpecifier.normalize(type_specifier)
        self.crs = crs
        self.bbox = tuple(bbox) if bbox else None
        self.time_range = tuple(time_range) if time_range else None
        self.time_period = time_period
        self.open_params_schema = open_params_schema

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                data_id=JsonStringSchema(min_length=1),
                type_specifier=TypeSpecifier.get_schema(),
                crs=JsonStringSchema(min_length=1),
                bbox=JsonArraySchema(items=[JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema()]),
                time_range=JsonDateSchema.new_range(nullable=True),
                time_period=JsonStringSchema(min_length=1),
                open_params_schema=JsonObjectSchema(additional_properties=True),
            ),
            required=['data_id', 'type_specifier'],
            additional_properties=True,
            factory=cls)

    # TODO: reactivate once needed
    #
    # @classmethod
    # def from_dict(cls, mapping: Mapping[str, Any]) -> 'DataDescriptor':
    #     """Create new instance from a JSON-serializable dictionary"""
    #     assert_in('data_id', mapping)
    #     assert_in('type_specifier', mapping)
    #     if TYPE_SPECIFIER_DATASET.is_satisfied_by(mapping['type_specifier']):
    #         return DatasetDescriptor.from_dict(mapping)
    #     elif TYPE_SPECIFIER_GEODATAFRAME.is_satisfied_by(mapping['type_specifier']):
    #         return GeoDataFrameDescriptor.from_dict(mapping)
    #     return DataDescriptor(data_id=mapping['data_id'],
    #                           type_specifier=mapping['type_specifier'],
    #                           crs=mapping.get('crs'),
    #                           bbox=mapping.get('bbox'),
    #                           time_range=mapping.get('time_range'),
    #                           time_period=mapping.get('time_period'),
    #                           open_params_schema=mapping.get('open_params_schema'))


class DatasetDescriptor(DataDescriptor):
    """
    A descriptor for a gridded, N-dimensional dataset represented by xarray.Dataset.
    Comprises a description of the data variables contained in the dataset.

    :param data_id: An identifier for the data
    :param type_specifier: A type specifier for the data
    :param crs: A coordinate reference system identifier, as an EPSG, PROJ or WKT string
    :param bbox: A bounding box of the data
    :param time_range: Start and end time delimiting this data's temporal extent (see
        https://github.com/dcs4cop/xcube/blob/master/docs/source/storeconv.md#date-time-and-duration-specifications )
    :param time_period: The data's periodicity if it is evenly temporally resolved (see
        https://github.com/dcs4cop/xcube/blob/master/docs/source/storeconv.md#date-time-and-duration-specifications )
    :param spatial_res: The spatial extent of a pixel in crs units
    :param dims: A mapping of the dataset's dimensions to their sizes
    :param coords: mapping of the dataset's data coordinate names to VariableDescriptors
        (``xcube.core.store.VariableDescriptor``).
    :param data_vars: A mapping of the dataset's variable names to VariableDescriptors
        (``xcube.core.store.VariableDescriptor``).
    :param attrs: A mapping containing arbitrary attributes of the dataset
    :param open_params_schema: A JSON schema describing the parameters that may be used to open
        this data
    """

    def __init__(self,
                 data_id: str,
                 *,
                 type_specifier: Union[str, TypeSpecifier] = TYPE_SPECIFIER_DATASET,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 spatial_res: float = None,
                 dims: Mapping[str, int] = None,
                 coords: Mapping[str, 'VariableDescriptor'] = None,
                 data_vars: Mapping[str, 'VariableDescriptor'] = None,
                 attrs: Mapping[Hashable, any] = None,
                 open_params_schema: JsonObjectSchema = None,
                 **additional_properties):
        super().__init__(data_id=data_id,
                         type_specifier=type_specifier,
                         crs=crs,
                         bbox=bbox,
                         time_range=time_range,
                         time_period=time_period,
                         open_params_schema=open_params_schema)
        self.type_specifier.assert_satisfies(TYPE_SPECIFIER_DATASET)
        if additional_properties:
            warnings.warn(f'Additional properties received;'
                          f' will be ignored: {additional_properties}')
        self.dims = dict(dims) if dims else None
        self.spatial_res = spatial_res
        self.coords = coords if coords else None
        self.data_vars = data_vars if data_vars else None
        self.attrs = _attrs_to_json(attrs) if attrs else None

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        schema = super().get_schema()
        schema.properties.update(
            dims=JsonObjectSchema(additional_properties=JsonIntegerSchema(minimum=0)),
            spatial_res=JsonNumberSchema(exclusive_minimum=0.0),
            coords=JsonObjectSchema(additional_properties=VariableDescriptor.get_schema()),
            data_vars=JsonObjectSchema(additional_properties=VariableDescriptor.get_schema()),
            attrs=JsonObjectSchema(additional_properties=True),
        )
        schema.required = ['data_id']
        schema.additional_properties = False
        schema.factory = cls
        return schema


class VariableDescriptor(JsonObject):
    """
    A descriptor for dataset variable represented by xarray.DataArray instances.
    They are part of dataset descriptor for an gridded, N-dimensional dataset represented by
    xarray.Dataset.

    :param name: The variable name
    :param dtype: The data type of the variable.
    :param dims: A list of the names of the variable's dimensions.
    :param chunks: A list of the chunk sizes of the variable's dimensions
    :param attrs: A mapping containing arbitrary attributes of the variable
    """

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 *,
                 chunks: Sequence[int] = None,
                 attrs: Mapping[Hashable, any] = None,
                 **additional_properties):
        assert_given(name, 'name')
        assert_given(dtype, 'dtype')
        assert_not_none(dims, 'dims')
        if additional_properties:
            warnings.warn(f'Additional properties received; will be ignored: {additional_properties}')
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.chunks = tuple(chunks) if chunks else None
        self.attrs = _attrs_to_json(attrs) if attrs else None

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                name=JsonStringSchema(min_length=1),
                dtype=JsonStringSchema(min_length=1),
                dims=JsonArraySchema(items=JsonStringSchema(min_length=1)),
                chunks=JsonArraySchema(items=JsonIntegerSchema(minimum=0)),
                attrs=JsonObjectSchema(additional_properties=True),
            ),
            required=['name', 'dtype', 'dims'],
            additional_properties=False,
            factory=cls)


class MultiLevelDatasetDescriptor(DatasetDescriptor):
    """
    A descriptor for a gridded, N-dimensional, multi-level, multi-resolution dataset represented by
    xcube.core.mldataset.MultiLevelDataset.

    :param data_id: An identifier of the multi-level dataset
    :param num_levels: The number of levels of this multi-level dataset
    :param type_specifier: A type specifier for the multi-level dataset
    """

    def __init__(self,
                 data_id: str,
                 num_levels: int,
                 *,
                 type_specifier: Union[str, TypeSpecifier] = TYPE_SPECIFIER_MULTILEVEL_DATASET,
                 **kwargs):
        assert_given(data_id, 'data_id')
        assert_given(num_levels, 'num_levels')
        super().__init__(data_id=data_id, type_specifier=type_specifier, **kwargs)
        self.type_specifier.assert_satisfies(TYPE_SPECIFIER_MULTILEVEL_DATASET)
        self.num_levels = num_levels

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        schema = super().get_schema()
        schema.properties.update(
            num_levels=JsonIntegerSchema(minimum=1),
        )
        schema.required = ['data_id', 'num_levels']
        schema.additional_properties = False
        schema.factory = cls
        return schema


class GeoDataFrameDescriptor(DataDescriptor):
    """
    A descriptor for a geo-vector dataset represented by a geopandas.GeoDataFrame instance.

    :param data_id: An identifier of the geopandas.GeoDataFrame
    :param feature_schema: A schema describing the properties of the vector data
    :param kwargs: Parameters passed to super :class:DataDescriptor
    """

    def __init__(self,
                 data_id: str,
                 *,
                 type_specifier=TYPE_SPECIFIER_GEODATAFRAME,
                 feature_schema: JsonObjectSchema = None,
                 **kwargs):
        super().__init__(data_id=data_id,
                         type_specifier=type_specifier,
                         **kwargs)
        self.type_specifier.assert_satisfies(TYPE_SPECIFIER_GEODATAFRAME)
        self.feature_schema = feature_schema

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        schema = super().get_schema()
        schema.properties.update(
            feature_schema=JsonObjectSchema(additional_properties=True),
        )
        schema.required = ['data_id']
        schema.additional_properties = False
        schema.factory = cls
        return schema


register_json_formatter(DataDescriptor)
register_json_formatter(DatasetDescriptor)
register_json_formatter(VariableDescriptor)
register_json_formatter(MultiLevelDatasetDescriptor)
register_json_formatter(GeoDataFrameDescriptor)


#############################################################################
# Implementation helpers


def _build_variable_descriptor_dict(variables) -> Mapping[str, 'VariableDescriptor']:
    return {
        str(var_name):
            VariableDescriptor(
                name=str(var_name),
                dtype=str(var.dtype),
                dims=var.dims,
                chunks=tuple([max(chunk) for chunk in tuple(var.chunks)]) if var.chunks else None,
                attrs=var.attrs
            )
        for var_name, var in variables.items()}


def _determine_bbox(data: xr.Dataset) -> Optional[Tuple[float, float, float, float]]:
    try:
        return get_dataset_bounds(data)
    except ValueError:
        if 'geospatial_lon_min' in data.attrs and \
                'geospatial_lat_min' in data.attrs and \
                'geospatial_lon_max' in data.attrs and \
                'geospatial_lat_max' in data.attrs:
            return (data.geospatial_lat_min, data.geospatial_lon_min,
                    data.geospatial_lat_max, data.geospatial_lon_max)


def _determine_spatial_res(data: xr.Dataset):
    # TODO get rid of these hard-coded coord names as soon as new resampling is available
    lat_dimensions = ['lat', 'latitude', 'y']
    for lat_dimension in lat_dimensions:
        if lat_dimension in data:
            lat_diff = data[lat_dimension].diff(dim=data[lat_dimension].dims[0]).values
            lat_res = lat_diff[0]
            lat_regular = np.allclose(lat_res, lat_diff, 1e-8)
            if lat_regular:
                return float(abs(lat_res))


def _determine_time_coverage(data: xr.Dataset):
    start_time, end_time = get_time_range_from_data(data)
    if start_time is not None:
        try:
            start_time = remove_time_part_from_isoformat(pd.to_datetime(start_time).isoformat())
        except TypeError:
            start_time = None
    if start_time is None:
        start_time = get_start_time_from_attrs(data)
    if end_time is not None:
        try:
            end_time = remove_time_part_from_isoformat(pd.to_datetime(end_time).isoformat())
        except TypeError:
            end_time = None
    if end_time is None:
        end_time = get_end_time_from_attrs(data)
    return start_time, end_time


def _determine_time_period(data: xr.Dataset):
    if 'time' in data and len(data['time'].values) > 1:
        time_diff = data['time'].diff(dim=data['time'].dims[0]).values.astype(np.float64)
        time_res = time_diff[0]
        time_regular = np.allclose(time_res, time_diff, 1e-8)
        if time_regular:
            time_period = pd.to_timedelta(time_res).isoformat()
            # remove leading P
            time_period = time_period[1:]
            # removing sub-day precision
            return time_period.split('T')[0]


def _attrs_to_json(attrs: Mapping[Hashable, Any]) -> Optional[Dict[str, Any]]:
    new_attrs: Dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, dask.array.Array):
            v = np.array(v).tolist()
        if isinstance(v, float) and np.isnan(v):
            v = None
        new_attrs[str(k)] = v
    return new_attrs
