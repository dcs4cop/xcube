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

import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Collection, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

from xcube.constants import CRS_WKT_EPSG_4326
from xcube.constants import EXTENSION_POINT_INPUT_PROCESSORS
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import rectify_dataset
from xcube.core.timecoord import to_time_in_days_since_1970
from xcube.util.plugin import ExtensionComponent
from xcube.util.plugin import get_extension_registry


class ReprojectionInfo:
    """
    Characterize input datasets so we can reproject.

    :param xy_names: Names of variables providing the spatial x- and y-coordinates,
           e.g. ('longitude', 'latitude')
    :param xy_tp_names: Optional names of tie-point variables providing the spatial y- and y-coordinates,
           e.g. ('TP_longitude', 'TP_latitude')
    :param xy_crs: Optional spatial reference system, e.g. 'EPSG:4326' or WKT or proj4 mapping
    :param xy_gcp_step: Optional step size for collecting ground control points from spatial
           coordinate arrays denoted by *xy_names*.
    :param xy_tp_gcp_step: Optional step size for collecting ground control points from spatial
           coordinate arrays denoted by *xy_tp_names*.
    """

    def __init__(self,
                 xy_names: Tuple[str, str] = None,
                 xy_tp_names: Tuple[str, str] = None,
                 xy_crs: Any = None,
                 xy_gcp_step: Union[int, Tuple[int, int]] = None,
                 xy_tp_gcp_step: Union[int, Tuple[int, int]] = None):
        self._xy_names = self._assert_name_pair('xy_names', xy_names)
        self._xy_tp_names = self._assert_name_pair('xy_tp_names', xy_tp_names)
        self._xy_crs = xy_crs
        self._xy_gcp_step = self._assert_step_pair('xy_gcp_step', xy_gcp_step)
        self._xy_tp_gcp_step = self._assert_step_pair('xy_tp_gcp_step', xy_tp_gcp_step)

    def derive(self,
               xy_names: Tuple[str, str] = None,
               xy_tp_names: Tuple[str, str] = None,
               xy_crs: Any = None,
               xy_gcp_step: Union[int, Tuple[int, int]] = None,
               xy_tp_gcp_step: Union[int, Tuple[int, int]] = None):
        return ReprojectionInfo(self.xy_names if xy_names is None else xy_names,
                                xy_tp_names=self.xy_tp_names if xy_tp_names is None else xy_tp_names,
                                xy_crs=self.xy_crs if xy_crs is None else xy_crs,
                                xy_gcp_step=self.xy_gcp_step if xy_gcp_step is None else xy_gcp_step,
                                xy_tp_gcp_step=self.xy_tp_gcp_step if xy_tp_gcp_step is None else xy_tp_gcp_step)

    @property
    def xy_names(self) -> Optional[Tuple[str, str]]:
        return self._xy_names

    @property
    def xy_tp_names(self) -> Optional[Tuple[str, str]]:
        return self._xy_tp_names

    @property
    def xy_crs(self) -> Any:
        return self._xy_crs

    @property
    def xy_gcp_step(self) -> Optional[int]:
        return self._xy_gcp_step

    @property
    def xy_tp_gcp_step(self) -> Optional[int]:
        return self._xy_tp_gcp_step

    def _assert_name_pair(self, keyword: str, value):
        if value is not None:
            v1, v2 = value
            self._assert_name(keyword, v1)
            self._assert_name(keyword, v2)
            return v1, v2
        return value

    def _assert_step_pair(self, keyword: str, value):
        if value is not None:
            if isinstance(value, int):
                v1, v2 = value, value
            else:
                v1, v2 = value
            self._assert_step(keyword, v1)
            self._assert_step(keyword, v2)
            return v1, v2
        return value

    def _assert_name(self, keyword: str, value):
        if value is None:
            raise ValueError(f'invalid {keyword}, missing name')
        if not isinstance(value, str) or not value:
            raise ValueError(f'invalid {keyword}, name must be a non-empty string')

    def _assert_step(self, keyword: str, value):
        if value is None:
            raise ValueError(f'invalid {keyword}, missing name')
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f'invalid {keyword}, step must be an integer number')


class InputProcessor(ExtensionComponent, metaclass=ABCMeta):
    """
    Read and process inputs for the gen tool.

    An InputProcessor can be configured by the following parameters:

    * ``input_reader``: The input format identifier. Required, no default.

    :param name: A unique input processor identifier.
    """

    def __init__(self, name: str, **parameters):
        super().__init__(EXTENSION_POINT_INPUT_PROCESSORS, name)
        self._parameters = {**self.default_parameters, **parameters}
        if 'input_reader' not in self._parameters:
            raise ValueError('missing input_reader in input processor parameters')

    @property
    def description(self) -> str:
        """
        :return: A description for this input processor
        """
        return self.get_metadata_attr('description', '')

    @property
    def default_parameters(self) -> Dict[str, Any]:
        return {}

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self._parameters

    @property
    def input_reader(self) -> str:
        return self.parameters['input_reader']

    @property
    def input_reader_params(self) -> dict:
        """
        :return: The input reader parameters for this input processor.
        """
        return self.parameters.get('input_reader_params', {})

    @abstractmethod
    def get_time_range(self, dataset: xr.Dataset) -> Optional[Tuple[float, float]]:
        """
        Return a tuple of two floats representing start/stop time (which may be same) in days since 1970.
        :param dataset: The dataset.
        :return: The time-range tuple of the dataset or None.
        """
        raise NotImplementedError()

    def get_extra_vars(self, dataset: xr.Dataset) -> Optional[Collection[str]]:
        """
        Get a set of names of variables that are required as input for the pre-processing and processing
        steps and should therefore not be dropped.
        However, the processing or post-processing steps may later remove them.

        Returns ``None`` by default.

        :param dataset: The dataset.
        :return: Collection of names of variables to be prevented from being dropping.
        """
        return None

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Do any pre-processing before reprojection.
        All variables in the output dataset must be 2D arrays with dimensions "lat" and "lon", in this order.
        For example, perform dataset validation, masking, and/or filtering using provided configuration parameters.

        The default implementation returns the unchanged *dataset*.

        :param dataset: The dataset.
        :return: The pre-processed dataset or the original one, if no pre-processing is required.
        """
        return dataset

    @abstractmethod
    def process(self,
                dataset: xr.Dataset,
                geo_coding: GridMapping,
                output_geom: GridMapping,
                output_resampling: str,
                include_non_spatial_vars=False) -> xr.Dataset:
        """
        Perform spatial transformation into the cube's WGS84 SRS such that all variables in the output dataset
        * must be 2D arrays with dimensions "lat" and "lon", in this order, and
        * must have shape (*dst_size[-1]*, *dst_size[-2]*), and
        * must have *dst_region* as their bounding box in geographic coordinates.

        :param dataset: The input dataset.
        :param geo_coding: The input's geo-coding.
        :param output_geom: The output's spatial image geometry.
        :param output_resampling: The output's spatial resampling method.
        :param include_non_spatial_vars: Whether to include non-spatial variables in the output.
        :return: The transformed output dataset or the original one, if no transformation is required.
        """
        raise NotImplementedError()

    def post_process(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Do any post-processing transformation. The input is a 3D array with dimensions ("time", "lat", "lon").
        Post-processing may, for example, generate new "wavelength" dimension for variables whose name follow
        a certain pattern.

        The default implementation returns the unchanged *dataset*.

        :param dataset: The dataset.
        :return: The post-processed dataset or the original one, if no post-processing is required.
        """
        return dataset


class XYInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Read and process inputs for the gen tool.

    An XYInputProcessor can be configured by the following parameters:

    * ``input_reader``: The input format identifier.
        Required, no default.
    * ``xy_names``: A tuple of names of the variable providing x,y geo-locations.
        Optional, looked up automatically if not given e.g. ``("lon", "lat")``.
    * ``xy_tp_names``: A tuple of names of the variable providing x,y tie-point geo-locations.
        Optional, no default.
    * ``xy_crs``: A WKT string that identifies the x,y coordinate reference system (CRS).
        Optional, no default.
    * ``xy_gcp_step``: An integer or tuple of integers that is used to sub-sample x,y coordinate variables
        for extracting ground control points (GCP).
        Optional, no default.
    * ``xy_tp_gcp_step``: An integer or tuple of integers that is used to sub-sample x,y tie-point coordinate variables
        for extracting ground control points (GCP).
        Optional, no default.
    """

    @property
    def default_parameters(self) -> Dict[str, Any]:
        default_parameters = super().default_parameters
        default_parameters.update(xy_names=('lon', 'lat'))
        return default_parameters

    def get_reprojection_info(self, dataset: xr.Dataset) -> ReprojectionInfo:
        """
        Information about special fields in input datasets used for reprojection.
        :param dataset: The dataset.
        :return: The reprojection information of the dataset or None.
        """
        parameters = self.parameters
        return ReprojectionInfo(xy_names=parameters.get('xy_names', ('lon', 'lat')),
                                xy_tp_names=parameters.get('xy_tp_names'),
                                xy_crs=parameters.get('xy_crs'),
                                xy_gcp_step=parameters.get('xy_gcp_step'),
                                xy_tp_gcp_step=parameters.get('xy_tp_gcp_step'))

    def get_extra_vars(self, dataset: xr.Dataset) -> Optional[Collection[str]]:
        """
        Return the names of variables containing spatial coordinates.
        They should not be removed, as they are required for the reprojection.
        """
        reprojection_info = self.get_reprojection_info(dataset)
        if reprojection_info is None:
            return dataset
        extra_vars = set()
        if reprojection_info.xy_names:
            extra_vars.update(set(reprojection_info.xy_names))
        if reprojection_info.xy_tp_names:
            extra_vars.update(set(reprojection_info.xy_tp_names))
        return extra_vars

    def process(self,
                dataset: xr.Dataset,
                geo_coding: GridMapping,
                output_geom: GridMapping,
                output_resampling: str,
                include_non_spatial_vars=False) -> xr.Dataset:
        """
        Perform reprojection using tie-points / ground control points.
        """
        reprojection_info = self.get_reprojection_info(dataset)

        warn_prefix = 'unsupported argument in np-GCP rectification mode'
        if reprojection_info.xy_crs is not None:
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'reprojection_info.xy_crs = {reprojection_info.xy_crs!r}')
        if reprojection_info.xy_tp_names is not None:
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'reprojection_info.xy_tp_names = {reprojection_info.xy_tp_names!r}')
        if reprojection_info.xy_gcp_step is not None:
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'reprojection_info.xy_gcp_step = {reprojection_info.xy_gcp_step!r}')
        if reprojection_info.xy_tp_gcp_step is not None:
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'reprojection_info.xy_tp_gcp_step = {reprojection_info.xy_tp_gcp_step!r}')
        if output_resampling != 'Nearest':
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'dst_resampling = {output_resampling!r}')
        if include_non_spatial_vars:
            warnings.warn(f'{warn_prefix}: ignoring '
                          f'include_non_spatial_vars = {include_non_spatial_vars!r}')

        geo_coding = geo_coding.derive(xy_var_names=(reprojection_info.xy_names[0],
                                                     reprojection_info.xy_names[1]))

        dataset = rectify_dataset(dataset,
                                  compute_subset=False,
                                  source_gm=geo_coding,
                                  target_gm=output_geom)
        if output_geom.is_tiled:
            # The following condition may become true,
            # if we have used rectified_dataset(input, ..., is_y_reverse=True)
            # In this case y-chunksizes will also be reversed. So that the first chunk is smaller than any other.
            # Zarr will reject such datasets, when written.
            if dataset.chunks.get('lat')[0] < dataset.chunks.get('lat')[-1]:
                dataset = dataset.chunk({'lat': output_geom.tile_height,
                                         'lon': output_geom.tile_width})
        if dataset is not None \
                and geo_coding.crs.is_geographic \
                and geo_coding.xy_var_names != ('lon', 'lat'):
            dataset = dataset.rename({geo_coding.xy_var_names[0]: 'lon',
                                      geo_coding.xy_var_names[1]: 'lat'})

        return dataset


class DefaultInputProcessor(XYInputProcessor):
    """
    Default input processor that expects input datasets to have the xcube standard format:

    * Have dimensions ``lat``, ``lon``, optionally ``time`` of length 1;
    * have coordinate variables ``lat[lat]``, ``lon[lat]``, ``time[time]`` (opt.), ``time_bnds[time, 2]`` (opt.);
    * have coordinate variables ``lat[lat]``, ``lon[lat]`` as decimal degrees on WGS84 ellipsoid,
      both linearly increasing with same constant delta;
    * have coordinate variable ``time[time]`` representing a date+time values with defined CF "units" attribute;
    * have any data variables of form ``<var>[time, lat, lon]``;
    * have global attribute pairs (``time_coverage_start``, ``time_coverage_end``), or (``start_time``, ``stop_time``)
      if ``time`` coordinate is missing.

    The default input processor can be configured by the following parameters:

    * ``input_reader``: The input format identifier.
        Required, defaults to ``"netcdf4"``.
    * ``xy_names``: A tuple of names of the variable providing x,y geo-locations.
        Optional, defaults to ``("lon", "lat")``.
    * ``xy_tp_names``: A tuple of names of the variable providing x,y tie-point geo-locations.
        Optional, no default.
    * ``xy_crs``: A WKT string that identifies the x,y coordinate reference system (CRS).
        Optional, defaults to WKT for EPSG:4326 (see ``xcube.constants.CRS_WKT_EPSG_4326`` constant).
    * ``xy_gcp_step``: An integer or tuple of integers that is used to sub-sample x,y coordinate variables
        for extracting ground control points (GCP).
        Optional, no default.
    * ``xy_tp_gcp_step``: An integer or tuple of integers that is used to sub-sample x,y tie-point coordinate variables
        for extracting ground control points (GCP).
        Optional, no default.
    """

    def __init__(self, **parameters):
        super().__init__('default', **parameters)

    @property
    def default_parameters(self) -> Dict[str, Any]:
        default_parameters = super().default_parameters
        default_parameters.update(input_reader='netcdf4',
                                  xy_names=('lon', 'lat'),
                                  xy_crs=CRS_WKT_EPSG_4326)
        return default_parameters

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        self._validate(dataset)

        if "time" in dataset.dims:
            # Remove time dimension of length 1.
            dataset = dataset.squeeze("time")

        return _normalize_lon_360(dataset)

    def get_time_range(self, dataset: xr.Dataset) -> Tuple[float, float]:
        time_coverage_start, time_coverage_end = None, None
        if "time" in dataset:
            time = dataset["time"]
            time_bnds_name = time.attrs.get("bounds", "time_bnds")
            if time_bnds_name in dataset:
                time_bnds = dataset[time_bnds_name]
                if time_bnds.shape == (1, 2):
                    time_coverage_start = str(time_bnds[0][0].data)
                    time_coverage_end = str(time_bnds[0][1].data)
            if time_coverage_start is None or time_coverage_end is None:
                time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
            if time_coverage_start is None or time_coverage_end is None:
                if time.shape == (1,):
                    time_coverage_start = str(time[0].data)
                    time_coverage_end = time_coverage_start
        if time_coverage_start is None or time_coverage_end is None:
            time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
        if time_coverage_start is None or time_coverage_end is None:
            raise ValueError("invalid input: missing time coverage information in dataset")

        return to_time_in_days_since_1970(time_coverage_start), to_time_in_days_since_1970(time_coverage_end)

    @classmethod
    def get_time_range_from_attrs(cls, dataset: xr.Dataset) -> Tuple[str, str]:
        time_start = time_stop = None
        if "time_coverage_start" in dataset.attrs:
            time_start = str(dataset.attrs["time_coverage_start"])
            time_stop = str(dataset.attrs.get("time_coverage_end", time_start))
        elif "time_start" in dataset.attrs:
            time_start = str(dataset.attrs["time_start"])
            time_stop = str(dataset.attrs.get("time_stop", dataset.attrs.get("time_end", time_start)))
        elif "start_time" in dataset.attrs:
            time_start = str(dataset.attrs["start_time"])
            time_stop = str(dataset.attrs.get("stop_time", dataset.attrs.get("end_time", time_start)))
        elif "start_date" in dataset.attrs:
            time_start = str(dataset.attrs["start_date"])
            time_stop = str(dataset.attrs.get("stop_date", dataset.attrs.get("end_time", time_start)))
        return time_start, time_stop

    def _validate(self, dataset):
        self._check_coordinate_var(dataset, "lon", min_length=2)
        self._check_coordinate_var(dataset, "lat", min_length=2)
        if "time" in dataset.dims:
            self._check_coordinate_var(dataset, "time", max_length=1)
            required_dims = ("time", "lat", "lon")
        else:
            required_dims = ("lat", "lon")
        count = 0
        for var_name in dataset.data_vars:
            var = dataset.data_vars[var_name]
            if var.dims == required_dims:
                count += 1
        if count == 0:
            raise ValueError(f"dataset has no variables with required dimensions {required_dims!r}")

    # noinspection PyMethodMayBeStatic
    def _check_coordinate_var(self, dataset: xr.Dataset, coord_var_name: str,
                              min_length: int = None, max_length: int = None):
        if coord_var_name not in dataset.coords:
            raise ValueError(f'missing coordinate variable "{coord_var_name}"')
        coord_var = dataset.coords[coord_var_name]
        if len(coord_var.shape) != 1:
            raise ValueError('coordinate variable "lon" must be 1D')
        coord_var_bnds_name = coord_var.attrs.get("bounds", coord_var_name + "_bnds")
        if coord_var_bnds_name in dataset:
            coord_bnds_var = dataset[coord_var_bnds_name]
            expected_shape = (len(coord_var), 2)
            if coord_bnds_var.shape != expected_shape:
                raise ValueError(f'coordinate bounds variable "{coord_bnds_var}" must have shape {expected_shape!r}')
        else:
            if min_length is not None and len(coord_var) < min_length:
                raise ValueError(f'coordinate variable "{coord_var_name}" must have at least {min_length} value(s)')
            if max_length is not None and len(coord_var) > max_length:
                raise ValueError(f'coordinate variable "{coord_var_name}" must have no more than {max_length} value(s)')


def _normalize_lon_360(dataset: xr.Dataset) -> xr.Dataset:
    """
    Fix the longitude of the given dataset ``dataset`` so that it ranges from -180 to +180 degrees.

    :param dataset: The dataset whose longitudes may be given in the range 0 to 360.
    :return: The fixed dataset or the original dataset.
    """

    if 'lon' not in dataset.coords:
        return dataset

    lon_var = dataset.coords['lon']

    if len(lon_var.shape) != 1:
        return dataset

    lon_size = lon_var.shape[0]
    if lon_size < 2:
        return dataset

    lon_size_05 = lon_size // 2
    lon_values = lon_var.values
    if not np.any(lon_values[lon_size_05:] > 180.):
        return dataset

    # roll_coords will be set to False by default in the future
    dataset = dataset.roll(lon=lon_size_05, roll_coords=True)
    dataset = dataset.assign_coords(lon=(((dataset.lon + 180) % 360) - 180))

    return dataset


def find_input_processor_class(name: str):
    extension = get_extension_registry().get_extension(EXTENSION_POINT_INPUT_PROCESSORS, name)
    if not extension:
        return None
    return extension.component
