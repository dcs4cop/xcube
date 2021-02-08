# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import itertools

import numpy as np
import pandas as pd
import xarray as xr


def new_cube(title='Test Cube',
             width=360,
             height=180,
             x_name='lon',
             y_name='lat',
             x_dtype='float64',
             y_dtype=None,
             x_units='degrees_east',
             y_units='degrees_north',
             x_res=1.0,
             y_res=None,
             x_start=-180.0,
             y_start=-90.0,
             inverse_y=False,
             time_name='time',
             time_dtype='datetime64[s]',
             time_units='seconds since 1970-01-01T00:00:00',
             time_calendar='proleptic_gregorian',
             time_periods=5,
             time_freq="D",
             time_start='2010-01-01T00:00:00',
             drop_bounds=False,
             variables=None):
    """
    Create a new empty cube. Useful for creating cubes templates with predefined
    coordinate variables and metadata. The function is also heavily used
    by xcube's unit tests.

    The values of the *variables* dictionary can be either constants,
    array-like objects, or functions that compute their return value from
    passed coordinate indexes. The expected signature is:::

        def my_func(time: int, y: int, x: int) -> Union[bool, int, float]

    :param title: A title. Defaults to 'Test Cube'.
    :param width: Horizontal number of grid cells. Defaults to 360.
    :param height: Vertical number of grid cells. Defaults to 180.
    :param x_name: Name of the x coordinate variable. Defaults to 'lon'.
    :param y_name: Name of the y coordinate variable. Defaults to 'lat'.
    :param x_dtype: Data type of x coordinates. Defaults to 'float64'.
    :param y_dtype: Data type of y coordinates. Defaults to 'float64'.
    :param x_units: Units of the x coordinates. Defaults to 'degrees_east'.
    :param y_units: Units of the y coordinates. Defaults to 'degrees_north'.
    :param x_start: Minimum x value. Defaults to -180.
    :param y_start: Minimum y value. Defaults to -90.
    :param x_res: Spatial resolution in x-direction. Defaults to 1.0.
    :param y_res: Spatial resolution in y-direction. Defaults to 1.0.
    :param inverse_y: Whether to create an inverse y axis. Defaults to False.
    :param time_name: Name of the time coordinate variable. Defaults to 'time'.
    :param time_periods: Number of time steps. Defaults to 5.
    :param time_freq: Duration of each time step. Defaults to `1D'.
    :param time_start: First time value. Defaults to '2010-01-01T00:00:00'.
    :param time_dtype: Numpy data type for time coordinates. Defaults to 'datetime64[s]'.
    :param time_units: Units for time coordinates. Defaults to 'seconds since 1970-01-01T00:00:00'.
    :param time_calendar: Calender for time coordinates. Defaults to 'proleptic_gregorian'.
    :param drop_bounds: If True, coordinate bounds variables are not created. Defaults to False.
    :param variables: Dictionary of data variables to be added. None by default.
    :return: A cube instance
    """
    y_dtype = y_dtype if y_dtype is not None else y_dtype
    y_res = y_res if y_res is not None else x_res
    if width < 0 or height < 0 or x_res <= 0.0 or y_res <= 0.0:
        raise ValueError()
    if time_periods < 0:
        raise ValueError()

    x_is_lon = x_name == 'lon' or x_units == 'degrees_east'
    y_is_lat = y_name == 'lat' or y_units == 'degrees_north'

    x_end = x_start + width * x_res
    y_end = y_start + height * y_res

    if x_is_lon and x_start < -180. or x_end > 180.:
        raise ValueError()

    if y_is_lat and y_start < -90. or y_end > 90.:
        raise ValueError()

    x_res_05 = 0.5 * x_res
    y_res_05 = 0.5 * y_res

    x_data = np.linspace(x_start + x_res_05, x_end - x_res_05, width, dtype=x_dtype)
    y_data = np.linspace(y_start + y_res_05, y_end - y_res_05, height, dtype=y_dtype)

    x_var = xr.DataArray(x_data, dims=x_name, attrs=dict(units=x_units))
    y_var = xr.DataArray(y_data, dims=y_name, attrs=dict(units=y_units))
    if inverse_y:
        y_var = y_var[::-1]

    if x_is_lon:
        x_var.attrs.update(long_name='longitude',
                           standard_name='longitude')
    else:
        x_var.attrs.update(long_name='x coordinate of projection',
                           standard_name='projection_x_coordinate')
    if y_is_lat:
        y_var.attrs.update(long_name='latitude',
                           standard_name='latitude')
    else:
        y_var.attrs.update(long_name='y coordinate of projection',
                           standard_name='projection_y_coordinate')

    time_data_p1 = pd.date_range(start=time_start, periods=time_periods + 1, freq=time_freq).values
    time_data_p1 = time_data_p1.astype(dtype=time_dtype)

    time_delta = time_data_p1[1] - time_data_p1[0]
    time_data = time_data_p1[0:-1] + time_delta // 2
    time_var = xr.DataArray(time_data, dims=time_name)
    time_var.encoding['units'] = time_units
    time_var.encoding['calendar'] = time_calendar

    coords = {x_name: x_var, y_name: y_var, time_name: time_var}
    if not drop_bounds:
        x_bnds_name = f'{x_name}_bnds'
        y_bnds_name = f'{y_name}_bnds'
        time_bnds_name = f'{time_name}_bnds'

        bnds_dim = 'bnds'

        x_bnds_data = np.zeros((width, 2), dtype=np.float64)
        x_bnds_data[:, 0] = np.linspace(x_start, x_end - x_res, width, dtype=x_dtype)
        x_bnds_data[:, 1] = np.linspace(x_start + x_res, x_end, width, dtype=x_dtype)
        y_bnds_data = np.zeros((height, 2), dtype=np.float64)
        y_bnds_data[:, 0] = np.linspace(y_start, y_end - x_res, height, dtype=y_dtype)
        y_bnds_data[:, 1] = np.linspace(y_start + x_res, y_end, height, dtype=y_dtype)
        if inverse_y:
            y_bnds_data = y_bnds_data[::-1, ::-1]

        x_bnds_var = xr.DataArray(x_bnds_data, dims=(x_name, bnds_dim), attrs=dict(units=x_units))
        y_bnds_var = xr.DataArray(y_bnds_data, dims=(y_name, bnds_dim), attrs=dict(units=y_units))

        x_var.attrs['bounds'] = x_bnds_name
        y_var.attrs['bounds'] = y_bnds_name

        time_bnds_data = np.zeros((time_periods, 2), dtype=time_data_p1.dtype)
        time_bnds_data[:, 0] = time_data_p1[:-1]
        time_bnds_data[:, 1] = time_data_p1[1:]
        time_bnds_var = xr.DataArray(time_bnds_data, dims=(time_name, bnds_dim))
        time_bnds_var.encoding['units'] = time_units
        time_bnds_var.encoding['calendar'] = time_calendar

        time_var.attrs['bounds'] = time_bnds_name

        coords.update({x_bnds_name: x_bnds_var, y_bnds_name: y_bnds_var, time_bnds_name: time_bnds_var})

    attrs = dict(Conventions="CF-1.7",
                 title=title,
                 time_coverage_start=str(time_data_p1[0]),
                 time_coverage_end=str(time_data_p1[-1]))

    if x_is_lon:
        attrs.update(dict(geospatial_lon_min=x_start,
                          geospatial_lon_max=x_end,
                          geospatial_lon_units=x_units))

    if y_is_lat:
        attrs.update(dict(geospatial_lat_min=y_start,
                          geospatial_lat_max=y_end,
                          geospatial_lat_units=y_units))

    data_vars = {}
    if variables:
        dims = (time_name, y_name, x_name)
        shape = (time_periods, height, width)
        size = time_periods * height * width
        for var_name, data in variables.items():
            if isinstance(data, xr.DataArray):
                data_vars[var_name] = data
            elif isinstance(data, int) or isinstance(data, float) or isinstance(data, bool):
                data_vars[var_name] = xr.DataArray(np.full(shape, data), dims=dims)
            elif callable(data):
                func = data
                data = np.zeros(shape)
                for index in itertools.product(*map(range, shape)):
                    data[index] = func(*index)
                data_vars[var_name] = xr.DataArray(data, dims=dims)
            elif data is None:
                data_vars[var_name] = xr.DataArray(np.random.uniform(0.0, 1.0, size).reshape(shape), dims=dims)
            else:
                data_vars[var_name] = xr.DataArray(data, dims=dims)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
