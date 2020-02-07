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

from typing import Dict, Any, Sequence, Union

import xarray as xr
import numpy as np

from xcube.core.schema import CubeSchema
from xcube.core.select import select_vars
from xcube.core.verify import assert_cube


def resample_in_time(cube: xr.Dataset,
                     frequency: str,
                     method: Union[str, Sequence[str]],
                     offset=None,
                     base: int = 0,
                     tolerance=None,
                     interp_kind=None,
                     time_chunk_size=None,
                     var_names: Sequence[str] = None,
                     metadata: Dict[str, Any] = None,
                     cube_asserted: bool = False) -> xr.Dataset:
    """
    Resample a xcube dataset in the time dimension.

    :param cube: The xcube dataset.
    :param frequency: Temporal aggregation frequency. Use format "<count><offset>"
        "where <offset> is one of 'H', 'D', 'W', 'M', 'Q', 'Y'.
    :param method: Resampling method or sequence of resampling methods.
    :param offset: Offset used to adjust the resampled time labels.
        Uses same syntax as *frequency*.
    :param base: For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '24H' frequency, base could range from 0 through 23.
    :param time_chunk_size: If not None, the chunk size to be used for the "time" dimension.
    :param var_names: Variable names to include.
    :param tolerance: Time tolerance for selective upsampling methods. Defaults to *frequency*.
    :param interp_kind: Kind of interpolation if *method* is 'interpolation'.
    :param metadata: Output metadata.
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: A new xcube dataset resampled in time.
    """
    if not cube_asserted:
        assert_cube(cube)

    if var_names:
        cube = select_vars(cube, var_names)
        
    if frequency == 'all':
        time_gap = np.array(cube.time[-1]) - np.array(cube.time[0])
        days = int((np.timedelta64(time_gap, 'D')/np.timedelta64(1, 'D'))+1)
        frequency = ''.join([str(days), 'D'])

    resampler = cube.resample(skipna=True,
                              closed='left',
                              label='left',
                              keep_attrs=True,
                              time=frequency,
                              loffset=offset,
                              base=base)

    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    resampled_cubes = []
    for method in methods:
        resampling_method = getattr(resampler, method)
        kwargs = get_method_kwargs(method, frequency, interp_kind, tolerance)
        resampled_cube = resampling_method(**kwargs)
        resampled_cube = resampled_cube.rename(
            {var_name: f'{var_name}_{method}' for var_name in resampled_cube.data_vars})
        resampled_cubes.append(resampled_cube)

    if len(resampled_cubes) == 1:
        resampled_cube = resampled_cubes[0]
    else:
        resampled_cube = xr.merge(resampled_cubes)

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = '%s' % cube.time[0]
    time_coverage_end = '%s' % cube.time[-1]

    resampled_cube.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    resampled_cube.attrs.update(time_coverage_start=time_coverage_start,
                                time_coverage_end=time_coverage_end)

    schema = CubeSchema.new(cube)
    chunk_sizes = {schema.dims[i]: schema.chunks[i] for i in range(schema.ndim)}

    if isinstance(time_chunk_size, int) and time_chunk_size >= 0:
        chunk_sizes['time'] = time_chunk_size

    return resampled_cube.chunk(chunk_sizes)


def get_method_kwargs(method, frequency, interp_kind, tolerance):
    if method == 'interpolate':
        kwargs = {'kind': interp_kind or 'linear'}
    elif method in {'nearest', 'bfill', 'ffill', 'pad'}:
        kwargs = {'tolerance': tolerance or frequency}
    elif method in {'first', 'last', 'sum', 'min', 'max', 'mean', 'median', 'std', 'var'}:
        kwargs = {'dim': 'time', 'keep_attrs': True, 'skipna': True}
    elif method == 'prod':
        kwargs = {'dim': 'time', 'skipna': True}
    elif method == 'count':
        kwargs = {'dim': 'time', 'keep_attrs': True}
    else:
        kwargs = {}
    return kwargs
