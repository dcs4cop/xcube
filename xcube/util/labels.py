# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger('xcube')


def ensure_time_compatible(var: xr.DataArray,
                           labels: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure that labels['time'] is timezone-naive, if necessary.

    This function returns either the passed-in labels object, or a copy of
    it with a modified value for labels['time'].

    If there is no 'time' key in the labels dictionary or if there is no
    'time' dimension in the var array, the original labels are returned.

    If there is a 'time' key, it is expected that its value will be
    a valid timestamp (i.e. a valid input to pandas.Timestamp.__init__), or
    a slice in which the start and stop fields are valid timestamps. For a
    slice, the start and stop fields are processed separately, and their
    modified values (if required) are returned as the start and stop fields
    of a new slice. The step field is included unchanged in the new slice.

    If var has a 'time' dimension of type datetime64 and labels has a 'time'
    key with a timezone-aware value, return a modified labels dictionary with
    a timezone-naive time value. Otherwise return the original labels.

    """

    if 'time' not in labels or 'time' not in var.dims:
        return labels

    timeval = labels['time']
    if isinstance(timeval, slice):
        # process start and stop separately, and pass step through unchanged
        return dict(labels, time=slice(
            _ensure_timestamp_compatible(var, timeval.start),
            _ensure_timestamp_compatible(var, timeval.stop),
            timeval.step))
    else:
        return dict(labels, time=_ensure_timestamp_compatible(var, timeval))


def _ensure_timestamp_compatible(var: xr.DataArray, timeval: Any):
    timestamp = pd.Timestamp(timeval)
    timeval_timezone = timestamp.tzinfo
    array_timezone = _get_array_timezone(var)

    if array_timezone is None and timeval_timezone is not None:
        return timestamp.tz_convert(None)
    elif array_timezone is not None and timeval_timezone is None:
        return timestamp.tz_localize(array_timezone)
    else:
        return timeval


def _get_array_timezone(var: xr.DataArray):
    # TODO: also check for non-datetime64 tz-naive arrays!

    # pandas treats all datetime64 arrays as timezone-naive
    if _has_datetime64_time(var):
        return None

    if isinstance(var.time.values[0], pd.Timestamp):
        return var.time.values[0].tzinfo

    logger.warning("Can't determine array timezone, assuming TZ-naive")
    return None


def _has_datetime64_time(var: xr.DataArray) -> bool:
    """Report whether var's time dimension has type datetime64

    Assumes that a 'time' key is present in var.dims."""
    return hasattr(var['time'], 'dtype') \
        and hasattr(var['time'].dtype, 'type') \
        and var['time'].dtype.type is np.datetime64
