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

import functools
import logging
import time
import warnings
from contextlib import AbstractContextManager
from typing import Optional

from xcube.constants import LOG


def measure_time_cm(logger=None, disabled=False):
    """Get a context manager for measuring execution time of code blocks and logging the result.

    Measure duration and log output:

    ```python
        measure_time = measure_time_cm()
        with measure_time("heavy computation"):
            do_heavy_computation()
    ```

    or just measure duration:

    ```python
        with measure_time() as cm:
            do_heavy_computation()
        print("heavy computation took %2.f seconds" % cm.duration)
    ```

    Args:
        logger: The logger to be used. May be a string or logger object.
            Defaults to "xcube".
        disabled: If True, efficiently disables timing and logging.

    Returns:
        a context manager callable
    """
    if disabled:
        return _do_not_measure_time_cm
    else:
        return functools.partial(measure_time, logger=logger)


class measure_time(AbstractContextManager):
    def __init__(
        self,
        *args,
        tag: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.message = tag
        self.args = args
        self.kwargs = kwargs
        if tag is not None:
            warnings.warn(
                'The keyword "tag" has been deprecated,'
                ' use first argument "message" instead',
                DeprecationWarning,
            )
        elif args:
            self.message = args[0]
            self.args = args[1:]
        else:
            warnings.warn(
                'Calling measure_time() without "message"' " argument is deprecated.",
                DeprecationWarning,
            )
            self.message = None
            self.args = None
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif logger is None:
            self.logger = LOG
        else:
            self.logger = logger
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration = time.perf_counter() - self.start_time
        if self.message:
            self.logger.info(
                self.message + ": took %.2fms",
                *self.args,
                self.duration * 1000,
                **self.kwargs
            )


class _do_not_measure_time_cm(AbstractContextManager):

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):
        self.duration = None
        self.logger = None
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
