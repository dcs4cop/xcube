# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
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

import abc
from typing import Union, Callable, Optional, Any, Awaitable, TypeVar

from tornado import concurrent

ReturnT = TypeVar("ReturnT")


class AsyncExecution(abc.ABC):
    """
    An interface that defines asynchronous function execution.
    """

    @abc.abstractmethod
    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        """Executes the given callable *callback* after *delay*
        seconds.

        The method returns a handle that can be used to cancel the
        callback.

        Args:
            delay: Delay in seconds.
            callback: Callback to be called.
            args: Positional arguments passed to *callback*.
            kwargs: Keyword arguments passed to *callback*.
        Returns:
             A handle that provides the methods
            `cancel()` and `cancelled()`.
        """

    @abc.abstractmethod
    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any
    ) -> Awaitable[ReturnT]:
        """
        Concurrently runs a *function* in a `concurrent.futures.Executor`.
        If *executor* is `None`, the framework's default
        executor will be used.

        Args:
            executor: An optional executor.
            function: The function to be run concurrently.
            args: Positional arguments passed to *function*.
            kwargs: Keyword arguments passed to *function*.
        Returns:
            The awaitable return value of *function*.
        """
