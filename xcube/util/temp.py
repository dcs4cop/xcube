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

import atexit
import os
import shutil
import tempfile
from typing import Tuple

DEFAULT_TEMP_FILE_PREFIX = "xcube-"


def new_temp_dir(prefix: str = None, suffix: str = None, dir_path: str = None) -> str:
    """Create new temporary directory using `tempfile.mkdtemp()`.
    The directory will be removed later when the process ends.

    Args:
        prefix: If not `None`, the file name will
            begin with that prefix; otherwise, a default prefix is used.
        suffix: If not `None`, the file name will
            end with that suffix, otherwise there will be no suffix.
        dir_path: If not `None`, the file will be created
            in that directory; otherwise, a default directory is used.
            The default directory is chosen from a platform-dependent
            list, but the user of the application can control the
            directory location by setting the *TMPDIR*, *TEMP* or *TMP*
            environment variables.
    Returns:
         The absolute path to the new directory.
    """
    prefix = DEFAULT_TEMP_FILE_PREFIX if prefix is None else prefix
    return remove_dir_later(
        tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir_path)
    )


def new_temp_file(
    prefix: str = None,
    suffix: str = None,
    dir_path: str = None,
    text_mode: bool = False,
) -> Tuple[int, str]:
    """Create new temporary file using `tempfile.mkstemp()```.
    The file will be removed later when the process ends.

    Args:
        prefix: If not `None`, the file name will begin with that
            prefix; otherwise, a default prefix is used.
        suffix: If not `None`, the file name will end with that
            suffix, otherwise there will be no suffix.
        dir_path: If not `None`, the file will be created in that
            directory; otherwise, a default directory is used. The
            default directory is chosen from a platform-dependent list,
            but the user of the application can control the directory
            location by setting the *TMPDIR*, *TEMP* or *TMP*
            environment variables.
        text_mode: If specified and true, the file is opened in text
            mode. Otherwise, (the default) the file is opened in binary
            mode.

    Returns:
        A tuple comprising the file descriptor (an int) and the absolute
        path to the new file.
    """
    prefix = DEFAULT_TEMP_FILE_PREFIX if prefix is None else prefix
    fd, file_path = tempfile.mkstemp(
        prefix=prefix, suffix=suffix, dir=dir_path, text=text_mode
    )
    return fd, remove_file_later(file_path)


def remove_dir_later(dir_path: str) -> str:
    """Remove directory later when the process ends.
    Removal failures are ignored.

    Args:
        dir_path: Path to directory

    Returns:
        The absolute path to the directory.
    """
    dir_path = os.path.abspath(dir_path)

    def _remove_dir_later():
        # noinspection PyBroadException
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
        except BaseException:
            pass

    atexit.register(_remove_dir_later)
    return dir_path


def remove_file_later(file_path: str) -> str:
    """Remove file later when the process ends.
    Removal failures are ignored.

    Args:
        file_path: path to a file

    Returns:
        The absolute path to the file.
    """
    file_path = os.path.abspath(file_path)

    def _remove_file_later():
        # noinspection PyBroadException
        try:
            os.remove(file_path)
        except BaseException:
            pass

    atexit.register(_remove_file_later)
    return file_path
