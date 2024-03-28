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

import pathlib
from typing import Type, Iterator

import fsspec
from fsspec.implementations.local import LocalFileSystem


def is_local_fs(fs: fsspec.AbstractFileSystem) -> bool:
    """Check whether *fs* is a local filesystem."""
    return "file" in fs.protocol or isinstance(fs, LocalFileSystem)


def get_fs_path_class(fs: fsspec.AbstractFileSystem) -> Type[pathlib.PurePath]:
    """Get the appropriate `pathlib.PurePath` class for the filesystem *fs*."""
    if is_local_fs(fs):
        # Will return PurePosixPath or a PureWindowsPath object
        return pathlib.PurePath
    # PurePosixPath for non-local filesystems such as S3,
    # so we force use of forward slashes as separators
    return pathlib.PurePosixPath


def resolve_path(path: pathlib.PurePath) -> pathlib.PurePath:
    """Resolve "." and ".." occurrences in *path* without I/O and
    return a new path.
    """
    reversed_parts = reversed(path.parts)
    reversed_norm_parts = list(_resolve_path_impl(reversed_parts))
    parts = reversed(reversed_norm_parts)
    return type(path)(*parts)


def _resolve_path_impl(reversed_parts: Iterator[str]):
    skips = False
    for part in reversed_parts:
        if part == ".":
            continue
        elif part == "..":
            skips += 1
            continue
        if skips == 0:
            yield part
        elif skips > 0:
            skips -= 1
    if skips != 0:
        raise ValueError('cannot resolve path, misplaced ".."')
