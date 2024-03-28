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

from abc import ABC, abstractmethod
from typing import Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.progress import observe_progress
from .helpers import is_empty_cube
from .helpers import strip_cube
from ..config import CubeConfig

TransformedCube = Tuple[xr.Dataset, GridMapping, CubeConfig]


class CubeTransformer(ABC):
    @abstractmethod
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        """
        Transform given *cube*, grid mapping *gm*, and cube
        configuration *cube_config* into a potentially new cube,
        a new grid mapping, and less restrictive cube configuration
        *cube_config*.

        The latter is achieved by returning a derived *cube_config* where
        all properties that have been "consumed" by this transformer
        are removed. See `CubeConfig.drop_property`.
        """


class CubeIdentity(CubeTransformer):
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        """Return *cube*, grid mapping *gm*, and parameters without change."""
        return cube, gm, cube_config


def transform_cube(
    t_cube: TransformedCube, transformer: CubeTransformer, label: str = ""
) -> TransformedCube:
    empty_cube = is_empty_cube(t_cube[0])
    identity = isinstance(transformer, CubeIdentity)
    if not label:
        label = f"{type(transformer).__name__}"
    if identity:
        label += " (step not applicable)"
    elif empty_cube:
        label += " (step not applicable, empty cube)"

    with observe_progress(label, 1) as progress:
        if not (identity or empty_cube):
            t_cube = transformer.transform_cube(*t_cube)
            t_cube = strip_cube(t_cube[0]), t_cube[1], t_cube[2]
        progress.worked(1)

    return t_cube
