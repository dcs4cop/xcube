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
from typing import Sequence, Optional, Callable

from xcube.cli._gen2.request import CubeConfig
from xcube.cli._gen2.request import InputConfig
from xcube.core.store.store import new_data_store
from xcube.util.extension import ExtensionRegistry


def open_cubes(input_configs: Sequence[InputConfig],
               cube_config: CubeConfig,
               progress_monitor: Callable,
               extension_registry: Optional[ExtensionRegistry] = None):
    cubes = []
    for input_config in input_configs:
        cube_store = new_data_store(input_config.cube_store_id,
                                    extension_registry=extension_registry,
                                    **input_config.cube_store_params)
        cube_id = input_config.cube_id
        open_params_schema = cube_store.get_open_data_params_schema(cube_id)
        open_params = open_params_schema.from_json(input_config.open_params) \
            if input_config.open_params else {}
        cube = cube_store.open_data(cube_id, **open_params, **cube_config)
        cubes.append(cube)

    return cubes
