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

import cProfile
import glob
import io
import os
import pstats
import time
import traceback
import warnings
from typing import Any, Callable, Dict, Sequence, Tuple

import xarray as xr

from xcube.core.dsio import DatasetIO, find_dataset_io, guess_dataset_format, rimraf
from xcube.core.evaluate import evaluate_dataset
from xcube.core.gen.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_RESAMPLING, DEFAULT_OUTPUT_SIZE
from xcube.core.gen.iproc import InputProcessor, find_input_processor_class
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping import CRS_WGS84
from xcube.core.optimize import optimize_dataset
from xcube.core.select import select_spatial_subset, select_variables_subset
from xcube.core.timecoord import add_time_coords, from_time_in_days_since_1970
from xcube.core.timeslice import find_time_slice
from xcube.core.update import update_dataset_attrs, update_dataset_temporal_attrs, update_dataset_var_attrs
from xcube.util.config import NameAnyDict, NameDictPairList, to_resolved_name_dict_pairs


def gen_cube(input_paths: Sequence[str] = None,
             input_processor_name: str = None,
             input_processor_params: Dict = None,
             input_reader_name: str = None,
             input_reader_params: Dict[str, Any] = None,
             output_region: Tuple[float, float, float, float] = None,
             output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
             output_resampling: str = DEFAULT_OUTPUT_RESAMPLING,
             output_path: str = DEFAULT_OUTPUT_PATH,
             output_writer_name: str = None,
             output_writer_params: Dict[str, Any] = None,
             output_metadata: NameAnyDict = None,
             output_variables: NameDictPairList = None,
             processed_variables: NameDictPairList = None,
             profile_mode: bool = False,
             no_sort_mode: bool = False,
             append_mode: bool = None,
             dry_run: bool = False,
             monitor: Callable[..., None] = None) -> bool:
    """
    Generate a xcube dataset from one or more input files.

    :param no_sort_mode:
    :param input_paths: The input paths.
    :param input_processor_name: Name of a registered input processor
        (xcube.core.gen.inputprocessor.InputProcessor) to be used to transform the inputs.
    :param input_processor_params: Parameters to be passed to the input processor.
    :param input_reader_name: Name of a registered input reader (xcube.core.util.dsio.DatasetIO).
    :param input_reader_params: Parameters passed to the input reader.
    :param output_region: Output region as tuple of floats: (lon_min, lat_min, lon_max, lat_max).
    :param output_size: The spatial dimensions of the output as tuple of ints: (width, height).
    :param output_resampling: The resampling method for the output.
    :param output_path: The output directory.
    :param output_writer_name: Name of an output writer
        (xcube.core.util.dsio.DatasetIO) used to write the cube.
    :param output_writer_params: Parameters passed to the output writer.
    :param output_metadata: Extra metadata passed to output cube.
    :param output_variables: Output variables.
    :param processed_variables: Processed variables computed on-the-fly.
    :param profile_mode: Whether profiling should be enabled.
    :param append_mode: Deprecated. The function will always either insert, replace, or append new time slices.
    :param dry_run: Doesn't write any data. For testing.
    :param monitor: A progress monitor.
    :return: True for success.
    """

    if append_mode is not None:
        warnings.warn('append_mode in gen_cube() is deprecated, '
                      'time slices will now always be inserted, replaced, or appended.')

    if input_processor_name is None:
        input_processor_name = 'default'
    elif input_processor_name == '':
        raise ValueError('input_processor_name must not be empty')

    input_processor_class = find_input_processor_class(input_processor_name)
    if not input_processor_class:
        raise ValueError(f'Unknown input_processor_name {input_processor_name!r}')

    if not issubclass(input_processor_class, InputProcessor):
        raise ValueError(f'Invalid input_processor_name {input_processor_name!r}: '
                         f'must name a sub-class of {InputProcessor.__qualname__}')

    try:
        input_processor = input_processor_class(**(input_processor_params or {}))
    except (ValueError, TypeError) as e:
        raise ValueError(f'Invalid input_processor_name or input_processor_params: {e}') from e

    input_reader = find_dataset_io(input_reader_name or input_processor.input_reader)
    if not input_reader:
        raise ValueError(f'Unknown input_reader_name {input_reader_name!r}')

    if not output_path:
        raise ValueError('Missing output_path')

    output_writer_name = output_writer_name or guess_dataset_format(output_path)
    if not output_writer_name:
        raise ValueError(f'Failed to guess output_writer_name from path {output_path}')
    output_writer = find_dataset_io(output_writer_name, modes={'w', 'a'})
    if not output_writer:
        raise ValueError(f'Unknown output_writer_name {output_writer_name!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    input_paths = [input_file for f in input_paths for input_file in glob.glob(f, recursive=True)]

    if not no_sort_mode and len(input_paths) > 1:
        input_paths = _get_sorted_input_paths(input_processor, input_paths)

    if not dry_run:
        output_dir = os.path.abspath(os.path.dirname(output_path))
        os.makedirs(output_dir, exist_ok=True)

    effective_input_reader_params = dict(input_processor.input_reader_params or {})
    effective_input_reader_params.update(input_reader_params or {})

    effective_output_writer_params = output_writer_params or {}

    status = False

    ds_count = len(input_paths)
    ds_count_ok = 0
    ds_index = 0
    for input_file in input_paths:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        # noinspection PyTypeChecker
        status = _process_input(input_processor,
                                input_reader,
                                effective_input_reader_params,
                                output_writer,
                                effective_output_writer_params,
                                input_file,
                                output_size,
                                output_region,
                                output_resampling,
                                output_path,
                                output_metadata,
                                output_variables,
                                processed_variables,
                                profile_mode,
                                dry_run,
                                monitor)
        ds_index += 1
        if status:
            ds_count_ok += 1

    monitor(f'{ds_count_ok} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_count_ok} were dropped due to errors')

    return status


def _process_input(input_processor: InputProcessor,
                   input_reader: DatasetIO,
                   input_reader_params: Dict[str, Any],
                   output_writer: DatasetIO,
                   output_writer_params: Dict[str, Any],
                   input_file: str,
                   output_size: Tuple[int, int],
                   output_region: Tuple[float, float, float, float],
                   output_resampling: str,
                   output_path: str,
                   output_metadata: NameAnyDict = None,
                   output_variables: NameDictPairList = None,
                   processed_variables: NameDictPairList = None,
                   profile_mode: bool = False,
                   dry_run: bool = False,
                   monitor: Callable[..., None] = None) -> bool:
    monitor('reading input slice...')
    # noinspection PyBroadException
    try:
        input_dataset = input_reader.read(input_file, **input_reader_params)
        monitor(f'Dataset read:\n{input_dataset}')
    except Exception as e:
        monitor(f'Error: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return False

    time_range = input_processor.get_time_range(input_dataset)
    if time_range[0] > time_range[1]:
        monitor('Error: start time is greater than end time: skipping...')
        return False

    if output_variables:
        output_variables = to_resolved_name_dict_pairs(output_variables, input_dataset, keep=True)
    else:
        output_variables = [(var_name, None) for var_name in input_dataset.data_vars]

    time_index, update_mode = find_time_slice(output_path,
                                              from_time_in_days_since_1970((time_range[0] + time_range[1]) / 2))

    width, height = output_size
    x_min, y_min, x_max, y_max = output_region
    xy_res = max((x_max - x_min) / width, (y_max - y_min) / height)
    tile_size = _get_tile_size(output_writer_params)

    output_geom = GridMapping.regular(size=output_size,
                                      xy_min=(x_min, y_min),
                                      xy_res=xy_res,
                                      crs=CRS_WGS84,
                                      tile_size=tile_size)

    steps = []

    # noinspection PyShadowingNames
    def step1(input_slice):
        return input_processor.pre_process(input_slice)

    steps.append((step1, 'pre-processing input slice'))

    geo_coding = None

    # noinspection PyShadowingNames
    def step1a(input_slice):
        nonlocal geo_coding
        geo_coding = GridMapping.from_dataset(input_slice)
        subset = select_spatial_subset(input_slice,
                                       xy_bbox=output_geom.xy_bbox,
                                       xy_border=output_geom.x_res,
                                       ij_border=1,
                                       geo_coding=geo_coding)
        if subset is None:
            monitor('no spatial overlap with input')
        elif subset is not input_slice:
            geo_coding = GridMapping.from_dataset(subset)
        return subset

    steps.append((step1a, 'spatial subsetting'))

    # noinspection PyShadowingNames
    def step2(input_slice):
        return evaluate_dataset(input_slice, processed_variables=processed_variables)

    steps.append((step2, 'computing input slice variables'))

    # noinspection PyShadowingNames
    def step3(input_slice):
        extra_vars = input_processor.get_extra_vars(input_slice)
        selected_variables = set([var_name for var_name, _ in output_variables])
        selected_variables.update(extra_vars or set())
        return select_variables_subset(input_slice, selected_variables)

    steps.append((step3, 'selecting input slice variables'))

    # noinspection PyShadowingNames
    def step4(input_slice):
        # noinspection PyTypeChecker
        return input_processor.process(input_slice,
                                       geo_coding=geo_coding,
                                       output_geom=output_geom,
                                       output_resampling=output_resampling,
                                       include_non_spatial_vars=False)

    steps.append((step4, 'transforming input slice'))

    if time_range is not None:
        def step5(input_slice):
            return add_time_coords(input_slice, time_range)

        steps.append((step5, 'adding time coordinates to input slice'))

    def step6(input_slice):
        return update_dataset_var_attrs(input_slice, output_variables)

    steps.append((step6, 'updating variable attributes of input slice'))

    def step7(input_slice):
        return input_processor.post_process(input_slice)

    steps.append((step7, 'post-processing input slice'))

    if update_mode == 'create':
        def step8(input_slice):
            if not dry_run:
                rimraf(output_path)
                output_writer.write(input_slice, output_path, **output_writer_params)
                _update_cube(output_writer, output_path, global_attrs=output_metadata, temporal_only=False)
            return input_slice

        steps.append((step8, f'creating input slice in {output_path}'))

    elif update_mode == 'append':
        def step8(input_slice):
            if not dry_run:
                output_writer.append(input_slice, output_path, **output_writer_params)
                _update_cube(output_writer, output_path, temporal_only=True, consolidate=True)
            return input_slice

        steps.append((step8, f'appending input slice to {output_path}'))

    elif update_mode == 'insert':
        def step8(input_slice):
            if not dry_run:
                output_writer.insert(input_slice, time_index, output_path)
                _update_cube(output_writer, output_path, temporal_only=True, consolidate=True)
            return input_slice

        steps.append((step8, f'inserting input slice before index {time_index} in {output_path}'))

    elif update_mode == 'replace':
        def step8(input_slice):
            if not dry_run:
                output_writer.replace(input_slice, time_index, output_path)
                _update_cube(output_writer, output_path, temporal_only=True, consolidate=True)
            return input_slice

        steps.append((step8, f'replacing input slice at index {time_index} in {output_path}'))

    if profile_mode:
        pr = cProfile.Profile()
        pr.enable()

    status = True
    try:
        num_steps = len(steps)
        dataset = input_dataset
        total_t1 = time.perf_counter()
        for step_index in range(num_steps):
            transform, label = steps[step_index]
            step_t1 = time.perf_counter()
            monitor(f'step {step_index + 1} of {num_steps}: {label}...')
            dataset = transform(dataset)
            step_t2 = time.perf_counter()
            if dataset is None:
                monitor(f'  {label} terminated after {step_t2 - step_t1} seconds, skipping input slice')
                status = False
                break
            monitor(f'  {label} completed in {step_t2 - step_t1} seconds')
        total_t2 = time.perf_counter()
        monitor(f'{num_steps} steps took {total_t2 - total_t1} seconds to complete')
    except RuntimeError as e:
        monitor(f'Error: something went wrong during processing, skipping input slice: {e}')
        traceback.print_exc()
        status = False
    finally:
        input_dataset.close()

    if profile_mode:
        # noinspection PyUnboundLocalVariable
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        monitor(s.getvalue())

    return status


def _get_tile_size(output_writer_params):
    tile_size = None
    if 'chunksizes' in output_writer_params:
        chunksizes = output_writer_params['chunksizes']
        if 'lon' in chunksizes or 'lat' in chunksizes:
            tile_size = chunksizes.get('lon', 512), chunksizes.get('lat', 512)
        elif 'x' in chunksizes or 'y' in chunksizes:
            tile_size = chunksizes.get('x', 512), chunksizes.get('y', 512)
    return tile_size


def _update_cube(output_writer: DatasetIO, output_path: str, global_attrs: Dict = None, temporal_only: bool = False,
                 consolidate: bool = True):
    if consolidate and os.path.isfile(os.path.join(output_path, '.zgroup')):
        optimize_dataset(input_path=output_path, in_place=True, unchunk_coords=True)
    cube = output_writer.read(output_path)
    if temporal_only:
        cube = update_dataset_temporal_attrs(cube, update_existing=True, in_place=True)
    else:
        cube = update_dataset_attrs(cube, update_existing=True, in_place=True)
    cube_attrs = dict(cube.attrs)
    cube.close()

    if global_attrs:
        cube_attrs.update(global_attrs)

    output_writer.update(output_path, global_attrs=cube_attrs)


def _get_sorted_input_paths(input_processor, input_paths: Sequence[str]):
    input_path_list = []
    time_list = []
    for input_file in input_paths:
        with xr.open_dataset(input_file) as dataset:
            t1, t2 = input_processor.get_time_range(dataset)
            time_list.append((t1 + t2) / 2)
            input_path_list.append(input_file)
            tuple_seq = zip(time_list, input_path_list)
    input_paths = [e[1] for e in sorted(tuple_seq, key=lambda e: e[0])]
    return input_paths
