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

import warnings
from typing import Optional, Dict, Any, Hashable, Union, Set, List

import pyproj
import xarray as xr


class GridCoords:
    """
    Grid coordinates comprising x and y of
    type xarray.DataArray.
    """

    def __init__(self,
                 x: xr.DataArray = None,
                 y: xr.DataArray = None):
        self.x = x
        self.y = y


class GridMapping:
    """
    Grid mapping comprising *crs* of type pyproj.crs.CRS,
    grid coordinates and an optional name.
    """

    def __init__(self,
                 crs: pyproj.crs.CRS,
                 name: str = None,
                 coords: GridCoords = None):
        self.crs = crs
        self.name = name
        self.coords = coords


def get_dataset_grid_mappings(dataset: xr.Dataset, *,
                              missing_latitude_longitude_crs: pyproj.crs.CRS = None,
                              missing_rotated_latitude_longitude_crs: pyproj.crs.CRS = None,
                              missing_projected_crs: pyproj.crs.CRS = None,
                              emit_warnings: bool = False) -> Dict[Union[Hashable, None], GridMapping]:
    """
    Find grid mappings encoded as described in the CF conventions
    [Horizontal Coordinate Reference Systems, Grid Mappings, and Projections]
    (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections).

    :param dataset:
    :param missing_latitude_longitude_crs:
    :param missing_rotated_latitude_longitude_crs:
    :param missing_projected_crs:
    :param emit_warnings:
    :return:
    """
    # Find any grid mapping variables
    #
    grid_mappings = dict()
    for var_name, var in dataset.variables.items():
        grid_mapping = _parse_crs_from_attrs(var.attrs)
        if grid_mapping is not None:
            grid_mappings[var_name] = grid_mapping

    # If no grid mapping variables found,
    # try if CRS is encoded in dataset attributes
    #
    if not grid_mappings:
        grid_mapping = _parse_crs_from_attrs(dataset.attrs)
        if grid_mapping is not None:
            grid_mappings[None] = grid_mapping

    # Find coordinate variables.
    #

    latitude_longitude_coords = GridCoords(x=None, y=None)
    rotated_latitude_longitude_coords = GridCoords(x=None, y=None)
    projected_coords = GridCoords(x=None, y=None)

    potential_coord_vars = _find_potential_coord_vars(dataset)

    # Find coordinate variables that use a CF standard_name.
    #
    coords_standard_names = (
        (latitude_longitude_coords,
         'longitude', 'latitude'),
        (rotated_latitude_longitude_coords,
         'grid_longitude', 'grid_latitude'),
        (projected_coords,
         'projection_x_coordinate', 'projection_y_coordinate')
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        standard_name = var.attrs.get('standard_name')
        for coords, x_name, y_name in coords_standard_names:
            if standard_name == x_name:
                coords.x = var
            if standard_name == y_name:
                coords.y = var

    # Find coordinate variables by common naming convention.
    #
    coords_var_names = (
        (latitude_longitude_coords,
         {'lon', 'longitude'}, {'lat', 'latitude'}),
        (rotated_latitude_longitude_coords,
         {'rlon', 'rlongitude'}, {'rlat', 'rlatitude'}),
        (projected_coords,
         {'x', 'xc', 'transformed_x'}, {'y', 'yc', 'transformed_y'})
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        for coords, x_names, y_names in coords_var_names:
            if var_name in x_names:
                coords.x = var
            if var_name in y_names:
                coords.y = var

    # Assign found coordinates to grid mappings
    #
    for grid_mapping in grid_mappings.values():
        if grid_mapping.name == 'latitude_longitude':
            grid_mapping.coords = latitude_longitude_coords
        elif grid_mapping.name == 'rotated_latitude_longitude':
            grid_mapping.coords = rotated_latitude_longitude_coords
        else:
            grid_mapping.coords = projected_coords

    _complement_grid_mapping_coords(latitude_longitude_coords,
                                    'latitude_longitude',
                                    missing_latitude_longitude_crs
                                    or pyproj.crs.CRS(4326),
                                    grid_mappings)
    _complement_grid_mapping_coords(rotated_latitude_longitude_coords,
                                    'rotated_latitude_longitude',
                                    missing_rotated_latitude_longitude_crs,
                                    grid_mappings)
    _complement_grid_mapping_coords(projected_coords,
                                    None,
                                    missing_projected_crs,
                                    grid_mappings)

    # Collect complete grid mappings
    complete_grid_mappings = dict()
    for var_name, grid_mapping in grid_mappings.items():
        if grid_mapping.coords is not None \
                and grid_mapping.coords.x is not None \
                and grid_mapping.coords.y is not None:
            # TODO: add more consistency checks, eg both 1d or both 2d with same dims
            complete_grid_mappings[var_name] = grid_mapping
        elif emit_warnings:
            warnings.warn(f'CRS "{grid_mapping.name}": '
                          f'missing x- and/or y-coordinates '
                          f'(grid mapping variable "{var_name}": '
                          f'grid_mapping_name="{grid_mapping.name}")')

    return complete_grid_mappings


def _parse_crs_from_attrs(attrs: Dict[Hashable, Any]) -> Optional[GridMapping]:
    # noinspection PyBroadException
    try:
        crs = pyproj.crs.CRS.from_cf(attrs)
    except pyproj.crs.CRSError:
        return None
    return GridMapping(crs=crs, name=attrs.get('grid_mapping_name'), coords=None)


def _complement_grid_mapping_coords(coords: GridCoords,
                                    grid_mapping_name: Optional[str],
                                    missing_crs: Optional[pyproj.crs.CRS],
                                    grid_mappings: Dict[Optional[str], GridMapping]):
    if coords.x is not None or coords.y is not None:
        grid_mapping = next((grid_mapping
                             for grid_mapping in grid_mappings.values()
                             if grid_mapping.name == grid_mapping_name), None)
        if grid_mapping is None:
            grid_mapping = grid_mappings.get(None)
            if grid_mapping is None and missing_crs is not None:
                grid_mapping = GridMapping(crs=missing_crs,
                                           name=grid_mapping_name,
                                           coords=None)
                grid_mappings[None] = grid_mapping

        if grid_mapping is not None and grid_mapping.coords is None:
            grid_mapping.coords = coords


def _find_potential_coord_vars(dataset: xr.Dataset) -> List[Hashable]:
    """
    Find potential coordinate variables.

    We need this function as we can not rely on xarray.Dataset.coords,
    because 2D coordinate arrays are most likely not indicated as such
    in many datasets.
    """

    # Collect bounds variables. We must exclude them.
    bounds_vars = set()
    for k in dataset.variables:
        var = dataset[k]
        bounds_k = var.attrs.get('bounds')
        if bounds_k is not None and bounds_k in dataset:
            bounds_vars.add(bounds_k)

    potential_coord_vars = []

    # First consider any CF global attribute "coordinates"
    coordinates = dataset.attrs.get('coordinates')
    if coordinates is not None:
        for var_name in coordinates.split():
            if _is_potential_coord_var(dataset, bounds_vars, var_name):
                potential_coord_vars.append(var_name)

    # Then consider any other 1D/2D variables
    for var_name in dataset.variables:
        if var_name not in potential_coord_vars \
                and _is_potential_coord_var(dataset, bounds_vars, var_name):
            potential_coord_vars.append(var_name)

    return potential_coord_vars


def _is_potential_coord_var(dataset: xr.Dataset, bounds_var_names: Set[str], var_name: Hashable) -> bool:
    if var_name in dataset:
        var = dataset[var_name]
        return var.ndim in (1, 2) and var_name not in bounds_var_names
    return False
