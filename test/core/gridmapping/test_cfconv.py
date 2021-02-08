import shutil
import unittest
from typing import Set

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping.cfconv import GridCoords
from xcube.core.gridmapping.cfconv import GridMapping
from xcube.core.gridmapping.cfconv import get_dataset_grid_mappings

CRS_WGS84 = pyproj.crs.CRS(4326)
CRS_CRS84 = pyproj.crs.CRS.from_string("urn:ogc:def:crs:OGC:1.3:CRS84")
CRS_UTM_33N = pyproj.crs.CRS(32633)

CRS_ROTATED_POLE = pyproj.crs.CRS.from_cf(
    dict(grid_mapping_name="rotated_latitude_longitude",
         grid_north_pole_latitude=32.5,
         grid_north_pole_longitude=170.))


class GetDatasetGridMappingsTest(unittest.TestCase):
    def test_no_crs_lon_lat_common_names(self):
        dataset = xr.Dataset(coords=dict(lon=xr.DataArray(np.linspace(10, 12, 11), dims='lon'),
                                         lat=xr.DataArray(np.linspace(50, 52, 11), dims='lat')))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual('latitude_longitude', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('lon', grid_mapping.coords.x.name)
        self.assertEqual('lat', grid_mapping.coords.y.name)

    def test_no_crs_lon_lat_standard_names(self):
        dataset = xr.Dataset(coords=dict(weird_x=xr.DataArray(np.linspace(10, 12, 11), dims='i',
                                                              attrs=dict(standard_name='longitude')),
                                         weird_y=xr.DataArray(np.linspace(50, 52, 11), dims='j',
                                                              attrs=dict(standard_name='latitude'))))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual('latitude_longitude', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('weird_x', grid_mapping.coords.x.name)
        self.assertEqual('weird_y', grid_mapping.coords.y.name)

    def test_crs_x_y_with_common_names(self):
        dataset = xr.Dataset(dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
                             coords=dict(x=xr.DataArray(np.linspace(1000, 12000, 11), dims='x'),
                                         y=xr.DataArray(np.linspace(5000, 52000, 11), dims='y')))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('crs', grid_mappings)
        grid_mapping = grid_mappings.get('crs')
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual('transverse_mercator', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('x', grid_mapping.coords.x.name)
        self.assertEqual('y', grid_mapping.coords.y.name)

    def test_crs_x_y_with_standard_names(self):
        dataset = xr.Dataset(dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
                             coords=dict(myx=xr.DataArray(np.linspace(1000, 12000, 11), dims='x',
                                                          attrs=dict(standard_name='projection_x_coordinate')),
                                         myy=xr.DataArray(np.linspace(5000, 52000, 11), dims='y',
                                                          attrs=dict(standard_name='projection_y_coordinate'))))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('crs', grid_mappings)
        grid_mapping = grid_mappings.get('crs')
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual('transverse_mercator', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('myx', grid_mapping.coords.x.name)
        self.assertEqual('myy', grid_mapping.coords.y.name)

    def test_rotated_pole_with_common_names(self):
        dataset = xr.Dataset(dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
                             coords=dict(rlon=xr.DataArray(np.linspace(-180, 180, 11), dims='rlon'),
                                         rlat=xr.DataArray(np.linspace(0, 90, 11), dims='rlat')))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('rotated_pole', grid_mappings)
        grid_mapping = grid_mappings.get('rotated_pole')
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual('Geographic 2D CRS', grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('rlon', grid_mapping.coords.x.name)
        self.assertEqual('rlat', grid_mapping.coords.y.name)

    def test_rotated_pole_with_standard_names(self):
        dataset = xr.Dataset(dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
                             coords=dict(u=xr.DataArray(np.linspace(-180, 180, 11), dims='u',
                                                        attrs=dict(standard_name='grid_longitude')),
                                         v=xr.DataArray(np.linspace(0, 90, 11), dims='v',
                                                        attrs=dict(standard_name='grid_latitude'))))
        grid_mappings = get_dataset_grid_mappings(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('rotated_pole', grid_mappings)
        grid_mapping = grid_mappings.get('rotated_pole')
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual('Geographic 2D CRS', grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('u', grid_mapping.coords.x.name)
        self.assertEqual('v', grid_mapping.coords.y.name)


class XarrayDecodeCfTest(unittest.TestCase):
    """Find out how xarray treats 1D and 2D coordinate variables when decode_cf=True or =False"""

    def test_cf_1d_coords(self):
        self._write_coords(*self._gen_1d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=False)

    def test_cf_1d_data_vars(self):
        self._write_data_vars(*self._gen_1d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=False)

    def test_cf_2d_coords(self):
        self._write_coords(*self._gen_2d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=False)

    def test_cf_2d_data_vars(self):
        self._write_data_vars(*self._gen_2d())
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=True)
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=False)

    def assertVarNames(self, exp_data_vars: Set[str], exp_coords: Set[str], decode_cf: bool):
        ds1 = xr.open_zarr('noise.zarr', decode_cf=decode_cf)
        self.assertEqual(exp_coords, set(ds1.coords))
        self.assertEqual(exp_data_vars, set(ds1.data_vars))

    @classmethod
    def _write_coords(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs), coords=dict(lon=lon, lat=lat))
        dataset.to_zarr('noise.zarr', mode='w')

    @classmethod
    def _write_data_vars(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs, lon=lon, lat=lat))
        dataset.to_zarr('noise.zarr', mode='w')

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('noise.zarr')

    def _gen_1d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=('lat', 'lon'))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims='lon')
        lat = xr.DataArray(np.linspace(50, 52, 11), dims='lat')
        noise.attrs['grid_mapping'] = 'crs'
        lon.attrs['standard_name'] = 'longitude'
        lat.attrs['standard_name'] = 'latitude'
        self.assertEqual(('lon',), lon.dims)
        self.assertEqual(('lat',), lat.dims)
        return noise, crs, lon, lat

    def _gen_2d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=('y', 'x'))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims='x')
        lat = xr.DataArray(np.linspace(50, 52, 11), dims='y')
        lat, lon = xr.broadcast(lat, lon)
        noise.attrs['grid_mapping'] = 'crs'
        lon.attrs['standard_name'] = 'longitude'
        lat.attrs['standard_name'] = 'latitude'
        self.assertEqual(('y', 'x'), lon.dims)
        self.assertEqual(('y', 'x'), lat.dims)
        return noise, crs, lon, lat
