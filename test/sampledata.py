import numpy as np
import pandas as pd
import xarray as xr


def new_test_dataset(time, height=180, **indexers):
    """
    Create a test dataset with dimensions ("time", "lat", "lon") and data variables given by *indexers*.

    :param time: Single date/time string or sequence of date-time strings.
    :param height: Size of the latitude dimension.
    :param indexers: Variable name to value mapping.
           Value may be a scalar or a vector of same length as *time*.
    :return: test dataset
    """
    # TODO (forman): get rid of this code here, utilise xcube.core.new_cube() instead
    time = [time] if isinstance(time, str) else time
    width = height * 2
    num_times = len(time)
    res = 180 / height
    shape = (1, height, width)
    data_vars = dict()
    for name, value in indexers.items():
        try:
            values = list(value)
        except TypeError:
            values = [value] * num_times
        if len(values) != num_times:
            raise ValueError()
        data_vars[name] = (['time', 'lat', 'lon'],
                           np.concatenate(tuple(np.full(shape, values[i]) for i in range(num_times))))
    time = np.array(pd.to_datetime(time), dtype=np.datetime64)
    return xr.Dataset(data_vars,
                      coords=dict(time=(
                          ['time'], time, dict(units='nanoseconds since 1970-01-01', calendar='proleptic_gregorian')),
                          lat=(['lat'], np.linspace(-90 + res, +90 - res, height)),
                          lon=(['lon'], np.linspace(-180 + res, +180 - res, width))))


def create_s2plus_dataset():
    x = xr.DataArray([310005., 310015., 310025., 310035., 310045.], dims=["x"],
                     attrs=dict(units="m", standard_name="projection_x_coordinate"))
    y = xr.DataArray([5689995., 5689985., 5689975., 5689965., 5689955.], dims=["y"],
                     attrs=dict(units="m", standard_name="projection_y_coordinate"))
    lon = xr.DataArray([[0.272763, 0.272906, 0.273050, 0.273193, 0.273336],
                        [0.272768, 0.272911, 0.273055, 0.273198, 0.273342],
                        [0.272773, 0.272917, 0.273060, 0.273204, 0.273347],
                        [0.272779, 0.272922, 0.273066, 0.273209, 0.273352],
                        [0.272784, 0.272927, 0.273071, 0.273214, 0.273358]],
                       dims=["y", "x"], attrs=dict(units="degrees_east", standard_name="longitude"))
    lat = xr.DataArray([[51.329464, 51.329464, 51.329468, 51.32947, 51.329475],
                        [51.329372, 51.329376, 51.32938, 51.329384, 51.329388],
                        [51.329285, 51.329285, 51.32929, 51.329292, 51.329296],
                        [51.329193, 51.329197, 51.32920, 51.329205, 51.329205],
                        [51.329100, 51.329105, 51.32911, 51.329113, 51.329117]],
                       dims=["y", "x"], attrs=dict(units="degrees_north", standard_name="latitude"))
    rrs_443 = xr.DataArray([[0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
                            [0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998]],
                           dims=["y", "x"], attrs=dict(units="sr-1", grid_mapping="transverse_mercator"))
    rrs_665 = xr.DataArray([[0.025002, 0.019001, 0.008999, 0.012001, 0.022999],
                            [0.028000, 0.021000, 0.009998, 0.008999, 0.022999],
                            [0.036999, 0.022999, 0.007999, 0.008999, 0.023998],
                            [0.041000, 0.022999, 0.007000, 0.009998, 0.021000],
                            [0.033001, 0.018002, 0.007999, 0.008999, 0.021000]],
                           dims=["y", "x"], attrs=dict(units="sr-1", grid_mapping="transverse_mercator"))
    transverse_mercator = xr.DataArray(np.array([0xffffffff], dtype=np.uint32),
                                       attrs=dict(grid_mapping_name="transverse_mercator",
                                                  scale_factor_at_central_meridian=0.9996,
                                                  longitude_of_central_meridian=3.0,
                                                  latitude_of_projection_origin=0.0,
                                                  false_easting=500000.0,
                                                  false_northing=0.0,
                                                  semi_major_axis=6378137.0,
                                                  inverse_flattening=298.257223563))
    return xr.Dataset(dict(rrs_443=rrs_443, rrs_665=rrs_665, transverse_mercator=transverse_mercator),
                      coords=dict(x=x, y=y, lon=lon, lat=lat),
                      attrs={
                          "title": "T31UCS_20180802T105621",
                          "conventions": "CF-1.6",
                          "institution": "VITO",
                          "product_type": "DCS4COP Sentinel2 Product",
                          "origin": "Copernicus Sentinel Data",
                          "project": "DCS4COP",
                          "time_coverage_start": "2018-08-02T10:59:38.888000Z",
                          "time_coverage_end": "2018-08-02T10:59:38.888000Z"
                      })


def create_highroc_dataset(no_spectra=False):
    """
    Simulates a HIGHROC OLCI L2 product in NetCDF 4 format
    """
    lon = np.array([[8, 9.3, 10.6, 11.9],
                    [8, 9.2, 10.4, 11.6],
                    [8, 9.1, 10.2, 11.3]], dtype=np.float32)
    lat = np.array([[56, 56.1, 56.2, 56.3],
                    [55, 55.2, 55.4, 55.6],
                    [54, 54.3, 54.6, 54.9]], dtype=np.float32)

    if not no_spectra:
        wavelengths = [(1, 400.0), (2, 412.5), (3, 442.5), (4, 490.0), (5, 510.0),
                       (6, 560.0), (7, 620.0), (8, 665.0), (9, 673.75), (10, 681.25),
                       (11, 708.75), (12, 753.75), (16, 778.75), (17, 865.0), (18, 885.0), (21, 940.0)]
        rtoa_desc = "Top-of-atmosphere reflectance"
        rrs_desc = "Atmospherically corrected angular dependent remote sensing reflectances"
        rtoa_vars = {f'rtoa_{i}': create_waveband(i, wl, '1', rtoa_desc) for i, wl in wavelengths}
        rrs_vars = {f'rrs_{i}': create_waveband(i, wl, 'sr^-1', rrs_desc) for i, wl in wavelengths}
    else:
        rtoa_vars = {}
        rrs_vars = {}

    return xr.Dataset(
        data_vars=dict(
            conc_chl=create_conc_chl(),
            c2rcc_flags=create_c2rcc_flag_var(),
            lon=(('y', 'x'), lon, dict(
                long_name="longitude",
                units="degrees_east",
            )),
            lat=(('y', 'x'), lat, dict(
                long_name="latitude",
                units="degrees_north",
            )),
            **rtoa_vars,
            **rrs_vars,
        ),
        attrs=dict(start_date='14-APR-2017 10:27:50.183264',
                   stop_date='14-APR-2017 10:31:42.736226')
    )


def create_waveband(index, wavelength, units, long_name=None):
    data = np.array([[7, 11, np.nan, 5],
                     [5, 10, 2, 21],
                     [16, 6, 20, 17]], dtype=np.float32)
    return (('y', 'x'), data, dict(
        long_name=long_name,
        units=units,
        spectral_band_index=index,
        wavelength=wavelength,
        bandwidth=15.0,
        valid_pixel_expression="c2rcc_flags.F1",
        _FillValue=np.nan,
    ))


def create_conc_chl():
    data = np.array([[7, 11, np.nan, 5],
                     [5, 10, 2, 21],
                     [16, 6, 20, 17]], dtype=np.float32)
    return (('y', 'x'), data, dict(
        long_name="Chlorophylll concentration",
        units="mg m^-3",
        _FillValue=np.nan,
        valid_pixel_expression="c2rcc_flags.F1",
    ))


def create_c2rcc_flag_var():
    data = np.array([[1, 1, 1, 1],
                     [1, 4, 1, 2],
                     [8, 1, 1, 1]], dtype=np.uint32)
    return xr.DataArray(data, dims=('y', 'x'), name='c2rcc_flags', attrs=dict(
        long_name="C2RCC quality flags",
        _Unsigned="true",
        flag_meanings="F1 F2 F3 F4",
        flag_masks=np.array([1, 2, 4, 8], np.int32),
        flag_coding_name="c2rcc_flags",
        flag_descriptions="D1 D2 D3 D4",
    ))


def create_cmems_sst_flag_var():
    sea = 1
    land = 2
    lake = 4
    ice = 8
    data = np.array([[[sea + ice, land + ice, lake + ice, lake],
                      [sea + ice, sea, land, land],
                      [sea, sea, sea, land]]], dtype=np.float32)
    return xr.DataArray(data, dims=('time', 'lat', 'lon'), name='mask', attrs=dict(
        long_name="land sea ice lake bit mask",
        flag_masks="0b, 1b, 2b, 3b",
        flag_meanings="sea land lake ice",
        valid_min=0,
        valid_max=12,
    ))


class SourceDatasetMixin:

    @classmethod
    def new_2x2_dataset_with_irregular_coords(cls):
        lon = np.array([[1.0, 6.0],
                        [0.0, 2.0]])
        lat = np.array([[56.0, 53.0],
                        [52.0, 50.0]])
        rad = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                               lat=xr.DataArray(lat, dims=('y', 'x')),
                               rad=xr.DataArray(rad, dims=('y', 'x'))))

    @classmethod
    def new_2x2_dataset_with_irregular_coords_antimeridian(cls):
        lon = np.array([[+179.0, -176.0],
                        [+178.0, +180.0]])
        lat = np.array([[56.0, 53.0],
                        [52.0, 50.0]])
        rad = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                               lat=xr.DataArray(lat, dims=('y', 'x')),
                               rad=xr.DataArray(rad, dims=('y', 'x'))))
