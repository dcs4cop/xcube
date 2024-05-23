import xarray as xr


def compute_indexes(dataset):
    b04 = dataset.B04
    b05 = dataset.B05
    b06 = dataset.B06
    b08 = dataset.B08
    b8a = dataset.B8A
    b11 = dataset.B11
    scl = dataset.SCL
    cld = dataset.CLD

    # See https://gisgeography.com/sentinel-2-bands-combinations/

    # moisture_index
    #
    moisture_index = (b8a - b11) / (b8a + b11)
    moisture_index.attrs.update(
        color_value_min=0.0,
        color_value_max=0.5,
        color_bar_name="Blues",
        units="-",
        description="Simple moisture index: (B8A-B11) / (B8A+B11)",
    )
    moisture_index = moisture_index.where(cld < 75)

    # vegetation_index
    #
    vegetation_index = (b08 - b04) / (b08 + b04)
    vegetation_index.attrs.update(
        color_value_min=0.0,
        color_value_max=1.0,
        color_bar_name="Greens",
        units="-",
        description="Simple vegetation index or NDVI: (B08-B04) / (B08+B04)",
    )
    vegetation_index = vegetation_index.where(cld < 75)

    # chlorophyll_index
    #
    b_from, b_peek, b_to = b04, b05, b06
    wlen_from = b04.attrs["wavelength"]
    wlen_peek = b05.attrs["wavelength"]
    wlen_to = b06.attrs["wavelength"]
    f = (wlen_peek - wlen_from) / (wlen_to - wlen_from)
    chlorophyll_index = (b_peek - b_from) - f * (b_to - b_from)
    chlorophyll_index.attrs.update(
        color_value_min=0.0,
        color_value_max=0.025,
        color_bar_name="viridis",
        units="-",
        description="Maximum chlorophyll index: (B05-B04) - f * (B06-B04)",
    )
    chlorophyll_index = chlorophyll_index.where(scl == 6)  # where "water"

    return xr.Dataset(
        dict(
            moisture_index=moisture_index,
            vegetation_index=vegetation_index,
            chlorophyll_index=chlorophyll_index,
        )
    )
