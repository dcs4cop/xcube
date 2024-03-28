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

import urllib.parse
import warnings
from typing import Dict, List, Tuple, Any, Mapping

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.tilingscheme import EARTH_CIRCUMFERENCE_WGS84
from xcube.core.tilingscheme import GEOGRAPHIC_CRS_NAME
from xcube.core.tilingscheme import TilingScheme
from xcube.core.tilingscheme import WEB_MERCATOR_CRS_NAME
from xcube.webapi.common.xml import Document
from xcube.webapi.common.xml import Element
from xcube.constants import CRS_CRS84
from .context import WmtsContext

WMTS_VERSION = "1.0.0"
WMTS_URL_PREFIX = f"wmts/{WMTS_VERSION}"
WMTS_TILE_FORMAT = "image/png"

WMTS_CRS84_TMS_ID = "WorldCRS84Quad"
WMTS_CRS84_TMS_TITLE = "CRS84 for the World"
OGC_CRS84_URN = "urn:ogc:def:crs:OGC:1.3:CRS84"
OGC_CRS84_WKSS_URN = "urn:ogc:def:wkss:OGC:1.0:GoogleCRS84Quad"

WMTS_WEB_MERCATOR_TMS_ID = "WorldWebMercatorQuad"
WMTS_WEB_MERCATOR_TMS_TITLE = "Google Maps Compatible for the World"
OGC_WEB_MERCATOR_URN = "urn:ogc:def:crs:EPSG::3857"
OGC_WEB_MERCATOR_WKSS_URN = "urn:ogc:def:wkss:OGC:1.0:GoogleMapsCompatible"

_STD_PIXEL_SIZE_IN_METERS = 0.28e-3

# '/tile/%s/%s/%s/' is the pattern for
# '/tile/{ds_name}/{var_name}/{tms_id}/'
_TILE_URL_TEMPLATE = (
    WMTS_URL_PREFIX + "/tile/%s/%s/%s/" "{TileMatrix}/" "{TileRow}/" "{TileCol}.png"
)


def get_wmts_capabilities_xml(
    ctx: WmtsContext, base_url: str, tms_id: str = WMTS_CRS84_TMS_ID
) -> str:
    """Get WMTSCapabilities.xml according to
    https://www.ogc.org/standards/wmts, WMTS 1.0.0.

    We create WMTSCapabilities.xml for individual tile matrix sets.
    If we'd include it into one, we would have to double all layers
    and make them available for the supported tile matrix sets
    "WorldCRS84Quad" and "WorldWebMercatorQuad".

    Args:
        ctx: server context
        base_url: the request base URL
        tms_id: time matrix set identifier,
            must be one of "WorldCRS84Quad" or "WorldWebMercatorQuad"
    Returns:
        XML plain text in UTF-8 encoding
    """
    element = get_capabilities_element(ctx, base_url, tms_id)
    document = Document(element)
    return document.to_xml(indent=4)


def get_capabilities_element(ctx: WmtsContext, base_url: str, tms_id: str) -> Element:
    service_provider_element = get_service_provider_element(ctx)
    operations_metadata_element = get_operations_metadata_element(ctx, base_url, tms_id)

    layer_base_url = ctx.get_service_url(base_url, _TILE_URL_TEMPLATE)

    contents_element = Element("Contents")
    themes_element = Element("Themes")

    common_tiling_scheme = TilingScheme.for_crs(get_crs_name_from_tms_id(tms_id))

    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config["Identifier"]

        ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_name)
        grid_mapping = ml_dataset.grid_mapping
        ds = ml_dataset.base_dataset

        x_name, y_name = grid_mapping.xy_dim_names

        try:
            crs84_bbox = get_crs84_bbox(grid_mapping)
        except ValueError:
            warnings.warn(
                f"cannot compute geographical"
                f" bounds for dataset {ds_name}, ignoring it"
            )
            continue

        ds_theme_element = get_ds_theme_element(ds_name, ds, dataset_config)
        themes_element.add(ds_theme_element)

        dim_elements_cache = dict()

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            var = ds[var_name]

            is_spatial_var = (
                var.ndim >= 2 and var.dims[-1] == x_name and var.dims[-2] == y_name
            )
            if not is_spatial_var:
                continue

            var_layer_element, var_theme_element = get_var_layer_and_theme_element(
                ds_name, var_name, var, crs84_bbox, layer_base_url, tms_id
            )

            dim_elements = get_dim_elements(ds_name, ds, var, dim_elements_cache)

            var_layer_element.add(*dim_elements)
            contents_element.add(var_layer_element)
            ds_theme_element.add(var_theme_element)

        # Here we compute min/max level for all datasets
        # which is actually not right. Otherwise, we had to create a new
        # TMS for every dataset (or even variable).
        tiling_scheme = ml_dataset.derive_tiling_scheme(common_tiling_scheme)
        if common_tiling_scheme.min_level is None:
            min_level = tiling_scheme.min_level
            max_level = tiling_scheme.max_level
        else:
            min_level = min(common_tiling_scheme.min_level, tiling_scheme.min_level)
            max_level = max(common_tiling_scheme.max_level, tiling_scheme.max_level)
        common_tiling_scheme = common_tiling_scheme.derive(
            min_level=min_level, max_level=max_level
        )

    contents_element.add(
        get_tile_matrix_set_element(common_tiling_scheme),
    )

    service_identification_element = get_service_identification_element()
    service_metadata_url_element = get_service_metadata_url_element(
        ctx, base_url, tms_id
    )

    return Element(
        "Capabilities",
        attrs={
            "xmlns": "http://www.opengis.net/wmts/1.0",
            "xmlns:ows": "http://www.opengis.net/ows/1.1",
            "xmlns:xlink": "http://www.w3.org/1999/xlink",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://www.opengis.net/wmts/1.0"
            " http://schemas.opengis.net/wmts/1.0.0/"
            "wmtsGetCapabilities_response.xsd",
            "version": WMTS_VERSION,
        },
        elements=[
            service_identification_element,
            service_provider_element,
            operations_metadata_element,
            contents_element,
            themes_element,
            service_metadata_url_element,
        ],
    )


def get_dim_elements(
    ds_name: str,
    ds: xr.Dataset,
    var: xr.DataArray,
    dim_element_cache: Dict[str, Element],
) -> List[Element]:
    dim_elements = []
    for dim_name in var.dims[0:-2]:
        if dim_name not in ds.coords:
            continue

        dim_id = f"{ds_name}.{dim_name}"
        if dim_id in dim_element_cache:
            dim_elements.append(dim_element_cache[dim_id])
            continue

        coord_var = ds.coords[dim_name]
        if len(coord_var.shape) != 1:
            # strange case
            continue

        coord_bnds_var_name = coord_var.attrs.get("bounds", f"{dim_name}_bnds")
        coord_bnds_var = None
        if coord_bnds_var_name in ds:
            coord_bnds_var = ds.coords[coord_bnds_var_name]
        if coord_bnds_var is not None:
            if (
                len(coord_bnds_var.shape) != 2
                or coord_bnds_var.shape[0] != coord_bnds_var.shape[0]
                or coord_bnds_var.shape[1] != 2
            ):
                # strange case
                coord_bnds_var = None

        dim_title = coord_var.attrs.get("long_name", dim_name)
        if dim_name == "time":
            units = "ISO8601"
            default = "current"
            current = "true"
        else:
            units = coord_var.attrs.get("units", "")
            default = "0"
            current = "false"

        dim_element = Element(
            "Dimension",
            elements=[
                Element("ows:Identifier", text=f"{dim_name}"),
                Element("ows:Title", text=dim_title),
                Element("ows:UOM", text=units),
                Element("Default", text=default),
                Element("Current", text=current),
            ],
        )

        if coord_bnds_var is not None:
            coord_bnds_var_values = coord_bnds_var.values
            for i in range(len(coord_var)):
                value1 = coord_bnds_var_values[i, 0]
                value2 = coord_bnds_var_values[i, 1]
                dim_element.add(Element("Value", text=f"{value1}/{value2}"))
        else:
            coord_var_values = coord_var.values
            for i in range(len(coord_var)):
                value = coord_var_values[i]
                dim_element.add(Element("Value", text=f"{value}"))

        dim_element_cache[dim_id] = dim_element
        dim_elements.append(dim_element)

    return dim_elements


def get_ds_theme_element(
    ds_name: str, ds: xr.Dataset, dataset_config: Mapping[str, Any]
) -> Element:
    ds_title = dataset_config.get("Title", ds.attrs.get("title", ds_name))
    ds_abstract = dataset_config.get(
        "Abstract", ds.attrs.get("abstract", ds.attrs.get("comment", ""))
    )
    return Element(
        "Theme",
        elements=[
            Element("ows:Identifier", text=ds_name),
            Element("ows:Title", text=ds_title),
            Element("ows:Abstract", text=ds_abstract),
        ],
    )


def get_var_layer_and_theme_element(
    ds_name: str,
    var_name: str,
    var: xr.DataArray,
    var_geo_bbox: Tuple[float, float, float, float],
    var_tile_url_templ_pattern: str,
    tms_id: str,
) -> Tuple[Element, Element]:
    var_id = f"{ds_name}.{var_name}"
    var_title = (
        ds_name + "/" + var.attrs.get("title", var.attrs.get("long_name", var_name))
    )
    var_abstract = var.attrs.get("comment", var.attrs.get("abstract", ""))
    var_theme_element = Element(
        "Theme",
        elements=[
            Element("ows:Identifier", text=var_id),
            Element("ows:Title", text=var_title),
            Element("ows:Abstract", text=var_abstract),
            Element("LayerRef", text=var_id),
        ],
    )
    # noinspection PyTypeChecker
    var_tile_url_templ_params = tuple(
        map(urllib.parse.quote_plus, [ds_name, var_name, tms_id])
    )
    var_tile_url_templ = var_tile_url_templ_pattern % var_tile_url_templ_params
    layer_element = Element(
        "Layer",
        elements=[
            Element("ows:Identifier", text=f"{ds_name}.{var_name}"),
            Element("ows:Title", text=f"{var_title}"),
            Element("ows:Abstract", text=f"{var_abstract}"),
            Element(
                "ows:WGS84BoundingBox",
                elements=[
                    Element(
                        "ows:LowerCorner",
                        text=f"{var_geo_bbox[0]}" f" {var_geo_bbox[1]}",
                    ),
                    Element(
                        "ows:UpperCorner",
                        text=f"{var_geo_bbox[2]}" f" {var_geo_bbox[3]}",
                    ),
                ],
            ),
            Element(
                "Style",
                attrs={"isDefault": "true"},
                elements=[
                    Element("ows:Identifier", text="Default"),
                ],
            ),
            Element("Format", text=WMTS_TILE_FORMAT),
            Element(
                "TileMatrixSetLink",
                elements=[
                    Element("TileMatrixSet", text=tms_id),
                ],
            ),
            Element(
                "ResourceURL",
                attrs=dict(
                    format=WMTS_TILE_FORMAT,
                    resourceType="tile",
                    template=var_tile_url_templ,
                ),
            ),
        ],
    )
    return layer_element, var_theme_element


def get_dataset_tiling_scheme(
    ml_dataset: MultiLevelDataset, tms_id: str
) -> TilingScheme:
    if tms_id == WMTS_CRS84_TMS_ID:
        tiling_scheme = TilingScheme.GEOGRAPHIC
    else:
        tiling_scheme = TilingScheme.WEB_MERCATOR
    return ml_dataset.derive_tiling_scheme(tiling_scheme)


def get_tile_matrix_set_element(tiling_scheme: TilingScheme) -> Element:
    if tiling_scheme.crs_name == tiling_scheme.GEOGRAPHIC.crs_name:
        return get_tile_matrix_set_crs84_element(tiling_scheme)
    else:
        return get_tile_matrix_set_web_mercator_element(tiling_scheme)


def get_tile_matrix_set_crs84_element(tiling_scheme: TilingScheme) -> Element:
    return _get_tile_matrix_set_element(
        WMTS_CRS84_TMS_ID,
        WMTS_CRS84_TMS_TITLE,
        OGC_CRS84_URN,
        OGC_CRS84_WKSS_URN,
        (-180, -90, +180, +90),
        EARTH_CIRCUMFERENCE_WGS84 / 360.0,
        tiling_scheme.min_level,
        tiling_scheme.max_level,
    )


def get_tile_matrix_set_web_mercator_element(tiling_scheme: TilingScheme) -> Element:
    return _get_tile_matrix_set_element(
        WMTS_WEB_MERCATOR_TMS_ID,
        WMTS_WEB_MERCATOR_TMS_TITLE,
        OGC_WEB_MERCATOR_URN,
        OGC_WEB_MERCATOR_WKSS_URN,
        (-20037508.3427892, -20037508.3427892, +20037508.3427892, +20037508.3427892),
        1.0,
        tiling_scheme.min_level,
        tiling_scheme.max_level,
    )


def _get_tile_matrix_set_element(
    tms_id: str,
    tms_title: str,
    crs_urn: str,
    wkss_urn: str,
    bbox: Tuple[float, float, float, float],
    meters_per_pixel: float,
    min_level: int,
    max_level: int,
) -> Element:
    element = Element(
        "TileMatrixSet",
        elements=[
            Element("ows:Identifier", text=tms_id),
            Element("ows:Title", text=tms_title),
            Element("ows:SupportedCRS", text=crs_urn),
            Element(
                "ows:BoundingBox",
                attrs=dict(crs=crs_urn),
                elements=[
                    Element("ows:LowerCorner", text=f"{bbox[0]} {bbox[1]}"),
                    Element("ows:UpperCorner", text=f"{bbox[2]} {bbox[3]}"),
                ],
            ),
            Element("WellKnownScaleSet", text=wkss_urn),
        ],
    )

    scale_factor = meters_per_pixel / _STD_PIXEL_SIZE_IN_METERS

    tiling_scheme = TilingScheme.for_crs(get_crs_name_from_tms_id(tms_id))
    tile_width = tiling_scheme.tile_width
    tile_height = tiling_scheme.tile_height
    num_x_tiles_0, num_y_tiles_0 = tiling_scheme.num_level_zero_tiles
    resolutions = tiling_scheme.get_resolutions(
        min_level=min_level, max_level=max_level
    )
    for level, res in zip(range(min_level, max_level + 1), resolutions):
        factor = 2**level
        num_x_tiles = factor * num_x_tiles_0
        num_y_tiles = factor * num_y_tiles_0
        scale_denominator = scale_factor * res
        element.add(
            Element(
                "TileMatrix",
                elements=[
                    Element("ows:Identifier", text=f"{level}"),
                    Element("ScaleDenominator", text=f"{scale_denominator}"),
                    Element("TopLeftCorner", text=f"{bbox[0]} {bbox[3]}"),
                    Element("TileWidth", text=f"{tile_width}"),
                    Element("TileHeight", text=f"{tile_height}"),
                    Element("MatrixWidth", text=f"{num_x_tiles}"),
                    Element("MatrixHeight", text=f"{num_y_tiles}"),
                ],
            )
        )

    return element


def get_operations_metadata_element(
    ctx: WmtsContext, base_url: str, tms_id: str
) -> Element:
    wmts_kvp_url = ctx.get_service_url(base_url, "wmts/kvp?")
    wmts_rest_cap_url = ctx.get_service_url(
        base_url, f"{WMTS_URL_PREFIX}/{tms_id}/WMTSCapabilities.xml"
    )
    wmts_rest_tile_url = ctx.get_service_url(base_url, WMTS_URL_PREFIX + "/")
    operations_metadata = Element(
        "ows:OperationsMetadata",
        elements=[
            _get_operation_element("GetCapabilities", wmts_kvp_url, wmts_rest_cap_url),
            _get_operation_element("GetTile", wmts_kvp_url, wmts_rest_tile_url),
        ],
    )
    return operations_metadata


def _get_operation_element(op: str, kvp_url: str, rest_url: str) -> Element:
    return Element(
        "ows:Operation",
        attrs={"name": op},
        elements=[
            Element(
                "ows:DCP",
                elements=[
                    Element(
                        "ows:HTTP",
                        elements=[
                            _get_operation_get_element(kvp_url, "KVP"),
                            _get_operation_get_element(rest_url, "REST"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _get_operation_get_element(url: str, value: str) -> Element:
    return Element(
        "ows:Get",
        attrs={"xlink:href": url},
        elements=[
            Element(
                "ows:Constraint",
                attrs={"name": "GetEncoding"},
                elements=[
                    Element(
                        "ows:AllowedValues", elements=[Element("ows:Value", text=value)]
                    ),
                ],
            ),
        ],
    )


def get_service_provider_element(ctx):
    service_provider = ctx.config.get("ServiceProvider")

    def _get_value(path):
        v = None
        node = service_provider
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return ""
            v = node[k]
            node = v
        return str(v) if v is not None else ""

    element = Element(
        "ows:ServiceProvider",
        elements=[
            Element("ows:ProviderName", text=_get_value(["ProviderName"])),
            Element(
                "ows:ProviderSite", attrs={"xlink:href": _get_value(["ProviderSite"])}
            ),
            Element(
                "ows:ServiceContact",
                elements=[
                    Element(
                        "ows:IndividualName",
                        text=_get_value(["ServiceContact", "IndividualName"]),
                    ),
                    Element(
                        "ows:PositionName",
                        text=_get_value(["ServiceContact", "PositionName"]),
                    ),
                    Element(
                        "ows:ContactInfo",
                        elements=[
                            Element(
                                "ows:Phone",
                                elements=[
                                    Element(
                                        "ows:Voice",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Phone",
                                                "Voice",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:Facsimile",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Phone",
                                                "Facsimile",
                                            ]
                                        ),
                                    ),
                                ],
                            ),
                            Element(
                                "ows:Address",
                                elements=[
                                    Element(
                                        "ows:DeliveryPoint",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "DeliveryPoint",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:City",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "City",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:AdministrativeArea",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "AdministrativeArea",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:PostalCode",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "PostalCode",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:Country",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "Country",
                                            ]
                                        ),
                                    ),
                                    Element(
                                        "ows:ElectronicMailAddress",
                                        text=_get_value(
                                            [
                                                "ServiceContact",
                                                "ContactInfo",
                                                "Address",
                                                "ElectronicMailAddress",
                                            ]
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    return element


def get_service_identification_element():
    return Element(
        "ows:ServiceIdentification",
        elements=[
            Element("ows:Title", text="xcube WMTS"),
            Element(
                "ows:Abstract",
                text="Web Map Tile Service (WMTS)" " for xcube-conformant data cubes",
            ),
            Element(
                "ows:Keywords",
                elements=[
                    Element("ows:Keyword", text="tile"),
                    Element("ows:Keyword", text="tile matrix set"),
                    Element("ows:Keyword", text="map"),
                ],
            ),
            Element("ows:ServiceType", text="OGC WMTS"),
            Element("ows:ServiceTypeVersion", text=WMTS_VERSION),
            Element("ows:Fees", text="none"),
            Element("ows:AccessConstraints", text="none"),
        ],
    )


def get_service_metadata_url_element(
    ctx: WmtsContext, base_url: str, tms_id: str
) -> Element:
    get_caps_rest_url = ctx.get_service_url(
        base_url, f"{WMTS_URL_PREFIX}/{tms_id}/WMTSCapabilities.xml"
    )
    return Element("ServiceMetadataURL", attrs={"xlink:href": get_caps_rest_url})


# TODO: move get_crs84_bbox() into GridMapping so we can adjust
#   global dataset attributes with result
def get_crs84_bbox(grid_mapping: GridMapping) -> Tuple[float, float, float, float]:
    if grid_mapping.crs.is_geographic:
        return grid_mapping.xy_bbox
    t = pyproj.Transformer.from_crs(grid_mapping.crs, CRS_CRS84, always_xy=True)
    x1, y1, x2, y2 = grid_mapping.xy_bbox
    # TODO (forman): Fixme, this will be wrong
    #   for datasets crossing anti-meridian.
    x, y = t.transform(np.array([x1, x2, x1, x2]), np.array([y1, y1, y2, y2]))
    coords = np.array([np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)])
    if not np.all(np.isfinite(coords)):
        raise ValueError(
            "grid mapping bbox cannot" " be represented in geographical coordinates"
        )
    # noinspection PyTypeChecker
    return tuple(map(float, coords))


def get_crs_name_from_tms_id(tms_id: str) -> str:
    if tms_id == WMTS_CRS84_TMS_ID:
        return GEOGRAPHIC_CRS_NAME
    if tms_id == WMTS_WEB_MERCATOR_TMS_ID:
        return WEB_MERCATOR_CRS_NAME
    raise ValueError(
        f"Tile matrix set must be one of"
        f' "{WMTS_CRS84_TMS_ID}" and "{WMTS_WEB_MERCATOR_TMS_ID}"'
    )
