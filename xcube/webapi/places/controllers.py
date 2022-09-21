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

from typing import Any, Dict, Union, Optional

import shapely.geometry
import shapely.wkt
from shapely.errors import WKTReadingError

from xcube.constants import LOG
from xcube.core.geom import get_box_split_bounds_geometry
from xcube.util.perf import measure_time
from .context import ALL_PLACES
from .context import PlacesContext
from ...server.api import ApiError
from ...util.assertions import assert_instance

GeoJsonFeatureCollection = Dict
GeoJsonFeature = Dict

GeometryLike = Union[str, Dict, shapely.geometry.base.BaseGeometry]


def find_places(
        ctx: PlacesContext,
        place_group_id: str,
        base_url: str,
        query_geometry: Optional[GeometryLike] = None,
        query_expr: Optional[Any] = None,
        comb_op: str = "and"
) -> GeoJsonFeatureCollection:
    try:
        if isinstance(query_geometry, str):
            try:
                query_geometry = get_box_split_bounds_geometry(
                    *[float(s) for s in query_geometry.split(",")]
                )
            except (TypeError, ValueError) as e:
                query_geometry = shapely.wkt.loads(query_geometry)
        elif isinstance(query_geometry, dict):
            if query_geometry["type"] == "FeatureCollection":
                query_geometry = shapely.geometry.shape(
                    query_geometry["features"][0]["geometry"]
                )
            elif query_geometry["type"] == "Feature":
                query_geometry = shapely.geometry.shape(
                    query_geometry["geometry"]
                )
            else:
                query_geometry = shapely.geometry.shape(
                    query_geometry
                )
        elif query_geometry is not None:
            assert_instance(query_geometry,
                            shapely.geometry.base.BaseGeometry,
                            name='query_geometry')
    except (WKTReadingError, TypeError, IndexError, ValueError, KeyError) as e:
        raise ApiError.BadRequest(
            "Received invalid geometry bbox, geometry WKT, or GeoJSON object"
        ) from e

    with measure_time() as cm:
        features = _find_places(ctx,
                                place_group_id,
                                base_url,
                                query_geometry,
                                query_expr,
                                comb_op)
    LOG.info(f"{len(features)} places found within {cm.duration} seconds")
    return features


def _find_places(
        ctx: PlacesContext,
        place_group_id: str,
        base_url: str,
        query_geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
        query_expr: Optional[Any] = None,
        comb_op: str = "and"
) -> GeoJsonFeatureCollection:
    if comb_op is not None and comb_op != "and":
        raise ApiError.NotImplemented("comb_op not yet supported")

    if place_group_id == ALL_PLACES:
        place_groups = ctx.get_global_place_groups(base_url,
                                                   load_features=True)
        features = []
        for place_group in place_groups:
            features.extend(place_group['features'])
        feature_collection = dict(type="FeatureCollection",
                                  features=features)
    else:
        feature_collection = ctx.get_global_place_group(place_group_id,
                                                        base_url,
                                                        load_features=True)
        feature_collection = dict(type="FeatureCollection",
                                  features=feature_collection['features'])

    if query_geometry is None:
        if query_expr is None:
            return feature_collection
        else:
            raise ApiError.NotImplemented("query_expr not yet supported")
    else:
        matching_places = []
        if query_expr is None:
            for feature in feature_collection['features']:
                geometry = shapely.geometry.shape(feature['geometry'])
                if geometry.intersects(query_geometry):
                    matching_places.append(feature)
        else:
            raise NotImplementedError()
        return dict(type="FeatureCollection",
                    features=matching_places)
