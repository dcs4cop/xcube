#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

import datetime
import json
import os.path
import pathlib

import tornado.web
from tornado.ioloop import IOLoop

from xcube.constants import LOG
from xcube.core.timecoord import timestamp_to_iso_string
from xcube.util.perf import measure_time
from xcube.util.versions import get_xcube_versions
from xcube.version import version
from xcube.webapi.auth import AuthMixin
from xcube.webapi.controllers.catalogue import (
    get_datasets,
    get_dataset_coordinates,
    get_color_bars,
    get_dataset,
    get_dataset_place_groups,
    get_dataset_place_group
)
from xcube.webapi.controllers.ogc.wmts import (
    get_wmts_capabilities_xml,
    get_crs_name_from_tms_id,
    WMTS_CRS84_TMS_ID,
    WMTS_TILE_FORMAT,
    WMTS_VERSION,
    WMTS_WEB_MERCATOR_TMS_ID
)
from xcube.webapi.controllers.places import (
    find_places,
    find_dataset_places
)
from xcube.webapi.controllers.tiles import (
    compute_ml_dataset_tile,
    get_dataset_tile,
    get_dataset_tile_grid,
    get_legend
)
from xcube.webapi.controllers.timeseries import get_time_series
from xcube.webapi.controllers.ts_legacy import (
    get_time_series_info,
    get_time_series_for_point,
    get_time_series_for_geometry,
    get_time_series_for_geometry_collection,
    get_time_series_for_feature_collection
)
from xcube.webapi.defaults import (
    SERVER_NAME,
    SERVER_DESCRIPTION
)
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.s3util import (
    dict_to_xml,
    list_s3_bucket_v1,
    list_bucket_result_to_xml,
    list_s3_bucket_v2,
    mtime_to_str,
    str_to_etag)
from xcube.webapi.service import ServiceRequestHandler

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

_LOG_S3BUCKET_HANDLER = False

_VALID_WMTS_TMS_IDS = (WMTS_CRS84_TMS_ID, WMTS_WEB_MERCATOR_TMS_ID)


def _assert_valid_tms_id(tms_id: str):
    if tms_id not in _VALID_WMTS_TMS_IDS:
        raise ServiceBadRequestError(
            f'Value for "tilematrixset" parameter'
            f' must be one of {_VALID_WMTS_TMS_IDS!r}'
        )


# noinspection PyAbstractClass
class WMTSKvpHandler(ServiceRequestHandler):

    async def get(self):
        # According to WMTS 1.0 spec, query parameters
        # must be case-insensitive.
        self.set_caseless_query_arguments()

        service = self.params.get_query_argument('service')
        if service != "WMTS":
            raise ServiceBadRequestError(
                'Value for "service" parameter must be "WMTS"'
            )
        request = self.params.get_query_argument('request')
        if request == "GetCapabilities":
            wmts_version = self.params.get_query_argument(
                "version", WMTS_VERSION
            )
            if wmts_version != WMTS_VERSION:
                raise ServiceBadRequestError(
                    f'Value for "version" parameter must be "{WMTS_VERSION}"'
                )
            tms_id = self.params.get_query_argument(
                "tilematrixset", WMTS_CRS84_TMS_ID
            )
            _assert_valid_tms_id(tms_id)
            capabilities = await IOLoop.current().run_in_executor(
                None,
                get_wmts_capabilities_xml,
                self.service_context,
                self.base_url,
                tms_id
            )
            self.set_header("Content-Type", "application/xml")
            await self.finish(capabilities)

        elif request == "GetTile":
            wmts_version = self.params.get_query_argument("version",
                                                          WMTS_VERSION)
            if wmts_version != WMTS_VERSION:
                raise ServiceBadRequestError(
                    f'Value for "version" parameter must be "{WMTS_VERSION}"'
                )
            layer = self.params.get_query_argument("layer")
            try:
                ds_id, var_name = layer.split(".")
            except ValueError as e:
                raise ServiceBadRequestError(
                    'Value for "layer" parameter must be'
                    ' "<dataset>.<variable>"'
                ) from e
            # For time being, we ignore "style"
            # style = self.params.get_query_argument("style"
            mime_type = self.params.get_query_argument(
                "format", WMTS_TILE_FORMAT
            ).lower()
            if mime_type not in (WMTS_TILE_FORMAT, "png"):
                raise ServiceBadRequestError(
                    f'Value for "format" parameter'
                    f' must be "{WMTS_TILE_FORMAT}"'
                )
            tms_id = self.params.get_query_argument(
                'tilematrixset', WMTS_CRS84_TMS_ID
            )
            _assert_valid_tms_id(tms_id)
            crs_name = get_crs_name_from_tms_id(tms_id)
            x = self.params.get_query_argument_int("tilecol")
            y = self.params.get_query_argument_int("tilerow")
            z = self.params.get_query_argument_int("tilematrix")
            tile = await IOLoop.current().run_in_executor(
                None,
                compute_ml_dataset_tile,
                self.service_context,
                ds_id,
                var_name,
                crs_name,
                x, y, z,
                self.params
            )
            self.set_header("Content-Type", "image/png")
            await self.finish(tile)
        elif request == "GetFeatureInfo":
            raise ServiceBadRequestError(
                'Request type "GetFeatureInfo" not yet implemented'
            )
        else:
            raise ServiceBadRequestError(
                f'Invalid request type "{request}"'
            )


# noinspection PyAbstractClass
class GetWMTSCapabilitiesXmlHandler(ServiceRequestHandler):

    async def get(self):
        capabilities = await IOLoop.current().run_in_executor(
            None,
            get_wmts_capabilities_xml,
            self.service_context,
            self.base_url,
            WMTS_CRS84_TMS_ID
        )
        self.set_header('Content-Type', 'application/xml')
        await self.finish(capabilities)


# noinspection PyAbstractClass
class GetWMTSCapabilitiesXmlTmsHandler(ServiceRequestHandler):

    async def get(self, tms_id: str):
        _assert_valid_tms_id(tms_id)
        capabilities = await IOLoop.current().run_in_executor(
            None,
            get_wmts_capabilities_xml,
            self.service_context,
            self.base_url,
            tms_id
        )
        self.set_header('Content-Type', 'application/xml')
        await self.finish(capabilities)


# noinspection PyAbstractClass
class GetDatasetsHandler(ServiceRequestHandler, AuthMixin):

    def get(self):
        with measure_time('Get granted scopes'):
            granted_scopes = self.granted_scopes
        details = bool(int(self.params.get_query_argument('details', '0')))
        tile_client = self.params.get_query_argument('tiles', None)
        point = self.params.get_query_argument_point('point', None)
        response = get_datasets(self.service_context,
                                details=details,
                                client=tile_client,
                                point=point,
                                base_url=self.base_url,
                                granted_scopes=granted_scopes)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response, indent=None if details else 2))


# noinspection PyAbstractClass
class GetDatasetHandler(ServiceRequestHandler, AuthMixin):

    def get(self, ds_id: str):
        with measure_time('Get granted scopes'):
            granted_scopes = self.granted_scopes
        tile_client = self.params.get_query_argument('tiles', None)
        response = get_dataset(self.service_context,
                               ds_id,
                               client=tile_client,
                               base_url=self.base_url,
                               granted_scopes=granted_scopes)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response, indent=2))


# noinspection PyAbstractClass
class GetDatasetPlaceGroupsHandler(ServiceRequestHandler):

    def get(self, ds_id: str):
        response = get_dataset_place_groups(self.service_context, ds_id, self.base_url)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response))


# noinspection PyAbstractClass
class GetDatasetPlaceGroupHandler(ServiceRequestHandler):

    def get(self, ds_id: str, place_group_id: str):
        response = get_dataset_place_group(self.service_context, ds_id, place_group_id, self.base_url)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response))


# noinspection PyAbstractClass
class GetDatasetCoordsHandler(ServiceRequestHandler):

    def get(self, ds_id: str, dim_name: str):
        response = get_dataset_coordinates(self.service_context, ds_id, dim_name)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response))


# noinspection PyAbstractClass
class ListS3BucketHandler(ServiceRequestHandler):

    async def get(self):

        prefix = self.get_query_argument('prefix', default=None)
        delimiter = self.get_query_argument('delimiter', default=None)
        max_keys = int(self.get_query_argument('max-keys', default='1000'))
        list_s3_bucket_params = dict(prefix=prefix, delimiter=delimiter,
                                     max_keys=max_keys)

        list_type = self.get_query_argument('list-type', default=None)
        if list_type is None:
            marker = self.get_query_argument('marker', default=None)
            list_s3_bucket_params.update(marker=marker)
            list_s3_bucket = list_s3_bucket_v1
        elif list_type == '2':
            start_after = self.get_query_argument('start-after', default=None)
            continuation_token = self.get_query_argument('continuation-token', default=None)
            list_s3_bucket_params.update(start_after=start_after, continuation_token=continuation_token)
            list_s3_bucket = list_s3_bucket_v2
        else:
            raise ServiceBadRequestError(f'Unknown bucket list type {list_type!r}')

        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'GET: list_s3_bucket_params={list_s3_bucket_params}')
        bucket_mapping = self.service_context.get_s3_bucket_mapping()
        list_bucket_result = list_s3_bucket(bucket_mapping, **list_s3_bucket_params)
        if _LOG_S3BUCKET_HANDLER:
            import json
            LOG.info(f'-->\n{json.dumps(list_bucket_result, indent=2)}')

        xml = list_bucket_result_to_xml(list_bucket_result)
        self.set_header('Content-Type', 'application/xml')
        self.write(xml)
        await self.flush()


# noinspection PyAbstractClass
class GetS3BucketObjectHandler(ServiceRequestHandler):
    async def head(self, ds_id: str, path: str = ''):
        key, local_path = self._get_key_and_local_path(ds_id, path)
        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'HEAD: key={key!r}, local_path={local_path!r}')
        if local_path is None or not local_path.exists():
            await self._key_not_found(key)
            return
        self.set_header('ETag', str_to_etag(str(local_path)))
        self.set_header('Last-Modified', mtime_to_str(local_path.stat().st_mtime))
        if local_path.is_file():
            self.set_header('Content-Length', local_path.stat().st_size)
        else:
            self.set_header('Content-Length', 0)
        await self.finish()

    async def get(self, ds_id: str, path: str = ''):
        key, local_path = self._get_key_and_local_path(ds_id, path)
        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'GET: key={key!r}, local_path={local_path!r}')
        if local_path is None or not local_path.exists():
            await self._key_not_found(key)
            return
        self.set_header('ETag', str_to_etag(str(local_path)))
        self.set_header('Last-Modified', mtime_to_str(local_path.stat().st_mtime))
        self.set_header('Content-Type', 'binary/octet-stream')
        if local_path.is_file():
            self.set_header('Content-Length', local_path.stat().st_size)
            chunk_size = 1024 * 1024
            with open(str(local_path), 'rb') as fp:
                while True:
                    chunk = fp.read(chunk_size)
                    if len(chunk) == 0:
                        break
                    self.write(chunk)
                    await self.flush()
        else:
            self.set_header('Content-Length', 0)
            await self.finish()

    def _key_not_found(self, key: str):
        self.set_header('Content-Type', 'application/xml')
        self.set_status(404)
        return self.finish(dict_to_xml('Error',
                                       dict(Code='NoSuchKey',
                                            Message='The specified key does not exist.',
                                            Key=key)))

    def _get_key_and_local_path(self, ds_id: str, path: str):
        dataset_config = self.service_context.get_dataset_config(ds_id)
        file_system = dataset_config.get('FileSystem', 'file')
        required_file_systems = ['file', 'local']
        if file_system not in required_file_systems:
            required_file_system_string = " or ".join(required_file_systems)
            raise ServiceBadRequestError(
                f'AWS S3 data access: currently, only datasets in file systems '
                f'{required_file_system_string!r} are supported, but dataset '
                f'{ds_id!r} uses file system {file_system!r}')

        key = f'{ds_id}/{path}'

        # validate path
        if path and '..' in path.split('/'):
            raise ServiceBadRequestError(f'AWS S3 data access: received illegal key {key!r}')

        bucket_mapping = self.service_context.get_s3_bucket_mapping()
        local_path = bucket_mapping.get(ds_id)
        local_path = os.path.join(local_path, path)

        local_path = os.path.normpath(local_path)

        return key, pathlib.Path(local_path)


# noinspection PyAbstractClass,PyBroadException
class GetWMTSTileHandler(ServiceRequestHandler):

    async def get(self,
                  ds_id: str,
                  var_name: str,
                  z: str, y: str, x: str):
        self.set_caseless_query_arguments()
        tms_id = self.params.get_query_argument(
            'tilematrixset', WMTS_CRS84_TMS_ID
        )
        _assert_valid_tms_id(tms_id)
        crs_name = get_crs_name_from_tms_id(tms_id)
        tile = await IOLoop.current().run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.service_context,
            ds_id,
            var_name,
            crs_name,
            x, y, z,
            self.params
        )
        self.set_header('Content-Type', 'image/png')
        await self.finish(tile)


# noinspection PyAbstractClass,PyBroadException
class GetWMTSTileTmsHandler(ServiceRequestHandler):

    async def get(self,
                  ds_id: str,
                  var_name: str,
                  tms_id: str,
                  z: str, y: str, x: str):
        self.set_caseless_query_arguments()
        crs_name = get_crs_name_from_tms_id(tms_id)
        tile = await IOLoop.current().run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.service_context,
            ds_id,
            var_name,
            crs_name,
            x, y, z,
            self.params
        )
        self.set_header('Content-Type', 'image/png')
        await self.finish(tile)


# noinspection PyAbstractClass,PyBroadException
class GetDatasetVarTileHandler(ServiceRequestHandler):

    async def get(self, ds_id: str, var_name: str, z: str, x: str, y: str):
        tile = await IOLoop.current().run_in_executor(None,
                                                      get_dataset_tile,
                                                      self.service_context,
                                                      ds_id, var_name,
                                                      x, y, z,
                                                      self.params)
        self.set_header('Content-Type', 'image/png')
        await self.finish(tile)


# noinspection PyAbstractClass,PyBroadException
class GetDatasetVarTile2Handler(ServiceRequestHandler):

    async def get(self, ds_id: str, var_name: str, z: str, y: str, x: str):
        tile = await IOLoop.current().run_in_executor(None,
                                                      compute_ml_dataset_tile,
                                                      self.service_context,
                                                      ds_id,
                                                      var_name,
                                                      None,
                                                      x, y, z,
                                                      self.params)
        self.set_header('Content-Type', 'image/png')
        await self.finish(tile)


# noinspection PyAbstractClass,PyBroadException
class GetDatasetVarLegendHandler(ServiceRequestHandler):

    async def get(self, ds_id: str, var_name: str):
        tile = await IOLoop.current().run_in_executor(None,
                                                      get_legend,
                                                      self.service_context,
                                                      ds_id, var_name,
                                                      self.params)
        self.set_header('Content-Type', 'image/png')
        await self.finish(tile)


# noinspection PyAbstractClass
class GetDatasetVarTileGridHandler(ServiceRequestHandler):

    def get(self, ds_id: str, var_name: str):
        tile_client = self.params.get_query_argument('tiles', "ol4")
        response = get_dataset_tile_grid(self.service_context,
                                         ds_id, var_name,
                                         tile_client, self.base_url)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(response, indent=2))


# noinspection PyAbstractClass
class GetColorBarsJsonHandler(ServiceRequestHandler):

    # noinspection PyShadowingBuiltins
    def get(self):
        mime_type = 'application/json'
        response = get_color_bars(self.service_context, mime_type)
        self.set_header('Content-Type', mime_type)
        self.write(response)


# noinspection PyAbstractClass
class GetColorBarsHtmlHandler(ServiceRequestHandler):

    # noinspection PyShadowingBuiltins
    def get(self):
        mime_type = 'text/html'
        response = get_color_bars(self.service_context, mime_type)
        self.set_header('Content-Type', mime_type)
        self.write(response)


# noinspection PyAbstractClass
class GetPlaceGroupsHandler(ServiceRequestHandler):

    # noinspection PyShadowingBuiltins
    def get(self):
        response = self.service_context.get_global_place_groups(self.base_url)
        self.set_header('Content-Type', "application/json")
        self.write(json.dumps(response, indent=2))


# noinspection PyAbstractClass
class FindPlacesHandler(ServiceRequestHandler):

    # noinspection PyShadowingBuiltins
    def get(self, place_group_id: str):
        query_expr = self.params.get_query_argument("query", None)
        geom_wkt = self.params.get_query_argument("geom", None)
        box_coords = self.params.get_query_argument("bbox", None)
        comb_op = self.params.get_query_argument("comb", "and")
        if geom_wkt and box_coords:
            raise ServiceBadRequestError('Only one of "geom" and "bbox" may be given')
        response = find_places(self.service_context,
                               place_group_id,
                               self.base_url,
                               geom_wkt=geom_wkt, box_coords=box_coords,
                               query_expr=query_expr, comb_op=comb_op)
        self.set_header('Content-Type', "application/json")
        self.write(json.dumps(response, indent=2))

    # noinspection PyShadowingBuiltins
    def post(self, place_group_id: str):
        query_expr = self.params.get_query_argument("query", None)
        comb_op = self.params.get_query_argument("comb", "and")
        geojson_obj = self.get_body_as_json_object()
        response = find_places(self.service_context,
                               place_group_id,
                               self.base_url,
                               geojson_obj=geojson_obj,
                               query_expr=query_expr, comb_op=comb_op)
        self.set_header('Content-Type', "application/json")
        self.write(json.dumps(response, indent=2))


# noinspection PyAbstractClass
class FindDatasetPlacesHandler(ServiceRequestHandler):

    # noinspection PyShadowingBuiltins
    def get(self, place_group_id: str, ds_id: str):
        query_expr = self.params.get_query_argument("query", None)
        comb_op = self.params.get_query_argument("comb", "and")
        response = find_dataset_places(self.service_context,
                                       place_group_id,
                                       ds_id,
                                       self.base_url,
                                       query_expr=query_expr,
                                       comb_op=comb_op)
        self.set_header('Content-Type', "application/json")
        self.write(json.dumps(response, indent=2))


# noinspection PyAbstractClass
class InfoHandler(ServiceRequestHandler):

    def get(self):
        config_time = timestamp_to_iso_string(datetime.datetime.fromtimestamp(self.service_context.config_mtime),
                                              freq="ms")
        server_time = timestamp_to_iso_string(datetime.datetime.now(), freq="ms")
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(dict(name=SERVER_NAME,
                                   description=SERVER_DESCRIPTION,
                                   version=version,
                                   versions=get_xcube_versions(),
                                   configTime=config_time,
                                   serverTime=server_time,
                                   serverPID=os.getpid()),
                              indent=2))


# noinspection PyAbstractClass
class MaintenanceHandler(ServiceRequestHandler):

    def get(self, action: str):
        def get_service():
            from xcube.webapi.service import Service
            # noinspection PyUnresolvedReferences
            service: Service = self.application.service
            assert isinstance(service, Service)
            return service

        if action == "update":
            LOG.warning("Forcing resource update...")
            IOLoop.current().add_callback(get_service().update_config)
            self.write("OK")
        elif action == "stop":
            get_service().stop()
            self.write("OK")
        elif action == "kill":
            get_service().stop(kill=True)
            self.write("OK")
        elif action == "fail":
            code = self.params.get_query_argument_int(
                'code', default=None
            )
            message = self.params.get_query_argument(
                'message', default='This is just a test error.'
            )
            if code is None:
                raise ValueError(message)
            else:
                raise tornado.web.HTTPError(code, log_message=message)
        else:
            raise tornado.web.HTTPError(
                400, log_message=f'Unknown action {action}'
            )


# noinspection PyAbstractClass
class GetTimeSeriesHandler(ServiceRequestHandler):

    async def post(self, ds_id: str, var_name: str):
        geo_json_object = self.get_body_as_json_object("GeoJSON object")
        agg_methods = self.params.get_query_argument('aggMethods', default=None)
        agg_methods = agg_methods.split(',') if agg_methods else None
        start_date = self.params.get_query_argument_datetime('startDate', default=None)
        end_date = self.params.get_query_argument_datetime('endDate', default=None)
        max_valids = self.params.get_query_argument_int('maxValids', default=None)
        _check_max_valids(max_valids)

        result = await IOLoop.current().run_in_executor(None,
                                                        get_time_series,
                                                        self.service_context,
                                                        ds_id,
                                                        var_name,
                                                        geo_json_object,
                                                        agg_methods,
                                                        start_date,
                                                        end_date,
                                                        max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(dict(result=result))


# noinspection PyAbstractClass
class GetTsLegacyInfoHandler(ServiceRequestHandler):

    async def get(self):
        response = await IOLoop.current().run_in_executor(None, get_time_series_info, self.service_context)
        self.set_header('Content-Type', 'application/json')
        await self.finish(response)


# noinspection PyAbstractClass
class GetTsLegacyForPointHandler(ServiceRequestHandler):

    async def get(self, ds_id: str, var_name: str):
        lon = self.params.get_query_argument_float('lon')
        lat = self.params.get_query_argument_float('lat')
        start_date = self.params.get_query_argument_datetime('startDate', default=None)
        end_date = self.params.get_query_argument_datetime('endDate', default=None)
        max_valids = self.params.get_query_argument_int('maxValids', default=None)
        _check_max_valids(max_valids)

        response = await IOLoop.current().run_in_executor(None,
                                                          get_time_series_for_point,
                                                          self.service_context,
                                                          ds_id, var_name,
                                                          lon, lat,
                                                          start_date,
                                                          end_date,
                                                          max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(response)


# noinspection PyAbstractClass
class GetTsLegacyForGeometryHandler(ServiceRequestHandler):

    async def post(self, ds_id: str, var_name: str):
        start_date = self.params.get_query_argument_datetime('startDate', default=None)
        end_date = self.params.get_query_argument_datetime('endDate', default=None)
        incl_count = self.params.get_query_argument_int('inclCount', default=1) != 0
        incl_stdev = self.params.get_query_argument_int('inclStDev', default=0) != 0
        max_valids = self.params.get_query_argument_int('maxValids', default=None)
        _check_max_valids(max_valids)
        geometry = self.get_body_as_json_object("GeoJSON geometry")

        response = await IOLoop.current().run_in_executor(None,
                                                          get_time_series_for_geometry,
                                                          self.service_context,
                                                          ds_id, var_name,
                                                          geometry,
                                                          start_date, end_date,
                                                          incl_count, incl_stdev,
                                                          max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(response)


# noinspection PyAbstractClass
class GetTsLegacyForGeometriesHandler(ServiceRequestHandler):

    async def post(self, ds_id: str, var_name: str):
        start_date = self.params.get_query_argument_datetime('startDate', default=None)
        end_date = self.params.get_query_argument_datetime('endDate', default=None)
        incl_count = self.params.get_query_argument_int('inclCount', default=1) != 0
        incl_stdev = self.params.get_query_argument_int('inclStDev', default=0) != 0
        max_valids = self.params.get_query_argument_int('maxValids', default=None)
        _check_max_valids(max_valids)
        geometry_collection = self.get_body_as_json_object("GeoJSON geometry collection")

        response = await IOLoop.current().run_in_executor(None,
                                                          get_time_series_for_geometry_collection,
                                                          self.service_context,
                                                          ds_id, var_name,
                                                          geometry_collection,
                                                          start_date, end_date,
                                                          incl_count, incl_stdev,
                                                          max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(response)


# noinspection PyAbstractClass
class GetTsLegacyForFeaturesHandler(ServiceRequestHandler):

    async def post(self, ds_id: str, var_name: str):
        start_date = self.params.get_query_argument_datetime('startDate', default=None)
        end_date = self.params.get_query_argument_datetime('endDate', default=None)
        incl_count = self.params.get_query_argument_int('inclCount', default=1) != 0
        incl_stdev = self.params.get_query_argument_int('inclStDev', default=0) != 0
        max_valids = self.params.get_query_argument_int('maxValids', default=None)
        _check_max_valids(max_valids)
        feature_collection = self.get_body_as_json_object("GeoJSON feature collection")

        response = await IOLoop.current().run_in_executor(None,
                                                          get_time_series_for_feature_collection,
                                                          self.service_context,
                                                          ds_id, var_name,
                                                          feature_collection,
                                                          start_date, end_date,
                                                          incl_count, incl_stdev,
                                                          max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(response)


def _check_max_valids(max_valids):
    if not (max_valids is None or max_valids == -1 or max_valids > 0):
        raise ServiceBadRequestError('If given, query parameter "maxValids" must be -1 or positive')
