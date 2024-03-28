# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import asyncio
import concurrent.futures
import functools
import logging
import traceback
import urllib.parse
from typing import Any, Optional, Sequence, Union, Callable, Type, Awaitable, Mapping

import tornado.escape
import tornado.httputil
import tornado.ioloop
import tornado.web

from xcube.constants import LOG
from xcube.constants import LOG_LEVEL_DETAIL
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.server.api import ApiRequest
from xcube.server.api import ApiResponse
from xcube.server.api import ApiRoute
from xcube.server.api import ApiStaticRoute
from xcube.server.api import ArgT
from xcube.server.api import Context
from xcube.server.api import JSON
from xcube.server.api import ReturnT
from xcube.server.config import get_reverse_url_prefix
from xcube.server.config import get_url_prefix
from xcube.server.framework import Framework
from xcube.util.assertions import assert_true
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.version import version

SERVER_CTX_ATTR_NAME = "__xcube_server_ctx"


class TornadoFramework(Framework):
    """
    The Tornado web server framework.
    """

    def __init__(
        self,
        application: Optional[tornado.web.Application] = None,
        io_loop: Optional[tornado.ioloop.IOLoop] = None,
        shared_io_loop: bool = False,
    ):
        self._application = application or tornado.web.Application()
        self._io_loop = io_loop
        self._shared_io_loop = io_loop is not None and shared_io_loop
        self.configure_logging()

    @property
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """Returns the JSON Schema for the configuration of the
        `tornado.httpserver.HTTPServer.initialize()`
        method.
        """
        return JsonObjectSchema(
            properties=dict(
                tornado=JsonObjectSchema(
                    # See kwargs of tornado.httpserver.HTTPServer.initialize()
                    properties=dict(
                        xheaders=JsonBooleanSchema(default=False),
                        protocol=JsonStringSchema(nullable=True),
                        no_keep_alive=JsonBooleanSchema(default=False),
                        decompress_request=JsonBooleanSchema(default=False),
                        ssl_options=JsonObjectSchema(additional_properties=True),
                        chunk_size=JsonIntegerSchema(
                            nullable=True, exclusive_minimum=0
                        ),
                        max_header_size=JsonIntegerSchema(
                            nullable=True, exclusive_minimum=0
                        ),
                        max_body_size=JsonIntegerSchema(
                            nullable=True, exclusive_minimum=0
                        ),
                    ),
                    additional_properties=True,
                )
            )
        )

    @property
    def application(self) -> tornado.web.Application:
        return self._application

    @property
    def io_loop(self) -> tornado.ioloop.IOLoop:
        return self._io_loop or tornado.ioloop.IOLoop.current()

    def add_static_routes(self, api_routes: Sequence[ApiStaticRoute], url_prefix: str):
        handlers = []
        for api_route in api_routes:
            base_url = f"{url_prefix}{api_route.path}"
            default_filename = api_route.default_filename
            handlers.append(
                (f"{base_url}", tornado.web.RedirectHandler, {"url": f"{base_url}/"})
            )
            handlers.append(
                (
                    f"{base_url}/(.*)",
                    tornado.web.StaticFileHandler,
                    {"path": api_route.dir_path, "default_filename": default_filename},
                )
            )
            LOG.log(
                LOG_LEVEL_DETAIL,
                f"Added static route"
                f" {api_route.path!r}"
                f" from API {api_route.api_name!r}",
            )
        if handlers:
            self.application.add_handlers(".*$", handlers)

    def add_routes(self, api_routes: Sequence[ApiRoute], url_prefix: str):
        handlers = []
        for api_route in api_routes:
            handlers.append(
                (
                    url_prefix + self.path_to_pattern(api_route.path),
                    TornadoRequestHandler,
                    {"api_route": api_route},
                )
            )

            LOG.log(
                LOG_LEVEL_DETAIL,
                f"Added route"
                f" {api_route.path!r}"
                f" from API {api_route.api_name!r}",
            )
        if handlers:
            self.application.add_handlers(".*$", handlers)

    def update(self, ctx: Context):
        setattr(self.application, SERVER_CTX_ATTR_NAME, ctx)

    def start(self, ctx: Context):
        config = ctx.config

        port = config["port"]
        address = config["address"]
        url_prefix = get_url_prefix(config)
        tornado_settings = config.get("tornado", {})

        self.application.listen(port, address=address, **tornado_settings)

        address_ = "127.0.0.1" if address == "0.0.0.0" else address
        # TODO: get test URL template from configuration
        test_url = f"http://{address_}:{port}{url_prefix}/openapi.html"
        LOG.info(f"Service running, listening on {address}:{port}")
        LOG.info(f"Try {test_url}")
        LOG.info(f"Press CTRL+C to stop service")

        if not self._shared_io_loop:
            self.io_loop.start()

    def stop(self, ctx: Context):
        if not self._shared_io_loop:
            self.io_loop.stop()

    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        return self.io_loop.call_later(delay, callback, *args, **kwargs)

    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[ReturnT]:
        return self.io_loop.run_in_executor(
            executor,
            functools.partial(function, **kwargs) if kwargs else function,
            *args,
        )

    @staticmethod
    def configure_logging():
        # Configure Tornado loggers to use root handlers, so we
        # have a single log level and format. Root handlers are
        # assumed to be already configured by xcube CLI.
        for logger_name in [
            "tornado",
            "tornado.access",
            "tornado.application",
            "tornado.general",
        ]:
            log = logging.getLogger(logger_name)
            # Remove Tornado's own handlers
            for h in list(log.handlers):
                log.removeHandler(h)
                h.close()
            # Add root handlers configured by xcube CLI
            for h in list(logging.root.handlers):
                log.addHandler(h)
            # Use common log level
            log.setLevel(logging.root.level)

    @staticmethod
    def path_to_pattern(path: str):
        """Convert a string *pattern* where any occurrences of `{NAME}`
        are replaced by an equivalent regex expression which will
        assign matching character groups to NAME. Characters match until
        one of the RFC 2396 reserved characters is found or the end of
        the *pattern* is reached.

        Args:
            path: URL path
        Returns:
            equivalent regex pattern
        Raises:
            ValueError: if *pattern* is invalid
        """
        var_pattern = r"(?P<%s>[^\;\/\?\:\@\&\=\+\$\,]+)"
        rest_var_pattern = r"\/?(?P<%s>.*)"
        num_rest_vars = 0
        rest_var_seen = False
        reg_expr = ""
        pos = 0
        while True:
            pos1 = path.find("{", pos)
            if pos1 >= 0:
                pos2 = path.find("}", pos1 + 1)
                if pos2 <= pos1:
                    raise ValueError('missing closing "}" in "%s"' % path)
                arg = path[pos1 + 1 : pos2]
                if arg.startswith("*"):
                    rest_var_seen = True
                    name = arg[1:]
                    pattern = rest_var_pattern
                    if pos1 > 0 and path[pos1 - 1] == "/":
                        # Consume a trailing "/" because it is
                        # covered in pattern
                        pos1 -= 1
                    num_rest_vars += 1
                else:
                    rest_var_seen = False
                    name = arg
                    pattern = var_pattern
                if not name.isidentifier():
                    raise ValueError(
                        '"{name}" in path must be a valid identifier,'
                        ' but got "%s"' % arg
                    )
                reg_expr += path[pos:pos1] + (pattern % name)
                pos = pos2 + 1
            else:
                pos2 = path.find("}", pos)
                if pos2 >= pos:
                    raise ValueError('missing opening "{" in "%s"' % path)
                rest_of_path = path[pos:]
                if rest_var_seen and rest_of_path:
                    raise ValueError(
                        "wildcard variable must be last in path,"
                        f' but path was "{path}"'
                    )
                reg_expr += rest_of_path
                rest_var_seen = False
                break
        if num_rest_vars > 1:
            raise ValueError(
                "only a single wildcard variable is allowed,"
                f' but found {num_rest_vars} in path "{path}"'
            )
        return reg_expr


# noinspection PyAbstractClass
class TornadoRequestHandler(tornado.web.RequestHandler):
    def __init__(
        self,
        application: tornado.web.Application,
        request: tornado.httputil.HTTPServerRequest,
        **kwargs: Any,
    ):
        super().__init__(application, request)
        server_ctx = getattr(application, SERVER_CTX_ATTR_NAME, None)
        assert isinstance(server_ctx, Context)
        api_route: ApiRoute = kwargs.pop("api_route")
        ctx: Context = server_ctx.get_api_ctx(api_route.api_name)
        self._api_handler: ApiHandler = api_route.handler_cls(
            ctx,
            TornadoApiRequest(
                request,
                get_url_prefix(server_ctx.config),
                get_reverse_url_prefix(server_ctx.config),
            ),
            TornadoApiResponse(self),
            **api_route.handler_kwargs,
        )

    def set_default_headers(self):
        self.set_header("Server", f"xcube-server/{version}")
        # TODO: get from config
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET,PUT,DELETE,OPTIONS")
        self.set_header(
            "Access-Control-Allow-Headers",
            "x-requested-with,access-control-allow-origin,"
            "authorization,content-type",
        )

    def write_error(self, status_code: int, **kwargs: Any):
        valid_json_types = str, int, float, bool, type(None)
        error_info = {
            k: v for k, v in kwargs.items() if isinstance(v, valid_json_types)
        }
        error_info.update(status_code=status_code)
        if "exc_info" in kwargs:
            exc_type, exc_val, exc_tb = kwargs["exc_info"]
            error_info.update(
                exception=traceback.format_exception(exc_type, exc_val, exc_tb),
                message=str(exc_val),
            )
            if isinstance(exc_val, tornado.web.HTTPError) and exc_val.reason:
                error_info.update(reason=exc_val.reason)
        self.finish({"error": error_info})

    async def head(self, *args, **kwargs):
        await self._call_method("head", *args, **kwargs)

    async def get(self, *args, **kwargs):
        await self._call_method("get", *args, **kwargs)

    async def post(self, *args, **kwargs):
        await self._call_method("post", *args, **kwargs)

    async def put(self, *args, **kwargs):
        await self._call_method("put", *args, **kwargs)

    async def delete(self, *args, **kwargs):
        await self._call_method("delete", *args, **kwargs)

    async def options(self, *args, **kwargs):
        await self._call_method("options", *args, **kwargs)

    async def _call_method(self, method_name: str, *args, **kwargs):
        method = getattr(self._api_handler, method_name)
        try:
            if asyncio.iscoroutinefunction(method):
                await method(*args, **kwargs)
            else:
                method(*args, **kwargs)
        except ApiError as e:
            raise tornado.web.HTTPError(e.status_code, log_message=e.message) from e


class TornadoApiRequest(ApiRequest):
    def __init__(
        self,
        request: tornado.httputil.HTTPServerRequest,
        url_prefix: str = "",
        reverse_url_prefix: str = "",
    ):
        self._request = request
        self._url_prefix = url_prefix
        self._reverse_url_prefix = reverse_url_prefix
        self._is_query_lower_case = False
        # print("full_url:", self._request.full_url())
        # print("protocol:", self._request.protocol)
        # print("host:", self._request.host)
        # print("uri:", self._request.uri)
        # print("path:", self._request.path)
        # print("query:", self._request.query)

    def make_query_lower_case(self):
        self._is_query_lower_case = True

    @functools.cached_property
    def query(self) -> Mapping[str, Sequence[str]]:
        mapping = urllib.parse.parse_qs(self._request.query)
        if self._is_query_lower_case:
            mapping = {k.lower(): v for k, v in mapping.items()}
        return mapping

    # noinspection PyShadowingBuiltins
    def get_query_args(
        self, name: str, type: Optional[Type[ArgT]] = None
    ) -> Sequence[ArgT]:
        name_lc = name.lower() if self._is_query_lower_case else name
        if not self.query or name_lc not in self.query:
            return []
        values = self.query[name_lc]
        if type is not None and type is not str:
            assert_true(callable(type), "type must be callable")
            try:
                return [type(v) for v in values]
            except (ValueError, TypeError):
                raise ApiError.BadRequest(
                    f"Query parameter {name!r}" f" must have type {type.__name__!r}."
                )
        return values

    def url_for_path(
        self, path: str, query: Optional[str] = None, reverse: bool = False
    ) -> str:
        """Get the URL for given *path* and *query*.
        If the *reverse* flag is set, the configuration parameter
        `reverse_url_prefix`, if provided, is used to construct the URL,
        otherwise only `url_prefix`, if provided, is used.
        """
        prefix = self._url_prefix
        if reverse:
            prefix = self._reverse_url_prefix or prefix
        uri = ""
        if path:
            uri = path if path.startswith("/") else "/" + path
        if query:
            uri += "?" + query
        if "://" in prefix:
            # Absolute prefix
            return f"{prefix}{uri}"
        else:
            # Relative prefix
            protocol = self._request.protocol
            host = self._request.host
            return f"{protocol}://{host}{prefix}{uri}"

    @property
    def url(self) -> str:
        return self._request.full_url()

    @property
    def headers(self) -> Mapping[str, str]:
        return self._request.headers

    @property
    def body(self) -> bytes:
        return self._request.body

    @property
    def json(self) -> JSON:
        try:
            return tornado.escape.json_decode(self.body)
        except ValueError as e:
            raise ApiError.BadRequest(f"Body does not contain valid JSON: {e}") from e


class TornadoApiResponse(ApiResponse):
    def __init__(self, handler: TornadoRequestHandler):
        self._handler = handler

    def set_header(self, name: str, value: str):
        self._handler.set_header(name, value)

    def set_status(self, status_code: int, reason: Optional[str] = None):
        self._handler.set_status(status_code, reason=reason)

    def write(self, data: Union[str, bytes, JSON], content_type: Optional[str] = None):
        if data is not None:
            self._handler.write(data)
        # https://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.write
        # "If the given chunk is a dictionary, we write it as JSON and set the
        # Content-Type of the response to be application/json. (if you want to
        # send JSON as a different Content-Type, call set_header after calling
        # write())."
        if content_type is not None:
            self._handler.set_header("Content-Type", content_type)

    def finish(
        self, data: Union[str, bytes, JSON] = None, content_type: Optional[str] = None
    ):
        self.write(data, content_type)
        return self._handler.finish()
