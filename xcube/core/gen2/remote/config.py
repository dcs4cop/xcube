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

from typing import Dict
from typing import Union

from xcube.util.config import load_json_or_yaml_config
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

DEFAULT_ENDPOINT_URL = "https://xcube-gen.brockmann-consult.de/api/v2/"

ServiceConfigLike = Union[str, Dict, "ServiceConfig"]


class ServiceConfig(JsonObject):
    def __init__(
        self,
        endpoint_url: str = None,
        client_id: str = None,
        client_secret: str = None,
        access_token: str = None,
    ):
        endpoint_url = endpoint_url or DEFAULT_ENDPOINT_URL
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        self.endpoint_url = endpoint_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token

    @classmethod
    def normalize(cls, service_config: ServiceConfigLike) -> "ServiceConfig":
        """Normalize given *service_config* to an instance of
        `ServiceConfig`.

        If *service_config* is already a ServiceConfig it is returned as is.

        If it is a `str`, it is interpreted as a YAML or JSON file path
        and the configuration is read from file
        using `ServiceConfig.from_file()`.´The file content may include
        template variables that are interpolated by environment variables,
        e.g. "${XCUBE_GEN_CLIENT_SECRET}".

        If it is a `dict`, it is interpreted as a JSON object and the
        request is parsed using `ServiceConfig.from_dict()`.

        Args:
            service_config The remote configuration,
                or configuration file path, or configuration JSON object.
        Raises:
            TypeError if *service_config* is not a `CubeGeneratorRequest`,
            `str`, or `dict`.
        """
        if isinstance(service_config, ServiceConfig):
            return service_config
        if isinstance(service_config, str):
            return ServiceConfig.from_file(service_config)
        if isinstance(service_config, dict):
            return ServiceConfig.from_dict(service_config)
        raise TypeError(
            "service_config must be a str, dict, " "or a ServiceConfig instance"
        )

    @classmethod
    def from_file(cls, service_config_file: str) -> "ServiceConfig":
        service_config = load_json_or_yaml_config(service_config_file)
        cls.get_schema().validate_instance(service_config)
        return ServiceConfig(**service_config)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                endpoint_url=JsonStringSchema(min_length=1),
                client_id=JsonStringSchema(min_length=1),
                client_secret=JsonStringSchema(min_length=1),
                access_token=JsonStringSchema(min_length=1),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )
