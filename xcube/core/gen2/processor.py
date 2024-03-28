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

from abc import abstractmethod, ABC

import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema

# Name of the (instance) method that processes a
# dataset and returns a new one.
# The one and only argument of the method is an xarray.Dataset.
# Then any number of processing parameters can be passed as
# keyword-arguments.
# See DatasetProcessor#process_dataset.
METHOD_NAME_DATASET_PROCESSOR = "process_dataset"

# Name of the (static/class) method that returns
# the JSON Object Schema for the processor parameters.
# The methods takes no arguments.
# See DatasetProcessor#get_process_params_schema.
METHOD_NAME_PARAMS_SCHEMA_GETTER = "get_process_params_schema"


class DatasetProcessor(ABC):
    """A generic dataset processor."""

    @abstractmethod
    def process_dataset(self, dataset: xr.Dataset, **process_params) -> xr.Dataset:
        """Process *dataset* into a new dataset using the
        given processing parameters *process_params*.

        The passed *process_params* must validate against
        the JSON schema returned by the `get_process_params_schema()`
        class method.

        Args:
            dataset: The dataset to be processed.
            **process_params: The processing parameters.

        Returns:
            A new dataset.
        """

    @classmethod
    def get_process_params_schema(cls) -> JsonObjectSchema:
        """Get the JSON Schema for the processing parameters passed
        to `process_dataset()`.

        The parameters are described by the JSON Object Schema's
        properties.

        The default implementation returns a JSON Schema
        that allows for any parameters of any type.

        Returns:
            a JSON Object Schema.
        """
        return JsonObjectSchema(additional_properties=True)
