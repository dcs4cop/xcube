# The MIT License (MIT)
# Copyright (c) 2020-2024 by the xcube development team and contributors
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

from xcube.core.store.fs.accessor import COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonComplexSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

REF_STORE_SCHEMA = JsonObjectSchema(
    properties=dict(
        refs=JsonArraySchema(
            description=(
                "The list of references to use for this"
                " instance. Items can be a path / URL to a reference"
                " JSON file or dictionaries with keys `'ref_path'` and"
                " `'data_descriptor'`. Reference files (URLs or local paths)"
                " are used in conjunction with target_options and"
                " target_protocol to open and parse JSON at this location."
            ),
            items=JsonComplexSchema(
                one_of=[
                    JsonStringSchema(min_length=1),
                    JsonObjectSchema(
                        properties=dict(
                            ref_path=JsonStringSchema(min_length=1),
                            data_id=JsonStringSchema(min_length=1),
                            # We could detail data_descriptor a bit more. Not now.
                            data_descriptor=JsonObjectSchema(
                                additional_properties=True
                            ),
                        ),
                        required=["ref_path"],
                    ),
                ],
            ),
            min_items=1,
        ),
        target_protocol=JsonStringSchema(
            description=(
                "Used for loading the reference files."
                " If not given, protocol will be derived from the given path"
            )
        ),
        target_options=JsonObjectSchema(
            description=("Extra filesystem options for loading the reference files."),
            additional_properties=True,
        ),
        remote_protocol=JsonStringSchema(
            description=(
                "The protocol of the filesystem on which the references"
                " will be evaluated. If not given, will be derived from"
                " the first URL in the references that has a protocol."
            )
        ),
        remote_options=JsonObjectSchema(
            description=("Extra filesystem options for loading the referenced data."),
            additional_properties=True,
        ),
        max_gap=JsonIntegerSchema(description="See max_block."),
        max_block=JsonIntegerSchema(
            description=(
                "For merging multiple concurrent requests to the same"
                " remote file. Neighboring byte ranges will only be"
                " merged when their inter-range gap is <= `max_gap`."
                " Default is 64KB. Set to 0 to only merge when it"
                " requires no extra bytes. Pass a negative number to"
                " disable merging, appropriate for local target files."
                " Neighboring byte ranges will only be merged when the"
                " size of the aggregated range is <= `max_block`."
                " Default is 256MB."
            )
        ),
        cache_size=JsonIntegerSchema(
            description=(
                "Maximum size of LRU cache, where"
                " cache_size*record_size denotes the total number of"
                " references that can be loaded in memory at once."
                " Only used for lazily loaded references."
            )
        ),
        **COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
    ),
    additional_properties=True,
)
