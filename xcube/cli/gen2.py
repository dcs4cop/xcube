# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

import json
import sys
import traceback

import click


@click.command(name="gen2", hidden=True)
@click.argument('request_path',
                type=str,
                required=False,
                metavar='REQUEST')
@click.option('--stores', 'stores_config_path',
              metavar='STORES_CONFIG',
              help='A JSON or YAML file that maps store names to '
                   'parameterized data stores.')
@click.option('--service', 'service_config_path',
              metavar='SERVICE_CONFIG',
              help='A JSON or YAML file that provides the configuration for an'
                   ' xcube Generator Service. If provided, the REQUEST will be'
                   ' passed to the service instead of generating the cube on '
                   ' this computer.')
@click.option('--info', '-i',
              is_flag=True,
              help='Output information about the data cube to be generated.'
                   ' Does not generate the data cube.')
@click.option('--output', '-o', 'output_file',
              metavar='RESULT',
              help='Write cube information or generation result into JSON'
                   ' file RESULT.'
                   ' If omitted, the JSON is dumped to stdout.')
@click.option('--verbose', '-v', 'verbosity',
              count=True,
              help='Control amount of information dumped to stdout.'
                   ' May be given multiple time to output more details,'
                   ' e.g. "-vvv".')
def gen2(request_path: str,
         stores_config_path: str = None,
         service_config_path: str = None,
         output_file: str = None,
         info: bool = False,
         verbosity: int = 0):
    """
    Generator tool for data cubes.

    Creates a cube view from one or more cube stores, optionally performs some
    cube transformation, and writes the resulting cube to some target cube
    store.

    REQUEST is the cube generation request. It may be provided as a JSON or
    YAML file (file extensions ".json" or ".yaml"). If the REQUEST file
    argument is omitted, it is expected that the Cube generation request is
    piped as a JSON string.

    STORE_CONFIGS is a path to a JSON or YAML file with xcube data store
    configurations. It is a mapping of arbitrary store names to configured
    data stores. Entries are dictionaries that have a mandatory "store_id"
    property which is a name of a registered xcube data store.
    The optional "store_params" property may define data store specific
    parameters. The following example defines a data store named
    "my_s3_store" which is an AWS S3 bucket store, and a data store
    "my_test_store" for testing, which is an in-memory data store:

    \b
    {
        "my_s3_store": {
            "store_id": "s3",
            "store_params": {
                "bucket_name": "eurodatacube",
                "aws_access_key_id": "jokljkjoiqqjvlaksd",
                "aws_secret_access_key": "1728349182734983248234"
            }
        },
        "my_test_store": {
            "store_id": "memory"
        }
    }

    SERVICE_CONFIG is a path to a JSON or YAML file with an xcube Generator
    service configuration. Here is a JSON example:

    \b
    {
        "endpoint_url": "https://xcube-gen.eurodatacube.com/api/v2/",
        "client_id": "ALDLPUIOSD5NHS3103",
        "client_secret": "lfaolb3klv904kj23lkdfsjkf430894341"
    }

    Values in the file may also be interpolated from current
    environment variables. In this case templates are used, for example:

    \b
    {
        "endpoint_url": "https://xcube-gen.eurodatacube.com/api/v2/",
        "client_id": "${XCUBE_GEN_CLIENT_ID}",
        "client_secret": "${XCUBE_GEN_CLIENT_SECRET}"
    }

    """
    from xcube.core.gen2 import CubeGenerator
    from xcube.core.gen2 import CubeGeneratorError
    from xcube.core.gen2 import CubeGeneratorRequest
    from xcube.util.versions import get_xcube_versions

    error = None

    # noinspection PyBroadException
    try:
        request = CubeGeneratorRequest.from_file(request_path,
                                                 verbosity=verbosity)
        generator = CubeGenerator.new(
            stores_config=stores_config_path,
            service_config=service_config_path,
            verbosity=verbosity
        )

        if info:
            result = generator.get_cube_info(request).to_dict()
        else:
            result = generator.generate_cube(request).to_dict()
    except BaseException as e:
        error = e

        result = dict(status='error',
                      message=f'{error}',
                      traceback=traceback.format_tb(error.__traceback__))
        if isinstance(error, CubeGeneratorError):
            if error.status_code is not None:
                result.update(status_code=error.status_code)
            if error.remote_output is not None:
                result.update(remote_output=error.remote_output)
            if error.remote_traceback is not None:
                result.update(remote_traceback=error.remote_traceback)
        if 'status_code' not in result:
            result['status_code'] = 500

    if result.get('status') == 'error':
        result['versions'] = get_xcube_versions()

    if output_file is not None:
        with open(output_file, 'w') as fp:
            json.dump(result, fp, indent=2)
        if verbosity:
            print(f'Result written to {output_file}')
    else:
        json.dump(result, sys.stdout, indent=2)

    if result.get('status') == 'error':
        message = result.get('message',
                             'Cube generation failed.'
                             if error is None else f'{error}')
        click_error = click.ClickException(message)
        raise click_error from error


if __name__ == '__main__':
    gen2()
