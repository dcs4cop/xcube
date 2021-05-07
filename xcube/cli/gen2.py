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

import sys
from typing import Sequence

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
@click.option('--verbose', '-v',
              is_flag=True,
              multiple=True,
              help='Control amount of information dumped to stdout.'
                   ' May be given multiple time, e.g. "-vvv".')
def gen2(request_path: str,
         stores_config_path: str = None,
         service_config_path: str = None,
         info: bool = False,
         verbose: Sequence[bool] = None):
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

    """
    from xcube.core.gen2 import CubeGenerator
    from xcube.core.gen2 import CubeGeneratorError
    from xcube.core.gen2 import CubeInfo
    from xcube.core.store import DataStoreError

    verbosity = len(verbose) if verbose else 0

    try:
        generator = CubeGenerator.from_file(request_path,
                                            stores_config_path=stores_config_path,
                                            service_config_path=service_config_path,
                                            verbosity=verbosity)
        if info:
            def dump_cube_info(cube_info: CubeInfo):
                import sys
                import json
                json.dump(cube_info.to_dict(), sys.stdout, indent=2)

            dump_cube_info(generator.get_cube_info())
        else:
            generator.generate_cube()

    except CubeGeneratorError as e:
        print_kwargs = dict(file=sys.stderr, flush=True)
        if e.remote_output:
            print(**print_kwargs)
            print('Remote output:', **print_kwargs)
            print('==============', **print_kwargs)
            for line in e.remote_output:
                print(line, file=sys.stderr, flush=True)
        if e.remote_traceback:
            print(**print_kwargs)
            print('Remote traceback:', **print_kwargs)
            print('=================', **print_kwargs)
            print(e.remote_traceback, file=sys.stderr)
        raise click.ClickException(f'{e}') from e

    except DataStoreError as e:
        raise click.ClickException(f'{e}') from e


if __name__ == '__main__':
    gen2()
