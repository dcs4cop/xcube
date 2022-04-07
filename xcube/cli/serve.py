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

import warnings
from typing import List

import click

from xcube.constants import LOG
from xcube.cli.common import (cli_option_quiet,
                              cli_option_verbosity,
                              configure_cli_output)
from xcube.webapi.defaults import (DEFAULT_PORT,
                                   DEFAULT_ADDRESS,
                                   DEFAULT_UPDATE_PERIOD,
                                   DEFAULT_TILE_CACHE_SIZE,
                                   DEFAULT_TILE_COMP_MODE)

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

VIEWER_ENV_VAR = 'XCUBE_VIEWER_PATH'
CONFIG_ENV_VAR = 'XCUBE_SERVE_CONFIG_FILE'
BASE_ENV_VAR = 'XCUBE_SERVE_BASE_DIR'


@click.command(name='serve')
@click.argument('cube', nargs=-1)
@click.option('--address', '-A', metavar='ADDRESS', default=DEFAULT_ADDRESS,
              help=f'Service address. Defaults to {DEFAULT_ADDRESS!r}.')
@click.option('--port', '-P', metavar='PORT', default=DEFAULT_PORT, type=int,
              help=f'Port number where the service will listen on. Defaults to {DEFAULT_PORT}.')
@click.option('--prefix', metavar='PREFIX',
              help='Service URL prefix. May contain template patterns such as "${version}" or "${name}". '
                   'For example "${name}/api/${version}". Will be used to prefix all API operation routes '
                   'and in any URLs returned by the service.')
@click.option('--revprefix', 'reverse_prefix', metavar='REVPREFIX',
              help='Service reverse URL prefix. May contain template patterns such as "${version}" or "${name}". '
                   'For example "${name}/api/${version}". Defaults to PREFIX, if any. Will be used only in URLs '
                   'returned by the service e.g. the tile URLs returned by the WMTS service.')
@click.option('--update', '-u', 'update_period', metavar='PERIOD', type=float,
              default=DEFAULT_UPDATE_PERIOD,
              help='Service will update after given seconds of inactivity. Zero or a negative value will '
                   'disable update checks. '
                   f'Defaults to {DEFAULT_UPDATE_PERIOD!r}.')
@click.option('--styles', '-S', metavar='STYLES', default=None,
              help='Color mapping styles for variables. '
                   'Used only, if one or more CUBE arguments are provided and CONFIG is not given. '
                   'Comma-separated list with elements of the form '
                   '<var>=(<vmin>,<vmax>) or <var>=(<vmin>,<vmax>,"<cmap>")')
@click.option('--config', '-c', 'config_file', metavar='CONFIG', default=None,
              help='Use datasets configuration file CONFIG. '
                   'Cannot be used if CUBES are provided. '
                   'If not given and also CUBES are not provided, '
                   f'the configuration may be given by environment variable {CONFIG_ENV_VAR}.')
@click.option('--base-dir', '-b', 'base_dir', metavar='BASE_DIR', default=None,
              help='Base directory used to resolve relative dataset paths in CONFIG '
                   'and relative CUBES paths. '
                   f'Defaults to value of environment variable {BASE_ENV_VAR}, if given, '
                   'otherwise defaults to the parent directory of CONFIG.')
@click.option('--tilecache', 'tile_cache_size', metavar='SIZE', default=DEFAULT_TILE_CACHE_SIZE,
              help=f'In-memory tile cache size in bytes. '
                   f'Unit suffixes {"K"!r}, {"M"!r}, {"G"!r} may be used. '
                   f'Defaults to {DEFAULT_TILE_CACHE_SIZE!r}. '
                   f'The special value {"OFF"!r} disables tile caching.')
@click.option('--tilemode', 'tile_comp_mode', metavar='MODE', default=None, type=int,
              help='Tile computation mode. '
                   'This is an internal option used to switch between different tile '
                   f'computation implementations. Defaults to {DEFAULT_TILE_COMP_MODE!r}.')
@click.option('--show', '-s', is_flag=True,
              help=f"Run viewer app. Requires setting the environment variable {VIEWER_ENV_VAR} "
                   f"to a valid xcube-viewer deployment or build directory. "
                   f"Refer to https://github.com/dcs4cop/xcube-viewer for more information.")
@cli_option_quiet
@cli_option_verbosity
@click.option('--traceperf', 'trace_perf', is_flag=True,
              help="Print performance diagnostics (stdout).")
@click.option('--aws-prof', 'aws_prof', metavar='PROFILE',
              help="To publish remote CUBEs, use AWS credentials from section "
                   "[PROFILE] found in ~/.aws/credentials.")
@click.option('--aws-env', 'aws_env', is_flag=True,
              help="To publish remote CUBEs, use AWS credentials from environment "
                   "variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
def serve(cube: List[str],
          address: str,
          port: int,
          prefix: str,
          reverse_prefix: str,
          update_period: float,
          styles: str,
          config_file: str,
          base_dir: str,
          tile_cache_size: str,
          tile_comp_mode: int,
          show: bool,
          quiet: bool,
          verbosity: int,
          trace_perf: bool,
          aws_prof: str,
          aws_env: bool):
    """
    Serve data cubes via web service.

    Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP interface.
    The RESTful API documentation can be found at https://app.swaggerhub.com/apis/bcdev/xcube-server.
    """

    configure_cli_output(quiet=quiet, verbosity=verbosity)

    from xcube.cli.common import parse_cli_kwargs
    import os.path

    if config_file and cube:
        raise click.ClickException("CONFIG and CUBES cannot be used at the same time.")
    if not config_file and not cube:
        config_file = os.environ.get(CONFIG_ENV_VAR)
    if styles:
        styles = parse_cli_kwargs(styles, "STYLES")
    if (aws_prof or aws_env) and not cube:
        raise click.ClickException(
            "AWS credentials are only valid in combination with given CUBE argument(s).")
    if config_file and not os.path.isfile(config_file):
        raise click.ClickException(
            f"Configuration file not found: {config_file}")

    base_dir = base_dir or os.environ.get(BASE_ENV_VAR,
                                          config_file and os.path.dirname(
                                              config_file)) or '.'
    if not os.path.isdir(base_dir):
        raise click.ClickException(f"Base directory not found: {base_dir}")

    from xcube.version import version
    from xcube.webapi.defaults import SERVER_NAME, SERVER_DESCRIPTION
    LOG.info(f'{SERVER_NAME}: {SERVER_DESCRIPTION}, version {version}')

    if show:
        _run_viewer()

    from xcube.webapi.app import new_application
    application = new_application(route_prefix=prefix, base_dir=base_dir)

    from xcube.webapi.service import Service
    service = Service(application,
                      prefix=reverse_prefix or prefix,
                      port=port,
                      address=address,
                      cube_paths=cube,
                      styles=styles,
                      config_file=config_file,
                      base_dir=base_dir,
                      tile_cache_size=tile_cache_size,
                      tile_comp_mode=tile_comp_mode,
                      update_period=update_period,
                      trace_perf=trace_perf,
                      aws_prof=aws_prof,
                      aws_env=aws_env)
    service.start()

    return 0


def _run_viewer():
    import subprocess
    import threading
    import webbrowser
    import os

    viewer_dir = os.environ.get(VIEWER_ENV_VAR)

    if viewer_dir is None:
        raise click.UsageError('Option "--show": '
                               f"In order to run the viewer, "
                               f"set environment variable {VIEWER_ENV_VAR} "
                               f"to a valid xcube-viewer deployment or build directory.")

    if not os.path.isdir(viewer_dir):
        raise click.UsageError('Option "--show": '
                               f"Viewer path set by environment variable {VIEWER_ENV_VAR} "
                               f"must be a directory: " + viewer_dir)

    def _run():
        LOG.info("Starting web server...")
        with subprocess.Popen(['python', '-m', 'http.server', '--directory', viewer_dir],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE):
            LOG.info("Opening viewer...")
            webbrowser.open("http://localhost:8000/index.html")

    threading.Thread(target=_run, name="xcube-viewer-runner").start()


def main(args=None):
    serve.main(args=args)


if __name__ == '__main__':
    main()
