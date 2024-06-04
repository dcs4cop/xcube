# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import os
import socket
import threading
from pathlib import Path
from typing import Optional, Union, Any, Tuple, Dict
from collections.abc import Iterable
from collections.abc import Mapping

import fsspec
import tornado.ioloop
import xarray as xr

from xcube.constants import LOG
from xcube.core.mldataset import MultiLevelDataset
from xcube.server.server import Server
from xcube.server.webservers.tornado import TornadoFramework
from xcube.webapi.datasets.context import DatasetsContext


# Name of the env var that contains a JupyterLab's base URL.
# If this env var is set, the following _LAB_INFO_FILE will not be used.
# We also imply that a jupyter-server-proxy is installed and enabled.
_LAB_URL_ENV_VAR = "XCUBE_JUPYTER_LAB_URL"

# The following file is generated by the xcube JupyterLab extension
# (xcube-jl-ext) if it is installed and enabled in JupyterLab.
_LAB_INFO_FILE = "~/.xcube/jupyterlab/lab-info.json"


_DEFAULT_MAX_DEPTH = 1


class Viewer:
    """xcube Viewer for Jupyter Notebooks.

    The viewer can be used to visualise and inspect datacubes
    with at least one data variable with dimensions ``["time", "lat", "lon"]``
    or, if a grid mapping is present, with arbitrary ``"time"`` and
    arbitrarily x- and y-dimensions, e.g., ``["time", "y", "x"]`` .

    Add datacubes from instances of ``xarray.Dataset``:

    ```
    viewer = Viewer()
    viewer.add_dataset(dataset)  # can set color styles here too, see doc below
    viewer.show()
    ```

    Display all datasets of formats Zarr, NetCDF, COG/GeoTIFF found in the
    given directories in the local filesystem or in a given S3 bucket:

    ```
    viewer = Viewer(roots=["/eodata/smos/l2", "s3://xcube/examples"])
    viewer.show()
    ```

    The `Viewer` class takes a xcube server configuration as first
    argument. More details regarding configuration parameters are given in the
    `server documentation <https://xcube.readthedocs.io/en/latest/cli/xcube_serve.html>`_.
    The full configuration reference can be generated by excecuting CLI command
    ``$ xcube serve --show configschema``.

    Args:
        server_config: Server configuration.
            See also output of ``$ xcube serve --show configschema``.
        roots: A path or URL or an iterable of paths or URLs
            that will each be scanned for datasets to be shown in the viewer.
        max_depth: defines the maximum subdirectory depth used to
            search for datasets in case *roots* is given.
    """

    def __init__(
        self,
        server_config: Optional[Mapping[str, Any]] = None,
        roots: Optional[Union[str, Iterable[str]]] = None,
        max_depth: Optional[int] = None,
    ):
        self._server_config, server_url = _get_server_config(
            server_config=server_config, roots=roots, max_depth=max_depth
        )
        self._server_url = server_url
        self._viewer_url = f"{server_url}/viewer/?serverUrl={server_url}"

        # Got trick from
        # https://stackoverflow.com/questions/55201748/running-a-tornado-server-within-a-jupyter-notebook
        self._io_loop = tornado.ioloop.IOLoop()
        thread = threading.Thread(target=self._io_loop.start)
        thread.daemon = True
        thread.start()

        self._server = Server(
            TornadoFramework(io_loop=self._io_loop, shared_io_loop=True),
            config=self._server_config,
        )

        self._io_loop.add_callback(self._server.start)

    @property
    def server_config(self) -> Mapping[str, Any]:
        """The server configuration used by this viewer."""
        return self._server_config

    @property
    def server_url(self):
        """The URL of the server used by this viewer."""
        return self._server_url

    @property
    def viewer_url(self):
        """The URL of this viewer."""
        return self._viewer_url

    @property
    def is_server_running(self) -> bool:
        """Whether the server is running."""
        return self._server is not None

    @property
    def datasets_ctx(self) -> DatasetsContext:
        """Gets the context for the server's "datasets" API."""
        assert self.is_server_running
        return self._server.ctx.get_api_ctx("datasets")

    def stop_server(self):
        """Stops this viewer's server."""
        if self._server is not None:
            # noinspection PyBroadException
            try:
                self._server.stop()
            except BaseException:
                pass
        self._server = None
        self._io_loop = None

    def add_dataset(
        self,
        dataset: Union[xr.Dataset, MultiLevelDataset],
        ds_id: Optional[str] = None,
        title: Optional[str] = None,
        style: Optional[str] = None,
        color_mappings: dict[str, dict[str, Any]] = None,
    ):
        """Add a dataset to this viewer.

        Args:
            dataset: The dataset to me added. Must be an instance of
                ``xarray.Dataset`` or
                ``xcube.core.mldataset.MultiLevelDataset``.
            ds_id: Optional dataset identifier. If not given, an
                identifier will be generated and returned.
            title: Optional dataset title. Overrides a title given by
                dataset metadata.
            style: Optional name of a style that must exist in the
                server configuration.
            color_mappings: Maps a variable name to a specific color
                mapping that is a dictionary comprising a "ValueRange"
                (a pair of numbers) and a "ColorBar" (a matplotlib color
                bar name).

        Returns:
            The dataset identifier.
        """
        if not self._check_server_running():
            return
        return self.datasets_ctx.add_dataset(
            dataset,
            ds_id=ds_id,
            title=title,
            style=style,
            color_mappings=color_mappings,
        )

    def remove_dataset(self, ds_id: str):
        """Remove a dataset from this viewer.

        Args:
            ds_id: The identifier of the dataset to be removed.
        """
        if not self._check_server_running():
            return
        self.datasets_ctx.remove_dataset(ds_id)

    def show(self, width: Union[int, str] = "100%", height: Union[str, int] = 800):
        """Show this viewer as an iframe.
        Intended to be used in a Jupyter notebook.
        If used outside a Jupyter notebook the viewer will be shown
        as a new browser tab.

        Args:
            width: The width of the viewer's iframe.
            height: The height of the viewer's iframe.
        """
        try:
            # noinspection PyPackageRequirements
            from IPython.core.display import HTML

            return HTML(
                f'<iframe src="{self._viewer_url}&compact=1"'
                f' width="{width}"'
                f' height="{height}"'
                f"/>"
            )
        except ImportError as e:
            print(f"Error: {e}; Trying to open Viewer in web browser...")
            # noinspection PyBroadException
            try:
                import webbrowser

                webbrowser.open_new_tab(self.viewer_url)
            except BaseException:
                print("Failed too.")

    def info(self):
        """Output viewer info."""
        # Consider outputting this as HTML if in Notebook
        print(f"Server: {self.server_url}")
        print(f"Viewer: {self.viewer_url}")

    def _check_server_running(self):
        if not self.is_server_running:
            print("Server not running")
        return self.is_server_running


def _get_server_config(
    server_config: Optional[Mapping[str, Any]] = None,
    roots: Optional[Union[str, Iterable[str]]] = None,
    max_depth: Optional[int] = None,
) -> tuple[dict[str, Any], str]:
    server_config = dict(server_config or {})
    max_depth = max_depth or _DEFAULT_MAX_DEPTH

    port = server_config.get("port")
    address = server_config.get("address")

    if port is None:
        port = _find_port()
    if address is None:
        address = "0.0.0.0"

    server_config["port"] = port
    server_config["address"] = address

    server_url, reverse_url_prefix = _get_server_url_and_rev_prefix(port)
    server_config["reverse_url_prefix"] = reverse_url_prefix

    if roots is not None:
        roots = [roots] if isinstance(roots, str) else roots
        config_stores = list(server_config.get("DataStores", []))
        root_stores = _get_data_stores_from_roots(roots, max_depth)
        server_config["DataStores"] = config_stores + root_stores

    return server_config, server_url


def _get_server_url_and_rev_prefix(port: int) -> tuple[str, str]:
    lab_url = os.environ.get(_LAB_URL_ENV_VAR) or None
    has_proxy = lab_url is not None

    if not lab_url:
        lab_info_path = Path(*_LAB_INFO_FILE.split("/")).expanduser()
        if lab_info_path.exists():
            try:
                with lab_info_path.open() as fp:
                    lab_info = json.load(fp)
                lab_url = lab_info["lab_url"]
                has_proxy = lab_info["has_proxy"]
            except (OSError, KeyError):
                LOG.warning(f"Failed loading {lab_info_path}")
                pass

    if lab_url and has_proxy:
        reverse_prefix = f"/proxy/{port}"
        if lab_url.endswith("/"):
            lab_url = lab_url[:-1]
        return lab_url + reverse_prefix, reverse_prefix

    return f"http://localhost:{port}", ""


def _find_port(start: int = 8000, end: Optional[int] = None) -> int:
    """Find a port not in use in range *start* to *end*"""
    end = end if isinstance(end, int) and end >= start else start + 12000
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No available port found")


def _get_data_stores_from_roots(
    roots: Iterable[str], max_depth: int
) -> list[dict[str, dict]]:
    extra_data_stores = []
    for index, root in enumerate(roots):
        protocol, path = fsspec.core.split_protocol(root)
        extra_data_stores.append(
            {
                "Identifier": f"_root_{index}",
                "StoreId": protocol or "file",
                "StoreParams": {"root": path, "max_depth": max_depth},
            }
        )
    return extra_data_stores
