# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import abc
import importlib
import pkgutil
import sys
import time
import traceback
import warnings
from typing import Callable, Dict, Optional, Any, List

from pkg_resources import iter_entry_points

from xcube.constants import PLUGIN_ENTRY_POINT_GROUP_NAME
from xcube.constants import PLUGIN_INIT_TIME__WARN_LIMIT
from xcube.constants import PLUGIN_LOAD_TIME_WARN_LIMIT
from xcube.constants import PLUGIN_MODULE_FUNCTION_NAME
from xcube.constants import PLUGIN_MODULE_NAME
from xcube.constants import PLUGIN_MODULE_PREFIX
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionRegistry

_PLUGIN_REGISTRY_INIT = dict()

# Mapping of xcube entry point names
# to JSON-serializable plugin meta-information.
_PLUGIN_REGISTRY: Dict[str, Dict[str, Any]] = _PLUGIN_REGISTRY_INIT


def init_plugins() -> None:
    """Load plugins if not already done."""
    global _PLUGIN_REGISTRY
    if _PLUGIN_REGISTRY is _PLUGIN_REGISTRY_INIT:
        _PLUGIN_REGISTRY = load_plugins()


def get_plugins() -> Dict[str, Dict]:
    """Get mapping of "xcube_plugins" entry point names
    to JSON-serializable plugin meta-information.
    """
    init_plugins()
    global _PLUGIN_REGISTRY
    return dict(_PLUGIN_REGISTRY)


def get_extension_registry() -> ExtensionRegistry:
    """Get populated extension registry."""
    from xcube.util.extension import get_extension_registry

    init_plugins()
    return get_extension_registry()


def discover_plugin_modules(module_prefixes=None):
    module_prefixes = module_prefixes or [PLUGIN_MODULE_PREFIX]
    entry_points = []
    for module_finder, module_name, ispkg in pkgutil.iter_modules():
        if any(
            [module_name.startswith(module_prefix) for module_prefix in module_prefixes]
        ):
            # Note: Consider turning this into debug log,
            #  but logging is not yet configured at this point.
            # print(f'xcube plugin module found: {module_name}')
            entry_points.append(_ModuleEntryPoint(module_name))
    return entry_points


def load_plugins(
    entry_points=None, ext_registry: Optional[ExtensionRegistry] = None
) -> Dict[str, Dict[str, Any]]:
    if ext_registry is None:
        from xcube.util.extension import get_extension_registry

        ext_registry = get_extension_registry()

    plugins = {}

    if entry_points:
        _collect_plugins(entry_points, ext_registry, True, plugins)
    else:
        # verbose=True for specified xcube plugins.
        _collect_plugins(
            list(iter_entry_points(group=PLUGIN_ENTRY_POINT_GROUP_NAME, name=None)),
            ext_registry,
            True,
            plugins,
        )
        # verbose=False for auto-detected xcube plugins,
        # where package name starts with "xcube_" but an
        # entry point is not explicitly specified.
        _collect_plugins(discover_plugin_modules(), ext_registry, False, plugins)

    return plugins


def _collect_plugins(
    entry_points: List[Any],
    ext_registry: ExtensionRegistry,
    verbose: bool,
    plugins: Dict[str, Dict[str, Any]],
):
    for entry_point in entry_points:
        # Note: Consider turning this into debug log,
        #  but logging is not yet configured at this point.
        #  print(f'loading xcube plugin {entry_point.name!r}')

        t0 = time.perf_counter()

        # noinspection PyBroadException
        try:
            plugin_init_function = entry_point.load()
        except Exception as e:
            if verbose:
                _emit_warning_for_error(entry_point, e)
            continue

        if plugin_init_function is None:
            # Not a plugin
            continue

        if verbose:
            millis = int(1000 * (time.perf_counter() - t0))
            if millis >= PLUGIN_LOAD_TIME_WARN_LIMIT:
                _emit_warning_for_slow_load(entry_point, millis)

        if not callable(plugin_init_function):
            if verbose:
                _emit_warning_on_init_function(entry_point, plugin_init_function)
            continue

        t0 = time.perf_counter()

        # noinspection PyBroadException
        try:
            plugin_init_function(ext_registry)
        except Exception as e:
            if verbose:
                _emit_warning_for_error(entry_point, e)
            continue

        if verbose:
            millis = int(1000 * (time.perf_counter() - t0))
            if millis >= PLUGIN_INIT_TIME__WARN_LIMIT:
                _emit_warning_for_slow_load(entry_point, millis)

        plugins[entry_point.name] = {
            "name": entry_point.name,
            "doc": plugin_init_function.__doc__,
        }


def _emit_warning_on_init_function(entry_point, plugin_init_function):
    # We use warning and not raise to allow loading
    # xcube despite a broken plugin. Raise would stop xcube.
    warnings.warn(
        f"xcube plugin {entry_point.name!r}"
        f" must be callable"
        f" but got a {type(plugin_init_function)!r}"
    )


def _emit_warning_for_slow_load(entry_point, millis):
    warnings.warn(
        f"Initializing xcube plugin {entry_point.name!r}"
        f" took {millis} ms,"
        f" consider code optimization."
        f" (For example, avoid eager import of packages,"
        f" consider lazy loading of resources, etc.)"
    )


def _emit_warning_for_error(entry_point, e):
    # We use warning and not raise to allow loading
    # xcube despite a broken plugin. Raise would stop xcube.
    warnings.warn(
        f"Unexpected exception while loading" f" xcube plugin {entry_point.name!r}: {e}"
    )
    traceback.print_exc(file=sys.stderr)


class _ModuleEntryPoint:
    def __init__(self, module_name: str):
        self._module_name = module_name

    @property
    def name(self) -> str:
        return self._module_name

    def load(self) -> Optional[Callable]:
        """Load function "init_plugin()" either from "<module_name>.plugin" or "<module_name>"

        Returns:
            plugin init function
        """
        try:
            plugin_module = importlib.import_module(
                f"{self._module_name}.{PLUGIN_MODULE_NAME}"
            )
        except ModuleNotFoundError:
            plugin_module = importlib.import_module(self._module_name)

        module_func_name = PLUGIN_MODULE_FUNCTION_NAME
        if hasattr(plugin_module, module_func_name):
            module_func = getattr(plugin_module, module_func_name)
            if callable(module_func):
                return module_func

        # Here: The module name looks like an xcube-plugin
        # but lacks a callable named module_func_name

        return None


class ExtensionComponent(metaclass=abc.ABCMeta):
    """Utility base class for extension components."""

    def __init__(self, extension_point: str, name: str):
        if not extension_point:
            raise ValueError("extension_point must be given")
        if not name:
            raise ValueError("name must be given")
        self._extension_point = extension_point
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def extension_point(self) -> str:
        """Returns:
        The extension point for this component.
        """
        return self._extension_point

    @property
    def extension(self) -> Optional[Extension]:
        """Returns:
        The extension for this component. None, if it has not (yet)
        been registered as an extension.
        """
        return get_extension_registry().get_extension(self._extension_point, self._name)

    def get_metadata_attr(self, key: str, default: Any = None) -> Any:
        """Returns:
        A metadata attribute for *key* and given *default* value.
        """
        extension = self.extension
        return extension.metadata.get(key, default) if extension is not None else ""
