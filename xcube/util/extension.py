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


import importlib
from typing import List, Any, Dict, Callable, Mapping, Sequence, Optional

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

from xcube.util.ipython import register_json_formatter

Component = Any
ComponentLoader = Callable[["Extension"], Component]
ComponentTransform = Callable[[Component, "Extension"], Component]
ExtensionPredicate = Callable[["Extension"], bool]


class Extension:
    """An extension that provides a component of any type.

    Extensions are registered in a `ExtensionRegistry`.

    Extension objects are not meant to be instantiated directly. Instead,
    `ExtensionRegistry.add_extension()` is used to register extensions.

    Args:
        point: extension point identifier
        name: extension name
        component: extension component
        loader: extension component loader function
        metadata: extension metadata
    """

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        point: str,
        name: str,
        component: Component = None,
        loader: ComponentLoader = None,
        **metadata,
    ):
        if point is None:
            raise ValueError(f"point must be given")
        if name is None:
            raise ValueError(f"name must be given")
        if (loader is not None and component is not None) or (
            loader is None and component is None
        ):
            raise ValueError(f"either component or loader must be given")
        if loader is not None and not callable(loader):
            raise ValueError(f"loader must be callable")

        self._component = component
        self._loader = loader
        self._point = point
        self._name = name
        self._metadata = metadata
        self._deleted = False

    @property
    def is_lazy(self) -> bool:
        """Whether this is a lazy extension that uses a loader."""
        return self._loader is not None

    @property
    def component(self) -> Component:
        """Extension component."""
        if self._component is None and self._loader is not None:
            self._component = self._loader(self)
        return self._component

    @property
    def point(self) -> str:
        """Extension point identifier."""
        return self._point

    @property
    def name(self) -> str:
        """Extension name."""
        return self._name

    @property
    def metadata(self) -> Dict[str, Any]:
        """Extension metadata."""
        return dict(self._metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Get a JSON-serializable dictionary representation of this extension."""

        # Note: we avoid loading the component!
        if self._component is not None:
            if hasattr(self._component, "to_dict") and callable(
                getattr(self._component, "to_dict")
            ):
                component = self._component.to_dict()
            else:
                component = repr(self._component)
        else:
            component = "<not loaded yet>"
        d = dict(
            name=self.name,
            **self.metadata,
            point=self.point,
            component=component,
        )
        return d


register_json_formatter(Extension)


# noinspection PyShadowingBuiltins
class ExtensionRegistry:
    """A registry of extensions.
    Typically used by plugins to register extensions.
    """

    def __init__(self):
        self._extension_points = {}

    def has_extension(self, point: str, name: str) -> bool:
        """Test if an extension with given *point* and *name* is registered.

        Args:
            point: extension point identifier
            name: extension name

        Returns:
            True, if extension exists
        """
        return point in self._extension_points and name in self._extension_points[point]

    def get_extension(self, point: str, name: str) -> Optional[Extension]:
        """Get registered extension for given *point* and *name*.

        Args:
            point: extension point identifier
            name: extension name

        Returns:
            the extension or None, if no such exists
        """
        if point not in self._extension_points:
            return None
        return self._extension_points[point].get(name)

    def get_component(self, point: str, name: str) -> Any:
        """Get extension component for given *point* and *name*.
        Raises a ValueError if no such extension exists.

        Args:
            point: extension point identifier
            name: extension name

        Returns:
            extension component
        """
        extension = self.get_extension(point, name)
        if extension is None:
            raise ValueError(
                f"extension {name!r} not found for extension point {point!r}"
            )
        return extension.component

    def find_extensions(
        self, point: str, predicate: ExtensionPredicate = None
    ) -> List[Extension]:
        """Find extensions for *point* and optional filter function *predicate*.

        The filter function is called with an extension and should return
        a truth value to indicate a match or mismatch.

        Args:
            point: extension point identifier
            predicate: optional filter function

        Returns:
            list of matching extensions
        """
        if point not in self._extension_points:
            return []
        point_extensions = self._extension_points[point]
        if predicate is None:
            return list(point_extensions.values())
        return [
            extension for extension in point_extensions.values() if predicate(extension)
        ]

    def find_components(
        self, point: str, predicate: ExtensionPredicate = None
    ) -> List[Component]:
        """Find extension components for *point* and optional filter function *predicate*.

        The filter function is called with an extension and should return
        a truth value to indicate a match or mismatch.

        Args:
            point: extension point identifier
            predicate: optional filter function

        Returns:
            list of matching extension components
        """
        return [
            extension.component
            for extension in self.find_extensions(point, predicate=predicate)
        ]

    def add_extension(
        self,
        point: str,
        name: str,
        component: Component = None,
        loader: ComponentLoader = None,
        **metadata,
    ) -> Extension:
        """Register an extension *component* or an extension component *loader* for
        the given extension *point*, *name*, and additional *metadata*.

        Either *component* or *loader* must be specified, but not both.

        A given *loader* must be a callable with one positional argument *extension* of type `Extension`
        and is expected to return the actual extension component, which may be of any type.
        The *loader* will only be called once and only when the actual extension component
        is requested for the first time. Consider using the function `import_component()` to create a
        loader that lazily imports a component from a module and optionally executes it.

        Args:
            point: extension point identifier
            name: extension name
            component: extension component
            loader: extension component loader function
            **metadata: extension metadata

        Returns:
            a registered extension
        """
        extension = Extension(
            point, name, component=component, loader=loader, **metadata
        )
        if point in self._extension_points:
            self._extension_points[point][name] = extension
        else:
            self._extension_points[point] = {name: extension}
        return extension

    def remove_extension(self, point: str, name: str):
        """Remove registered extension *name* from given *point*.

        Args:
            point: extension point identifier
            name: extension name
        """
        point_extensions = self._extension_points[point]
        del point_extensions[name]

    def to_dict(self):
        """Get a JSON-serializable dictionary representation of this extension registry."""
        return {
            k: {ek: ev.to_dict() for ek, ev in v.items()}
            for k, v in self._extension_points.items()
        }


register_json_formatter(ExtensionRegistry)

_EXTENSION_REGISTRY_SINGLETON = ExtensionRegistry()


def get_extension_registry() -> ExtensionRegistry:
    """Return the extension registry singleton."""
    return _EXTENSION_REGISTRY_SINGLETON


def import_component(
    spec: str,
    transform: ComponentTransform = None,
    call: bool = False,
    call_args: Sequence[Any] = None,
    call_kwargs: Mapping[str, Any] = None,
) -> ComponentLoader:
    """Return a component loader that imports a module or module component from *spec*.
    To import a module, *spec* should be the fully qualified module name. To import a
    component, *spec* must also append the component name to the fully qualified module name
    separated by a color (":") character.

    An optional *transform* callable my be used to transform the imported component. If given,
    a new component is computed::

        component = transform(component, extension)

    If the *call* flag is set, the component is expected to be a callable which will be called
    using the given *call_args* and *call_kwargs* to produce a new component::

        component = component(*call_kwargs, **call_kwargs)

    Finally, the component is returned.

    Args:
        spec: String of the form "module_path" or
            "module_path:component_name"
        transform: callable that takes two positional arguments, the
            imported component and the extension of type
            `Extension`
        call: Whether to finally call the component with given
            *call_args* and *call_kwargs*
        call_args: arguments passed to a callable component if *call*
            flag is set
        call_kwargs: keyword arguments passed to callable component if
            *call* flag is set

    Returns:
        a component loader
    """

    # noinspection PyUnusedLocal
    def _load(extension: Extension):
        nonlocal spec
        component = _import_component(spec, force_component=call)
        if transform is not None:
            component = transform(component, extension)
        if call or call_args or call_kwargs:
            component = component(*(call_args or []), **(call_kwargs or {}))
        return component

    return _load


def _import_component(component_spec: str, force_component: bool = False):
    """Import a module or module component from *spec*.

    Args:
        component_spec: String of the form "module_name" or
            "module_name:component_name" where module_name must an
            absolute, fully qualified path to a module.
        force_component: If True, *spec* must specify a component name

    Returns:
        the imported module or module component
    """
    if ":" in component_spec:
        module_name, component_name = component_spec.split(":", maxsplit=1)
    else:
        module_name, component_name = component_spec, None

    if force_component and component_name is None:
        raise ValueError("illegal spec, must specify a component")
    if module_name == "":
        raise ValueError("illegal spec, missing module path")
    if component_name == "":
        raise ValueError("illegal spec, missing component name")

    module = importlib.import_module(module_name)
    return getattr(module, component_name) if component_name else module
